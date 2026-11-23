"""Image alignment of mammograms and realignment of images and coordinates."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from mammo_aligner.metadata import (
    FovHorizontalFlip,
    FovRotation,
    Laterality,
    Manufacturer,
    PatientOrientation,
    ProjectionSide,
)

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


class Edge(Enum):
    """An edge of the displayed image."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    UNKNOWN = "unknown"


_HORIZONTAL = {Edge.LEFT, Edge.RIGHT}
_VERTICAL = {Edge.TOP, Edge.BOTTOM}
_TRANSPOSED = {Edge.LEFT: Edge.TOP, Edge.TOP: Edge.LEFT, Edge.RIGHT: Edge.BOTTOM, Edge.BOTTOM: Edge.RIGHT}


@dataclass(frozen=True)
class Alignment:
    """Where the anatomy of a mammogram sits in its pixel array.

    ``chest_wall`` is the image edge holding the chest wall. ``superior`` is the
    image edge holding the superior side of the breast; for CC-type views,
    which have no superior/inferior axis, it holds the lateral side instead.

    The reference alignment is chest wall on the left and superior (lateral for
    CC) at the top. Realignment flips, and transposes when the source and
    target differ by a quarter turn (chest wall on a top or bottom edge).
    """

    chest_wall: Edge
    superior: Edge

    def __post_init__(self) -> None:
        object.__setattr__(self, "chest_wall", Edge(self.chest_wall))
        object.__setattr__(self, "superior", Edge(self.superior))
        for axis in (_HORIZONTAL, _VERTICAL):
            if self.chest_wall in axis and self.superior in axis:
                raise ValueError(f"chest_wall and superior must be on different axes: {self!r}")

    def __str__(self) -> str:
        return f"chest wall {self.chest_wall.value}, superior {self.superior.value}"

    # Construction -----------------------------------------------------------

    @classmethod
    def reference(cls) -> "Alignment":
        """Chest wall on the left, superior (lateral for CC) at the top."""

        return cls(Edge.LEFT, Edge.TOP)

    @classmethod
    def unknown(cls) -> "Alignment":
        """An alignment the metadata does not determine; it cannot be realigned."""

        return cls(Edge.UNKNOWN, Edge.UNKNOWN)

    @classmethod
    def from_orientation(cls, orientation: Any, laterality: Any = None) -> "Alignment":
        """Infer alignment from DICOM Patient Orientation and image laterality.

        Each orientation value is read by its principal (first) letter: A/P
        locate the chest wall and H/F the superior side. Laterality is only
        needed when a value is plain L/R (CC-type views), to tell lateral
        from medial.
        """

        parsed = PatientOrientation.coerce(orientation)
        side = Laterality.coerce(laterality)

        edges = {}
        for axis, component in (("x", parsed.row), ("y", parsed.column)):
            anatomy = _anatomy(component, side)
            if anatomy is None:
                continue
            field, edge = _EDGE_TOWARD[axis, anatomy]
            if field in edges:
                return cls.unknown()
            edges[field] = edge
        return cls(edges.get("chest_wall", Edge.UNKNOWN), edges.get("superior", Edge.UNKNOWN))

    @classmethod
    def from_fov(
        cls,
        rotation: Any,
        horizontal_flip: Any,
        laterality: Any,
        view_position: Any,
        manufacturer: Any,
    ) -> "Alignment":
        """Infer alignment from DICOM Field of View Rotation and Horizontal Flip.

        Field of View values describe the image relative to the detector, so
        the result depends on the vendor's detector frame and on which side
        of the breast the view is taken from. Only Hologic and GE, and views
        checked against Patient Orientation, are supported; anything else
        gives an unknown alignment. Prefer :meth:`from_orientation` when
        Patient Orientation is available.
        """

        key = (
            Laterality.coerce(laterality),
            FovRotation.coerce(rotation),
            FovHorizontalFlip.coerce(horizontal_flip),
        )
        edges = _FOV_ALIGNMENT.get(key)
        vendor = Manufacturer.coerce(manufacturer)
        side = ProjectionSide.coerce(view_position)
        if not edges or vendor is Manufacturer.UNKNOWN or side is ProjectionSide.UNKNOWN:
            return cls.unknown()

        chest_wall, superior = edges
        if vendor is Manufacturer.GE:  # detector frame is rotated 180° relative to Hologic
            chest_wall, superior = _OPPOSITE[chest_wall], _OPPOSITE[superior]
        if side is ProjectionSide.REVERSE:  # detector on the other side mirrors top to bottom
            superior = _OPPOSITE[superior]
        return cls(chest_wall, superior)

    @classmethod
    def from_metadata(cls, record: Any) -> "Alignment":
        """Infer alignment from a pydicom ``Dataset``, a dict, or a pandas row.

        Uses Patient Orientation first and falls back to the Field of View
        attributes. Laterality is read from ``ImageLateralityFinal`` (EMBED),
        then ``ImageLaterality``, then ``Laterality``. Multi-frame objects such
        as DBT keep Patient Orientation and laterality per frame, so when the
        top-level attributes are missing they are read from the first
        shared or per-frame functional group. If no Patient Orientation exists
        at all (e.g. GE Pristina DBT), it is derived from Image Orientation
        (Patient), at the top level or in the Plane Orientation functional group.
        """

        laterality = _field(record, "ImageLateralityFinal", "ImageLaterality", "Laterality")
        if laterality is None:
            laterality = _frame_field(record, "FrameAnatomySequence", "FrameLaterality")
        orientation = _field(record, "PatientOrientation")
        if orientation is None:
            orientation = _frame_field(record, "PatientOrientationInFrameSequence", "PatientOrientation")
        if orientation is None:
            cosines = _field(record, "ImageOrientationPatient")
            if cosines is None:
                cosines = _frame_field(record, "PlaneOrientationSequence", "ImageOrientationPatient")
            if cosines is not None:
                orientation = PatientOrientation.from_cosines(cosines)

        alignment = cls.from_orientation(orientation, laterality)
        if alignment.is_known:
            return alignment
        return cls.from_fov(
            _field(record, "FieldOfViewRotation"),
            _field(record, "FieldOfViewHorizontalFlip"),
            laterality,
            _field(record, "ViewPosition"),
            _field(record, "Manufacturer"),
        )

    # Checks -----------------------------------------------------------------

    @property
    def is_known(self) -> bool:
        """Whether both edges are known, so the alignment can be realigned."""

        return Edge.UNKNOWN not in (self.chest_wall, self.superior)

    @property
    def is_reference(self) -> bool:
        """Whether this is already the reference alignment, so realignment is a no-op."""

        return self == self.reference()

    # Realignment ------------------------------------------------------------

    def realign_image(self, image: Any, target: Optional["Alignment"] = None) -> Any:
        """Realign an array whose last two axes are (rows, columns) into ``target``.

        Works on 2D images and on DBT volumes shaped (frames, rows, columns).
        ``target`` defaults to the reference alignment. When the source and
        target differ by a quarter turn, rows and columns are swapped, so the
        result has shape (..., columns, rows).
        """

        transpose, flip_x, flip_y = self._transform(target)
        if transpose:
            image = image.swapaxes(-1, -2)
        if flip_x:
            image = image[..., ::-1]
        if flip_y:
            image = image[..., ::-1, :]
        return image

    def realign_point(
        self, point: Any, *, height: int, width: int, target: Optional["Alignment"] = None
    ) -> Union[Point, List[Point]]:
        """Realign ``(y, x)`` pixel indices of a ``height`` x ``width`` image into ``target``.

        ``point`` is one ``(y, x)`` pair or a sequence of them (e.g. a list or
        an ``(N, 2)`` array). One pair gives a tuple back; a sequence gives a
        list of tuples. Raises ``ValueError`` if an item is not two numbers.
        """

        transform = self._transform(target)
        return _map_items(point, 2, "point", lambda p: _realign_yx(transform, height, width, *p))

    def realign_box(
        self, box: Any, *, height: int, width: int, target: Optional["Alignment"] = None
    ) -> Union[Box, List[Box]]:
        """Realign inclusive ``(y_min, x_min, y_max, x_max)`` pixel boxes into ``target``.

        ``box`` is one box or a sequence of them (e.g. EMBED ``ROI_coords`` or
        an ``(N, 4)`` array). One box gives a tuple back; a sequence gives a
        list of tuples. ``height`` and ``width`` are the source image's
        dimensions. Raises ``ValueError`` if an item is not four numbers or a
        minimum exceeds its maximum.
        """

        transform = self._transform(target)

        def realign(b: Sequence[Any]) -> Box:
            y_min, x_min, y_max, x_max = b
            if y_min > y_max or x_min > x_max:
                raise ValueError(f"box minimum exceeds its maximum: {tuple(b)!r}")
            ya, xa = _realign_yx(transform, height, width, y_min, x_min)
            yb, xb = _realign_yx(transform, height, width, y_max, x_max)
            return min(ya, yb), min(xa, xb), max(ya, yb), max(xa, xb)

        return _map_items(box, 4, "box", realign)

    def _transform(self, target: Optional["Alignment"]) -> Tuple[bool, bool, bool]:
        """Return (transpose, flip_x, flip_y), applied in that order, taking ``self`` to ``target``."""

        target = target or self.reference()
        for alignment in (self, target):
            if not alignment.is_known:
                raise ValueError(f"Cannot realign with an unknown alignment: {alignment}")

        source = self
        transpose = (self.chest_wall in _HORIZONTAL) is not (target.chest_wall in _HORIZONTAL)
        if transpose:  # swapping rows and columns moves each edge to its diagonal partner
            source = Alignment(_TRANSPOSED[self.chest_wall], _TRANSPOSED[self.superior])

        flip_x = flip_y = False
        for have, want in ((source.chest_wall, target.chest_wall), (source.superior, target.superior)):
            if have is not want:
                if have in _HORIZONTAL:
                    flip_x = True
                else:
                    flip_y = True
        return transpose, flip_x, flip_y


# Coordinates -----------------------------------------------------------------


def _realign_yx(transform: Tuple[bool, bool, bool], height: int, width: int, y: Any, x: Any) -> Point:
    """Apply (transpose, flip_x, flip_y) to one ``(y, x)`` pixel index."""

    transpose, flip_x, flip_y = transform
    if transpose:
        y, x, height, width = x, y, width, height
    if flip_x:
        x = width - 1 - x
    if flip_y:
        y = height - 1 - y
    return y, x


def _map_items(value: Any, length: int, kind: str, realign: Callable[[Sequence[Any]], Any]) -> Any:
    """Apply ``realign`` to one item, or to each item of a sequence of them.

    ``value`` is one item if its first element is a number, and a sequence of
    items otherwise. One item gives one result; a sequence gives a list.
    """

    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"expected a {kind} or a sequence of them, got {value!r}") from None
    single = bool(items) and isinstance(items[0], numbers.Real)
    results = []
    for item in [items] if single else items:
        try:
            values = list(item)
        except TypeError:
            values = []
        if len(values) != length or not all(isinstance(v, numbers.Real) for v in values):
            raise ValueError(f"a {kind} must be {length} numbers, got {item!r}")
        results.append(realign(values))
    return results[0] if single else results


# Patient Orientation ---------------------------------------------------------

# Which image edge each anatomical direction of +x / +y points toward.
_EDGE_TOWARD = {
    ("x", "posterior"): ("chest_wall", Edge.RIGHT),
    ("x", "anterior"): ("chest_wall", Edge.LEFT),
    ("x", "superior"): ("superior", Edge.RIGHT),
    ("x", "inferior"): ("superior", Edge.LEFT),
    ("y", "posterior"): ("chest_wall", Edge.BOTTOM),
    ("y", "anterior"): ("chest_wall", Edge.TOP),
    ("y", "superior"): ("superior", Edge.BOTTOM),
    ("y", "inferior"): ("superior", Edge.TOP),
}


def _anatomy(component: str, side: Laterality) -> Optional[str]:
    """Map an orientation component to anterior/posterior/superior/inferior.

    Uses the principal (first) letter. Plain L/R counts as superior when it
    points lateral and inferior when it points medial (the CC-type convention).
    """

    principal = component[:1]
    if principal in _PRINCIPAL:
        return _PRINCIPAL[principal]
    if principal in ("L", "R") and side is not Laterality.UNKNOWN:
        return "superior" if principal == side.value else "inferior"
    return None


_PRINCIPAL = {"A": "anterior", "P": "posterior", "H": "superior", "F": "inferior"}


# Field of View ---------------------------------------------------------------

_L, _R = Laterality.LEFT, Laterality.RIGHT
_ROT0, _ROT180 = FovRotation.NONE, FovRotation.FULL
_FLIP, _NOFLIP = FovHorizontalFlip.YES, FovHorizontalFlip.NO

_OPPOSITE = {Edge.LEFT: Edge.RIGHT, Edge.RIGHT: Edge.LEFT, Edge.TOP: Edge.BOTTOM, Edge.BOTTOM: Edge.TOP}

# Alignment for Hologic standard views; from_fov adjusts it for GE and reverse views.
_FOV_ALIGNMENT = {
    (_L, _ROT180, _FLIP): (Edge.RIGHT, Edge.TOP),
    (_L, _ROT180, _NOFLIP): (Edge.LEFT, Edge.TOP),
    (_L, _ROT0, _FLIP): (Edge.LEFT, Edge.BOTTOM),
    (_L, _ROT0, _NOFLIP): (Edge.RIGHT, Edge.BOTTOM),
    (_R, _ROT180, _FLIP): (Edge.RIGHT, Edge.BOTTOM),
    (_R, _ROT180, _NOFLIP): (Edge.LEFT, Edge.BOTTOM),
    (_R, _ROT0, _FLIP): (Edge.LEFT, Edge.TOP),
    (_R, _ROT0, _NOFLIP): (Edge.RIGHT, Edge.TOP),
}


# Record access ---------------------------------------------------------------


def _field(record: Any, *names: str) -> Any:
    """Return the first present, non-missing value among ``names``."""

    for name in names:
        value = record.get(name) if hasattr(record, "get") else getattr(record, name, None)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first_item(record: Any, sequence: str) -> Any:
    items = _field(record, sequence)
    try:
        return items[0] if items is not None and len(items) else None
    except TypeError:
        return None


def _frame_field(record: Any, macro: str, name: str) -> Any:
    """Read ``name`` from a functional group macro, shared groups first, then frame 1."""

    for groups in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        group = _first_item(record, groups)
        item = _first_item(group, macro) if group is not None else None
        value = _field(item, name) if item is not None else None
        if value is not None:
            return value
    return None
