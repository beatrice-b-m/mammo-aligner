"""Parsing for the DICOM and EMBED metadata values that determine alignment.

Every parser is non-throwing: malformed or missing source values become
``UNKNOWN`` so that callers can decide how to handle them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class _CoercibleEnum(Enum):
    """Enum whose ``coerce`` maps any source value to a member or ``UNKNOWN``."""

    @classmethod
    def coerce(cls, value: Any):
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError):
            return cls._coerce_unknown(value)

    @classmethod
    def _coerce_unknown(cls, value: Any):
        return cls.UNKNOWN  # type: ignore[attr-defined]


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


class Laterality(_CoercibleEnum):
    """Breast side shown in an image."""

    LEFT = "L"
    RIGHT = "R"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "Laterality":
        aliases = {"L": cls.LEFT, "LEFT": cls.LEFT, "R": cls.RIGHT, "RIGHT": cls.RIGHT}
        return aliases.get(_normalized_text(value), cls.UNKNOWN)


@dataclass(frozen=True)
class PatientOrientation:
    """DICOM Patient Orientation (0020,0020): the patient directions of the image axes.

    ``row`` is the direction of increasing column index (+x) and ``column`` is
    the direction of increasing row index (+y). Each is a code such as ``"P"``
    or ``"FL"`` built from A, P, R, L, H, F, with the principal direction first
    (DICOM PS3.3 C.7.6.1.1.1). Invalid components are stored as ``""``.
    """

    row: str
    column: str

    @classmethod
    def coerce(cls, value: Any) -> "PatientOrientation":
        """Parse a pydicom value, ``"P\\L"``, ``"['P', 'L']"``, or a pair."""

        if isinstance(value, cls):
            return value
        try:
            row, column = _values(value)
        except (TypeError, ValueError):
            row = column = None
        return cls(_component(row), _component(column))

    @classmethod
    def from_cosines(cls, value: Any) -> "PatientOrientation":
        """Derive the principal directions from DICOM Image Orientation (Patient).

        ``value`` holds six direction cosines in the patient coordinate system
        (+X patient left, +Y posterior, +Z head): three for +x, then three for
        +y. Each axis is labelled by its largest component, which is all
        alignment needs.
        """

        try:
            cosines = [float(v) for v in _values(value)]
        except (TypeError, ValueError):
            cosines = []
        if len(cosines) != 6:
            return cls("", "")
        return cls(_principal_letter(cosines[:3]), _principal_letter(cosines[3:]))


def _values(value: Any) -> Any:
    """Turn ``"A\\B"`` or ``"['A', 'B']"`` into a sequence; pass sequences through."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith(("[", "(")):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    return text.split("\\")


def _component(value: Any) -> str:
    text = _normalized_text(value)
    return text if re.fullmatch(r"[APRLHF]{1,3}", text) else ""


_AXIS_LETTERS = (("L", "R"), ("P", "A"), ("H", "F"))  # (+, -) along patient X, Y, Z


def _principal_letter(vector: list) -> str:
    axis = max(range(3), key=lambda i: abs(vector[i]))
    if abs(vector[axis]) < 0.5:  # not a unit direction
        return ""
    positive, negative = _AXIS_LETTERS[axis]
    return positive if vector[axis] > 0 else negative


class Manufacturer(_CoercibleEnum):
    """Acquisition device manufacturer, for vendors with validated Field of View handling."""

    HOLOGIC = "HOLOGIC"
    GE = "GE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "Manufacturer":
        text = _normalized_text(value)
        if "HOLOGIC" in text or "LORAD" in text:
            return cls.HOLOGIC
        if text == "GE" or text.startswith(("GE ", "GE,")):
            return cls.GE
        return cls.UNKNOWN


class ProjectionSide(_CoercibleEnum):
    """Which side of the breast a view is imaged from, parsed from View Position.

    Reverse views (lateromedial and caudocranial) put the detector on the
    opposite side, which mirrors the Field of View frame top to bottom.
    """

    STANDARD = "STANDARD"
    REVERSE = "REVERSE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "ProjectionSide":
        text = _normalized_text(value)
        if text in _REVERSE_VIEWS:
            return cls.REVERSE
        if text in _STANDARD_VIEWS:
            return cls.STANDARD
        return cls.UNKNOWN


# Views checked against Patient Orientation on EMBED metadata.
_STANDARD_VIEWS = {
    "CC", "XCCL", "XCCM", "CV", "CCID", "RL", "RM",
    "MLO", "MLOID", "ML", "MLID", "AT", "TAN", "ISO",
}
_REVERSE_VIEWS = {"LM", "LMO", "LMID", "FB", "SIO"}


class FovRotation(_CoercibleEnum):
    """DICOM Field of View Rotation (0018,7032), in degrees."""

    NONE = 0.0
    FULL = 180.0
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "FovRotation":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls.UNKNOWN
        return {0.0: cls.NONE, 180.0: cls.FULL}.get(numeric, cls.UNKNOWN)


class FovHorizontalFlip(_CoercibleEnum):
    """DICOM Field of View Horizontal Flip (0018,7034)."""

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "FovHorizontalFlip":
        aliases = {
            "Y": cls.YES, "YES": cls.YES, "TRUE": cls.YES, "1": cls.YES,
            "N": cls.NO, "NO": cls.NO, "FALSE": cls.NO, "0": cls.NO,
        }
        return aliases.get(_normalized_text(value), cls.UNKNOWN)
