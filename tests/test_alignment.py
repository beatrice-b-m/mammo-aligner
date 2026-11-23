import itertools

import numpy as np
import pytest

from mammo_aligner import Alignment, Edge

HORIZONTAL, VERTICAL = [Edge.LEFT, Edge.RIGHT], [Edge.TOP, Edge.BOTTOM]
ALL = [Alignment(c, s) for c, s in itertools.product(HORIZONTAL, VERTICAL)] + [
    Alignment(c, s) for c, s in itertools.product(VERTICAL, HORIZONTAL)  # rotated a quarter turn
]


@pytest.mark.parametrize("source", ALL)
@pytest.mark.parametrize("target", ALL)
def test_coordinates_follow_the_pixels_they_mark(source, target):
    image = np.arange(4 * 5).reshape(4, 5)
    box = (1, 0, 2, 3)  # inclusive (y_min, x_min, y_max, x_max)
    marked = image[box[0] : box[2] + 1, box[1] : box[3] + 1]

    flipped = source.realign_image(image, target)
    y0, x0, y1, x1 = source.realign_box(box, height=4, width=5, target=target)
    assert sorted(flipped[y0 : y1 + 1, x0 : x1 + 1].ravel()) == sorted(marked.ravel())

    y, x = source.realign_point((1, 0), height=4, width=5, target=target)
    assert flipped[y, x] == image[1, 0]


@pytest.mark.parametrize("source", ALL)
def test_round_trip_restores_the_original(source):
    image = np.arange(3 * 4 * 5).reshape(3, 4, 5)  # DBT-shaped: (frames, rows, columns)
    reference = Alignment.reference()
    realigned = source.realign_image(image)
    assert np.array_equal(reference.realign_image(realigned, target=source), image)

    box = (0, 1, 2, 4)
    there = source.realign_box(box, height=4, width=5)
    height, width = realigned.shape[-2:]  # swapped when source is rotated
    assert reference.realign_box(there, height=height, width=width, target=source) == box


@pytest.mark.parametrize("source", ALL)
def test_nested_input_matches_one_at_a_time(source):
    boxes = [(1, 0, 2, 3), (0, 0, 1, 1), (2, 2, 3, 4), (0, 1, 0, 1)]  # four boxes, like one box's four values
    nested = source.realign_box(boxes, height=4, width=5)
    assert nested == [source.realign_box(b, height=4, width=5) for b in boxes]
    assert source.realign_box(np.array(boxes), height=4, width=5) == nested

    points = [(1, 0), (3, 4)]
    assert source.realign_point(points, height=4, width=5) == [
        source.realign_point(p, height=4, width=5) for p in points
    ]


def test_single_input_gives_a_tuple_and_nested_input_a_list():
    reference = Alignment.reference()
    assert reference.realign_box([1, 0, 2, 3], height=4, width=5) == (1, 0, 2, 3)
    assert reference.realign_box([[1, 0, 2, 3]], height=4, width=5) == [(1, 0, 2, 3)]
    assert reference.realign_box([], height=4, width=5) == []
    assert reference.realign_point(np.array([1, 2]), height=4, width=5) == (1, 2)


@pytest.mark.parametrize(
    "box",
    [
        (1, 0, 2),  # too few values
        [[1, 0, 2, 3], [1, 0, 2]],  # one malformed box among good ones
        [[[1, 0, 2, 3]]],  # nested too deep
        (2, 0, 1, 3),  # y_min > y_max
        (0, 3, 1, 2),  # x_min > x_max
        "0123",
        5,
    ],
)
def test_malformed_boxes_raise(box):
    with pytest.raises(ValueError):
        Alignment.reference().realign_box(box, height=4, width=5)


def test_malformed_points_raise():
    with pytest.raises(ValueError):
        Alignment.reference().realign_point((1, 2, 3), height=4, width=5)


def test_realigning_unknown_raises():
    with pytest.raises(ValueError):
        Alignment(Edge.LEFT, Edge.UNKNOWN).realign_image(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Alignment.reference().realign_image(np.zeros((2, 2)), target=Alignment.unknown())


def _edge_pixels(image, edge):
    # Interior of an edge, leaving out the corners it shares with the perpendicular edges.
    return {
        Edge.LEFT: image[1:-1, 0], Edge.RIGHT: image[1:-1, -1],
        Edge.TOP: image[0, 1:-1], Edge.BOTTOM: image[-1, 1:-1],
    }[edge]


@pytest.mark.parametrize("source", ALL)
@pytest.mark.parametrize("target", ALL)
def test_anatomy_lands_on_the_target_edges(source, target):
    image = np.zeros((5, 7), dtype=int)
    _edge_pixels(image, source.chest_wall)[:] = 1
    _edge_pixels(image, source.superior)[:] = 2

    realigned = source.realign_image(image, target)
    assert (_edge_pixels(realigned, target.chest_wall) == 1).all()
    assert (_edge_pixels(realigned, target.superior) == 2).all()


def test_quarter_turn_swaps_the_image_shape():
    image = np.zeros((3, 4, 5))  # DBT-shaped
    assert Alignment(Edge.BOTTOM, Edge.RIGHT).realign_image(image).shape == (3, 5, 4)


def test_edges_must_be_on_different_axes():
    with pytest.raises(ValueError):
        Alignment(Edge.LEFT, Edge.RIGHT)


@pytest.mark.parametrize(
    "orientation, laterality",
    [
        (["A", "L"], "R"),  # CC: +y points medial, so lateral is at the top
        ("A\\FL", "R"),  # MLO: principal letter F is inferior
        ("['A', 'F']", None),  # H/F needs no laterality
    ],
)
def test_orientation_reference_cases(orientation, laterality):
    assert Alignment.from_orientation(orientation, laterality).is_reference


def test_orientation_mirror_case():
    # Left CC displayed chest-wall-right: +x posterior, +y medial.
    assert Alignment.from_orientation(["P", "R"], "L") == Alignment(Edge.RIGHT, Edge.TOP)


def test_head_pointing_down_puts_superior_at_the_bottom():
    assert Alignment.from_orientation(["P", "H"], "R") == Alignment(Edge.RIGHT, Edge.BOTTOM)


def test_transposed_orientation_is_recognised_as_rotated():
    assert Alignment.from_orientation(["L", "A"], "R") == Alignment(Edge.TOP, Edge.LEFT)


@pytest.mark.parametrize(
    "orientation, laterality",
    [
        (None, "L"),
        ("garbage", "L"),
        (["P", "X"], "L"),
        (["P", "R"], None),  # plain L/R needs laterality
        (["A", "P"], "L"),  # both axes claim the chest wall
    ],
)
def test_orientation_incomplete_or_malformed_is_not_known(orientation, laterality):
    assert not Alignment.from_orientation(orientation, laterality).is_known


def test_fov_unsupported_rotation_is_unknown():
    assert not Alignment.from_fov(90, "NO", "L", "CC", "HOLOGIC, Inc.").is_known


@pytest.mark.parametrize("view, manufacturer", [("CC", "iCAD, Inc."), ("CC", None), ("SPOT", "HOLOGIC, Inc."), (None, "HOLOGIC, Inc.")])
def test_fov_unvalidated_vendor_or_view_is_unknown(view, manufacturer):
    assert not Alignment.from_fov(0, "NO", "L", view, manufacturer).is_known


def test_fov_flip_moves_only_the_chest_wall():
    plain = Alignment.from_fov(0, "NO", "R", "CC", "HOLOGIC, Inc.")
    flipped = Alignment.from_fov(0, "YES", "R", "CC", "HOLOGIC, Inc.")
    assert plain.superior is flipped.superior
    assert plain.chest_wall is not flipped.chest_wall


def test_fov_ge_is_hologic_rotated_180():
    hologic = Alignment.from_fov(0, "NO", "L", "CC", "HOLOGIC, Inc.")
    ge = Alignment.from_fov(0, "NO", "L", "CC", "GE MEDICAL SYSTEMS")
    assert hologic.chest_wall is not ge.chest_wall
    assert hologic.superior is not ge.superior


def test_fov_reverse_view_mirrors_only_the_superior_edge():
    standard = Alignment.from_fov(0, "NO", "L", "ML", "HOLOGIC, Inc.")
    reverse = Alignment.from_fov(0, "NO", "L", "LM", "HOLOGIC, Inc.")
    assert standard.chest_wall is reverse.chest_wall
    assert standard.superior is not reverse.superior


# (PatientOrientation, laterality, view, FOV rotation, FOV flip, manufacturer) seen in EMBED metadata.
@pytest.mark.parametrize(
    "orientation, laterality, view, rotation, flip, manufacturer",
    [
        (["A", "R"], "L", "CC", 0.0, "NO", "GE MEDICAL SYSTEMS"),
        (["P", "FL"], "R", "MLO", 180.0, "NO", "GE HEALTHCARE"),
        (["P", "F"], "L", "LM", 0.0, "NO", "HOLOGIC, Inc."),
        (["P", "H"], "R", "LM", 0.0, "NO", "HOLOGIC, Inc."),
        (["P", "F"], "L", "LM", 180.0, "NO", "GE MEDICAL SYSTEMS"),
        (["P", "F"], "R", "MLO", 0.0, "NO", "HOLOGIC, Inc."),
        (["A", "L"], "R", "CV", 0.0, "YES", "HOLOGIC, Inc."),
    ],
)
def test_fov_matches_orientation_on_observed_metadata(orientation, laterality, view, rotation, flip, manufacturer):
    by_orientation = Alignment.from_orientation(orientation, laterality)
    assert by_orientation.is_known
    assert Alignment.from_fov(rotation, flip, laterality, view, manufacturer) == by_orientation


def test_from_metadata_prefers_orientation_then_falls_back_to_fov():
    record = {
        "ImageLateralityFinal": "R",
        "ViewPosition": "CC ",
        "Manufacturer": "HOLOGIC, Inc.",
        "PatientOrientation": "['A', 'L']",
        "FieldOfViewRotation": 0.0,
        "FieldOfViewHorizontalFlip": "YES",
    }
    assert Alignment.from_metadata(record).is_reference

    record["PatientOrientation"] = float("nan")
    assert Alignment.from_metadata(record) == Alignment.from_fov(0.0, "YES", "R", "CC", "HOLOGIC, Inc.")


def test_from_metadata_reads_attributes():
    class Dataset:  # stands in for pydicom.Dataset attribute access
        ImageLaterality = "L"
        ViewPosition = "MLO"
        PatientOrientation = ["P", "FR"]

    assert Alignment.from_metadata(Dataset()) == Alignment(Edge.RIGHT, Edge.TOP)


def test_from_metadata_with_nothing_is_unknown():
    assert not Alignment.from_metadata({}).is_known


def test_from_metadata_reads_multiframe_functional_groups():
    # DBT objects keep orientation and laterality per frame rather than at the top level.
    record = {
        "SharedFunctionalGroupsSequence": [
            {"FrameAnatomySequence": [{"FrameLaterality": "L"}]},
        ],
        "PerFrameFunctionalGroupsSequence": [
            {"PatientOrientationInFrameSequence": [{"PatientOrientation": ["P", "R"]}]},
        ],
    }
    assert Alignment.from_metadata(record) == Alignment(Edge.RIGHT, Edge.TOP)


def test_from_metadata_derives_orientation_from_image_orientation_patient():
    # Shape of a GE Senographe Pristina DBT header (right CC): no Patient Orientation anywhere,
    # only direction cosines and laterality in the shared functional group.
    record = {
        "SharedFunctionalGroupsSequence": [
            {
                "FrameAnatomySequence": [{"FrameLaterality": "R"}],
                "PlaneOrientationSequence": [{"ImageOrientationPatient": [0, 1, 0, 1, 0, -0.0]}],
            }
        ],
    }
    assert Alignment.from_metadata(record) == Alignment.from_orientation(["P", "L"], "R")


@pytest.mark.parametrize(
    "cosines, expected",
    [
        ([0, 1, 0, 1, 0, 0], ("P", "L")),
        ([0, -1, 0, -1, 0, 0], ("A", "R")),
        ([0, 1, 0, 0.6, 0, -0.8], ("P", "F")),  # oblique (MLO-like): the larger component wins
        ("[0, 1, 0, 1, 0, 0]", ("P", "L")),
        ([0, 0, 0, 1, 0, 0], ("", "L")),  # not a direction
        ([1, 0, 0], ("", "")),
        (None, ("", "")),
    ],
)
def test_direction_cosines_to_patient_orientation(cosines, expected):
    from mammo_aligner.metadata import PatientOrientation

    parsed = PatientOrientation.from_cosines(cosines)
    assert (parsed.row, parsed.column) == expected
