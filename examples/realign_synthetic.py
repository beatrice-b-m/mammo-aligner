"""Realign a synthetic left CC mammogram and its ROI into the reference frame.

Run with: uv run python examples/realign_synthetic.py
"""

import numpy as np

from mammo_aligner import Alignment

# A tiny 4 x 6 "left CC" image as a scanner might store it: chest wall in the
# right-hand columns, lateral side at the bottom. Values mark tissue (1) and an
# ROI (9) near the chest wall on the lateral side.
image = np.array(
    [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 9, 9],
        [0, 0, 1, 1, 9, 9],
    ]
)
roi = (2, 4, 3, 5)  # inclusive (y_min, x_min, y_max, x_max), as in EMBED ROI_coords

# The DICOM metadata (a pydicom Dataset, dict, or pandas row all work).
metadata = {
    "ImageLateralityFinal": "L",
    "ViewPosition": "CC",
    "PatientOrientation": ["P", "L"],  # +x points posterior, +y points lateral
}

alignment = Alignment.from_metadata(metadata)
print("Stored as:", alignment)  # for CC views, "superior" is the lateral side

height, width = image.shape
aligned = alignment.realign_image(image)
aligned_roi = alignment.realign_box(roi, height=height, width=width)

print("Aligned image (chest wall left, lateral top):")
print(aligned)
print("ROI:", roi, "->", aligned_roi)

y0, x0, y1, x1 = aligned_roi
assert (aligned[y0 : y1 + 1, x0 : x1 + 1] == 9).all()

# Map back into the original frame, e.g. to write predictions next to the DICOM.
# Pass the aligned image's shape: it differs from the original for rotated images.
restored = Alignment.reference().realign_box(
    aligned_roi, height=aligned.shape[0], width=aligned.shape[1], target=alignment
)
print("ROI mapped back:", restored)
