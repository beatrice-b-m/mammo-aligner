# mammo-aligner

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24472822.svg)](https://doi.org/10.5281/zenodo.24472822)

Read the orientation of a mammogram from its DICOM metadata and realign the
image, and points or boxes on it, into one common frame.

Mammograms from different vendors and pipelines are stored in different
on-screen orientations: the chest wall may sit on the left or the right, the
superior edge at the top or the bottom, and some images are stored a quarter
turn off. `mammo-aligner` infers where the anatomy sits from standard DICOM
attributes, then flips (and if needed transposes) the pixels and coordinates so
the chest wall is on the left and the superior side is at the top. It handles
FFDM, synthetic 2D (C-View), and DBT, and was built and checked on the [Emory
Breast Imaging Dataset (EMBED)](https://github.com/Emory-HITI/EMBED_Open_Data).

![A left CC stored with the chest wall on the right is flipped in x and y; a right MLO stored a quarter turn off is transposed and flipped in x. Both end with the chest wall on the left and superior (lateral for CC) at the top, and the ROI box moves with the image.](docs/realignment.svg)

Pure Python with no runtime dependencies; images are NumPy arrays.

## Installation

```bash
pip install "mammo-aligner @ git+https://github.com/beatrice-b-m/mammo-aligner.git@v0.2.0"
```

or, with uv:

```bash
uv add "mammo-aligner @ git+https://github.com/beatrice-b-m/mammo-aligner.git@v0.2.0"
```

## Quick start

`Alignment.from_metadata` takes a pydicom `Dataset`, a dict, or a pandas row
(for example from the EMBED metadata table):

```python
import ast
import pandas as pd
from mammo_aligner import Alignment

meta = pd.read_csv("embed_metadata.csv")  # any table with DICOM attributes as columns

for _, row in meta.iterrows():
    alignment = Alignment.from_metadata(row)
    if not alignment.is_known:
        continue  # the metadata does not determine the orientation

    image = load_pixels(row)  # your loader, e.g. pydicom.dcmread(path).pixel_array
    h, w = image.shape[-2:]
    aligned = alignment.realign_image(image)

    # EMBED stores ROI_coords as a serialized list of boxes
    rois = ast.literal_eval(row["ROI_coords"]) if isinstance(row["ROI_coords"], str) else []
    aligned_rois = alignment.realign_box(rois, height=h, width=w)
```

[`examples/realign_synthetic.py`](examples/realign_synthetic.py) runs the same
steps on a 4 × 6 array, so you can see the result:

```text
Stored as: chest wall right, superior bottom
Aligned image (chest wall left, lateral top):
[[9 9 1 1 0 0]
 [9 9 1 1 1 0]
 [1 1 1 1 0 0]
 [1 1 1 0 0 0]]
ROI: (2, 4, 3, 5) -> (0, 0, 1, 1)
ROI mapped back: (2, 4, 3, 5)
```

## The reference frame

An `Alignment` records two image edges: the one holding the **chest wall** and
the one holding the **superior** side of the breast. CC-type views have no
superior/inferior axis, so for them the `superior` edge holds the **lateral**
side instead. The reference frame, `Alignment.reference()`, is chest wall on
the left and superior (lateral for CC) at the top.

## How the orientation is inferred

`from_metadata` reads Patient Orientation from the first of sources 1–3 that
has it, and falls back to source 4 if that does not give a known alignment:

1. **Patient Orientation** (0020,0020), e.g. `P\L`. Each value is read by its
   principal (first) letter: A/P place the chest wall and H/F the superior
   side. Plain L/R values (CC-type views) also need the laterality, read from
   `ImageLateralityFinal` (EMBED), `ImageLaterality`, or `Laterality`.
2. For multi-frame objects such as DBT, Patient Orientation and Frame
   Laterality in the shared or first per-frame **functional group**.
3. If no Patient Orientation exists at all (e.g. GE Pristina DBT), it is
   derived from **Image Orientation (Patient)** direction cosines.
4. **Field of View Rotation and Horizontal Flip** (0018,7032 and 0018,7034).
   These describe the image relative to the detector, so the result is
   corrected for the vendor (GE detectors are rotated 180° relative to
   Hologic) and for reverse views (LM, LMO, LMID, FB, SIO). Only Hologic and
   GE, and the views checked on EMBED, are supported.

If none of these works, the result is an unknown alignment (`is_known` is
`False`); missing or malformed metadata never raises. Each source has its own
constructor if you want to bypass the fallback:

| Constructor | Input |
| --- | --- |
| `Alignment.from_orientation(orientation, laterality=None)` | Patient Orientation as `["P", "L"]`, `"P\\L"`, or `"['P', 'L']"` |
| `Alignment.from_fov(rotation, horizontal_flip, laterality, view_position, manufacturer)` | Field of View attributes |
| `Alignment(chest_wall, superior)` | Explicit edges, e.g. `Alignment("right", "top")` |

## Realignment

`realign_image`, `realign_point`, and `realign_box` move an image or
coordinates into the reference frame, or into any other alignment passed as
`target=`. Mapping results back into the stored frame (for example,
model predictions) is the same call from the reference side:

```python
original_box = Alignment.reference().realign_box(
    aligned_rois[0], height=aligned.shape[-2], width=aligned.shape[-1], target=alignment
)
```

- **Coordinates are `(y, x)`**, i.e. (row, column) inclusive pixel indices.
  Boxes are `(y_min, x_min, y_max, x_max)`, matching EMBED `ROI_coords`.
- `realign_point` and `realign_box` take one point or box, or a sequence of
  them (a list, or an `(N, 2)` / `(N, 4)` array). One item comes back as a
  tuple; a sequence comes back as a list of tuples. An input counts as one
  item when its first element is a number. Items of the wrong length, and
  boxes whose minimum exceeds their maximum, raise `ValueError`.
- `height` and `width` are always those of the image the coordinates are
  currently on.
- The last two image axes are (rows, columns), so 2D images and DBT volumes
  shaped (frames, rows, columns) both work. Frame order is unchanged.
- An image stored a quarter turn off (chest wall at the top or bottom) is
  transposed as well as flipped, so its realigned shape is
  (..., columns, rows).
- Realigning an unknown alignment raises `ValueError`.
- `realign_image` returns a NumPy view, not a copy. Views from flips have
  negative strides, which `torch.from_numpy` rejects: wrap the result in
  `np.ascontiguousarray` first. Pass NumPy arrays, not torch tensors, since
  torch does not support reversed slicing.

The library only reorients. Pixel spacing, cropping, intensity inversion
(MONOCHROME1), and windowing are left to your pipeline.

## Validation

The [`validation/`](validation/) folder holds the checks used on EMBED; both
need EMBED access. `embed_metadata.py` infers each image's alignment from
Patient Orientation and from Field of View and reports agreement by
manufacturer, model, image type, and view (aggregate counts only).
`visual_spot_check.ipynb` shows sampled images as stored and realigned, with
the edges the metadata claims and the ROI boxes.

```bash
uv run --group validation python validation/embed_metadata.py path/to/metadata.csv
```

## Citation

If you use `mammo-aligner` in your research, please cite the version you used.
Each release has its own DOI on Zenodo; [10.5281/zenodo.24472822](https://doi.org/10.5281/zenodo.24472822)
always resolves to the latest. GitHub's **Cite this repository** button (from
[`CITATION.cff`](CITATION.cff)) gives APA and BibTeX; for v0.2.0:

```bibtex
@software{brownmulry_mammo_aligner,
  author  = {Brown-Mulry, Beatrice},
  title   = {mammo-aligner},
  version = {0.2.0},
  year    = {2026},
  doi     = {10.5281/zenodo.24472823},
  url     = {https://doi.org/10.5281/zenodo.24472823}
}
```

## License

Apache 2.0.
