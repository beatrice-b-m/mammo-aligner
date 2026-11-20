# mg-aligner

A minimal Python library for checking the orientation of mammography images and
realigning them, and the coordinates on them, into a common frame.

Mammograms from different vendors and pipelines are stored with different
on-screen orientations: the chest wall may sit on the left or the right, and the
superior edge may be at the top or the bottom. `mg-aligner` works out an image's
orientation from its metadata and flips the image, points, or bounding boxes into
a single reference frame (or any other frame you choose).

It is built primarily for the [Emory Breast Imaging Dataset
(EMBED)](https://github.com/Emory-HITI/EMBED_Open_Data), and aims to support
synthetic 2D, FFDM, and DBT images from multiple manufacturers.

> **Status:** early development. The public API is not yet stable; installation
> and usage examples will be added here once it is.

## Development

The project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```
