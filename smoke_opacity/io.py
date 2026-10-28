from pathlib import Path
import numpy as np
from PIL import Image
import xarray as xr


def load_png_sequence(
    png_dir,
    pattern="*.png",
    start=0,
    n_frames=None,
    step=1,
):
    import imageio.v3 as iio

    png_dir = Path(png_dir)

    files = sorted(png_dir.glob(pattern))

    if n_frames is None:
        selected_files = files[start::step]
    else:
        selected_files = files[start:start + n_frames * step:step]

    rgb_sequence = []
    times = []

    for f in selected_files:
        img = iio.imread(f)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.shape[-1] == 4:
            img = img[..., :3]

        rgb_sequence.append(img)
        times.append(f.stem)

    rgb_sequence = np.stack(rgb_sequence, axis=0)

    return rgb_sequence, times, selected_files


def open_tracking_dataset(path):
    return xr.open_dataset(path)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
