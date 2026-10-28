from pathlib import Path
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image

import torch
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops

from .io import load_png_sequence, ensure_dir


_MODEL = None
_PROCESSOR = None


def get_sam3_processor(device="cuda"):
    """
    Lazy-load SAM3 only when segmentation is actually called.

    This avoids loading the model just because the module is imported.
    """
    global _MODEL, _PROCESSOR

    if _PROCESSOR is not None:
        return _PROCESSOR

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print("Loading SAM3...")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    _MODEL = build_sam3_image_model()
    if device == "cuda":
        _MODEL = _MODEL.cuda()
    _MODEL = _MODEL.eval()

    _PROCESSOR = Sam3Processor(_MODEL)
    print("SAM3 model + processor OK")
    return _PROCESSOR


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    if hasattr(x, "cpu"):
        x = x.cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.asarray(x)


def box_xyxy_to_cxcywh_norm(box_xyxy, image_shape):
    x0, y0, x1, y1 = box_xyxy
    H, W = image_shape[:2]

    cx = (x0 + x1) / 2 / W
    cy = (y0 + y1) / 2 / H
    w = (x1 - x0) / W
    h = (y1 - y0) / H

    return [float(cx), float(cy), float(w), float(h)]


def clean_mask(
    mask,
    min_area_px=300,
    max_area_fraction=0.25,
    fill_holes=True,
    source_xy=None,
    max_source_distance_px=600,
):
    import numpy as np
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label, regionprops

    mask = np.asarray(mask).astype(bool)

    if fill_holes:
        mask = binary_fill_holes(mask)

    H, W = mask.shape
    max_area_px = max_area_fraction * H * W

    lab = label(mask)
    props = regionprops(lab)

    if len(props) == 0:
        return np.zeros_like(mask, dtype=bool)

    candidates = []

    for r in props:
        area = r.area

        if area < min_area_px:
            continue

        if area > max_area_px:
            continue

        cy, cx = r.centroid

        if source_xy is not None:
            sx, sy = source_xy
            dist = np.hypot(cx - sx, cy - sy)

            if dist > max_source_distance_px:
                continue
        else:
            dist = 0.0

        candidates.append((r, dist, area))

    if len(candidates) == 0:
        return np.zeros_like(mask, dtype=bool)

    # Prefer closest component to fire source, then larger area
    candidates = sorted(candidates, key=lambda x: (x[1], -x[2]))

    best = candidates[0][0]

    return lab == best.label


def mask_to_bbox(mask, pad=100, min_area_px=300, image_shape=None):
    mask = np.asarray(mask).astype(bool)
    lab = label(mask)
    props = regionprops(lab)

    if len(props) == 0:
        return None

    largest = max(props, key=lambda r: r.area)

    if largest.area < min_area_px:
        return None

    y0, x0, y1, x1 = largest.bbox

    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad

    if image_shape is not None:
        H, W = image_shape[:2]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W - 1, x1)
        y1 = min(H - 1, y1)

    return [int(x0), int(y0), int(x1), int(y1)]


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.nan, np.nan
    return float(xs.mean()), float(ys.mean())


def bbox_from_mask_with_anchor(mask, anchor_xy=None, pad_px=500, image_shape=None):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    if anchor_xy is not None:
        ax, ay = anchor_xy
        x0 = min(x0, ax)
        x1 = max(x1, ax)
        y0 = min(y0, ay)
        y1 = max(y1, ay)

    x0 -= pad_px
    y0 -= pad_px
    x1 += pad_px
    y1 += pad_px

    if image_shape is not None:
        H, W = image_shape[:2]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W - 1, x1)
        y1 = min(H - 1, y1)

    return [int(x0), int(y0), int(x1), int(y1)]


def enforce_min_box_size(box, min_w=1000, min_h=800, image_shape=None):
    if box is None:
        return None

    x0, y0, x1, y1 = box

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    w = max(x1 - x0, min_w)
    h = max(y1 - y0, min_h)

    x0 = cx - w / 2
    x1 = cx + w / 2
    y0 = cy - h / 2
    y1 = cy + h / 2

    if image_shape is not None:
        H, W = image_shape[:2]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W - 1, x1)
        y1 = min(H - 1, y1)

    return [int(x0), int(y0), int(x1), int(y1)]


def sam3_predict_box(rgb, box_xyxy, prompt="smoke", device="cuda"):
    """
    Run SAM3 using text prompt + geometric box prompt.

    Parameters
    ----------
    rgb : np.ndarray
        uint8 RGB image, shape (H, W, 3)
    box_xyxy : list
        [x0, y0, x1, y1] in pixel coordinates
    prompt : str
        Text prompt, e.g. "smoke"

    Returns
    -------
    mask : np.ndarray
        Boolean mask, shape (H, W)
    """
    processor = get_sam3_processor(device=device)

    img = Image.fromarray(rgb.astype(np.uint8))
    box_norm = box_xyxy_to_cxcywh_norm(box_xyxy, rgb.shape)

    autocast_enabled = device == "cuda"

    with torch.inference_mode():
        if autocast_enabled:
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            context = torch.autocast(device_type="cpu", enabled=False)

        with context:
            state = processor.set_image(img)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            out = processor.add_geometric_prompt(
                state=state,
                box=box_norm,
                label=True,
            )

    masks = out["masks"]
    scores = out.get("scores", None)

    masks_np = np.asarray(to_numpy(masks))

    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, :, :]

    if masks_np.shape[0] == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)

    if scores is not None:
        scores_np = np.asarray(to_numpy(scores)).ravel()
        best = int(np.nanargmax(scores_np))
    else:
        areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
        best = int(np.argmax(areas))

    mask = masks_np[best] > 0.5
    return mask


def plot_tracking_qc(
    rgb,
    mask,
    input_box=None,
    output_box=None,
    prev_mask=None,
    title="",
    out_png=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)

    if prev_mask is not None:
        ax.contour(prev_mask, levels=[0.5], linewidths=1, linestyles="--")

    ax.imshow(np.ma.masked_where(~mask, mask), alpha=0.45)
    ax.contour(mask, levels=[0.5], linewidths=1)

    if input_box is not None:
        x0, y0, x1, y1 = input_box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2)

    if output_box is not None:
        x0, y0, x1, y1 = output_box
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            linestyle="--",
            linewidth=2,
        )

    ax.set_title(title)
    ax.axis("off")

    if out_png is not None:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_tracking_netcdf(
    out_nc,
    rgb_sequence,
    mask_sequence,
    times,
    input_boxes,
    output_boxes,
    records,
    text_prompt,
    pad_px,
    min_area_px,
):
    rgb_arr = np.stack(rgb_sequence).astype(np.uint8)
    mask_arr = np.stack(mask_sequence).astype(np.uint8)

    input_boxes_arr = np.asarray(input_boxes, dtype=np.float32)
    output_boxes_arr = np.asarray(output_boxes, dtype=np.float32)

    area_px = np.asarray([r["area_px"] for r in records], dtype=np.int32)
    centroid_x = np.asarray([r["centroid_x"] for r in records], dtype=np.float32)
    centroid_y = np.asarray([r["centroid_y"] for r in records], dtype=np.float32)
    success = np.asarray([r["success"] for r in records], dtype=np.uint8)

    nt, ny, nx, nc = rgb_arr.shape

    ds_out = xr.Dataset(
        data_vars={
            "rgb": (("time", "y", "x", "channel"), rgb_arr),
            "plume_mask": (("time", "y", "x"), mask_arr),
            "input_box": (("time", "bbox_coord"), input_boxes_arr),
            "output_box": (("time", "bbox_coord"), output_boxes_arr),
            "area_px": (("time",), area_px),
            "centroid_x": (("time",), centroid_x),
            "centroid_y": (("time",), centroid_y),
            "success": (("time",), success),
        },
        coords={
            "time": times,
            "y": np.arange(ny),
            "x": np.arange(nx),
            "channel": ["red", "green", "blue"],
            "bbox_coord": ["x0", "y0", "x1", "y1"],
        },
        attrs={
            "description": "Primitive sequential SAM3 plume segmentation tracking",
            "sam_prompt": text_prompt,
            "pad_px": pad_px,
            "min_area_px": min_area_px,
            "mask_values": "0=background, 1=plume",
        },
    )

    encoding = {
        "rgb": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "plume_mask": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "input_box": {"zlib": True, "complevel": 4},
        "output_box": {"zlib": True, "complevel": 4},
    }

    ds_out.to_netcdf(out_nc, encoding=encoding)
    return ds_out


def run_sam3_primitive_tracking(
    png_dir,
    out_dir,
    first_box_xyxy,
    pattern="*.png",
    prompt="smoke",
    pad_px=700,
    min_area_px=3000,
    source_xy=None,
    min_box_w=1200,
    min_box_h=900,
    device="cuda",
    save_qc=True,
):
    """
    Main primitive tracking loop from your notebook.

    This is stage 1 of the workflow: plume segmentation.
    """
    out_dir = ensure_dir(out_dir)

    qc_dir = ensure_dir(out_dir / "qc_png")
    mask_dir = ensure_dir(out_dir / "masks_npy")
    meta_dir = ensure_dir(out_dir / "metadata_json")

    rgb_sequence, times, files = load_png_sequence(png_dir, pattern=pattern)

    mask_sequence = []
    input_boxes = []
    output_boxes = []
    records = []

    prev_mask = None
    prev_box = first_box_xyxy

    for i, (rgb, t, f) in enumerate(zip(rgb_sequence, times, files)):
        print(f"Processing {i+1}/{len(rgb_sequence)}: {f.name}")

        prev_mask_for_plot = prev_mask.copy() if prev_mask is not None else None

        if i == 0:
            input_box = first_box_xyxy
        else:
            input_box = bbox_from_mask_with_anchor(
                prev_mask,
                anchor_xy=source_xy,
                pad_px=pad_px,
                image_shape=rgb.shape,
            )

            input_box = enforce_min_box_size(
                input_box,
                min_w=min_box_w,
                min_h=min_box_h,
                image_shape=rgb.shape,
            )

            if input_box is None:
                input_box = prev_box

        raw_mask = sam3_predict_box(
            rgb=rgb,
            box_xyxy=input_box,
            prompt=prompt,
            device=device,
        )

        mask = clean_mask(
        raw_mask,
        min_area_px=min_area_px,
        max_area_fraction=0.20,
        fill_holes=True,
        source_xy=source_xy,
        max_source_distance_px=700,
        )

        output_box = bbox_from_mask_with_anchor(
            mask,
            anchor_xy=source_xy,
            pad_px=pad_px,
            image_shape=rgb.shape,
        )

        output_box = enforce_min_box_size(
            output_box,
            min_w=min_box_w,
            min_h=min_box_h,
            image_shape=rgb.shape,
        )

        success = output_box is not None

        if success:
            prev_box = output_box
        else:
            output_box = [np.nan, np.nan, np.nan, np.nan]

        prev_mask = mask
        cx, cy = mask_centroid(mask)

        rec = {
            "index": int(i),
            "time": str(t),
            "file": str(f),
            "input_box": [float(v) for v in input_box],
            "output_box": [float(v) for v in output_box],
            "area_px": int(mask.sum()),
            "centroid_x": cx,
            "centroid_y": cy,
            "success": bool(success),
            "source_x": None if source_xy is None else float(source_xy[0]),
            "source_y": None if source_xy is None else float(source_xy[1]),
        }

        mask_sequence.append(mask)
        input_boxes.append(input_box)
        output_boxes.append(output_box)
        records.append(rec)

        tag = f"{i:04d}_{t}"
        np.save(mask_dir / f"mask_{tag}.npy", mask.astype(np.uint8))

        with open(meta_dir / f"meta_{tag}.json", "w") as fh:
            json.dump(rec, fh, indent=2)

        if save_qc:
            plot_tracking_qc(
                rgb,
                mask,
                input_box=input_box,
                output_box=None if not success else output_box,
                prev_mask=prev_mask_for_plot,
                title=f"{i:04d} {t} | area={int(mask.sum())} px",
                out_png=qc_dir / f"qc_{tag}.png",
            )

    ds_track = save_tracking_netcdf(
        out_nc=out_dir / "plume_tracking.nc",
        rgb_sequence=rgb_sequence,
        mask_sequence=mask_sequence,
        times=times,
        input_boxes=input_boxes,
        output_boxes=output_boxes,
        records=records,
        text_prompt=prompt,
        pad_px=pad_px,
        min_area_px=min_area_px,
    )

    with open(out_dir / "tracking_records.json", "w") as fh:
        json.dump(records, fh, indent=2)

    return ds_track, records

def run_sam3_independent_frames(
    png_dir,
    out_dir,
    box_xyxy,
    pattern="*.png",
    prompt="smoke",
    min_area_px=100,
    device="cuda",
    save_qc=True,
    start=0,
    n_frames=None,
    step=1,
):
    from pathlib import Path
    import json
    import numpy as np
    import imageio.v3 as iio

    from .io import ensure_dir

    png_dir = Path(png_dir)
    out_dir = ensure_dir(out_dir)
    qc_dir = ensure_dir(out_dir / "qc_png")
    mask_dir = ensure_dir(out_dir / "masks_npy")

    # ------------------------------------------------------------
    # Select files BEFORE loading images
    # ------------------------------------------------------------
    files_all = sorted(png_dir.glob(pattern))

    if len(files_all) == 0:
        raise FileNotFoundError(f"No PNG files found in {png_dir} with pattern {pattern}")

    if n_frames is None:
        selected = list(enumerate(files_all))[start::step]
    else:
        selected = list(enumerate(files_all))[start:start + n_frames * step:step]

    if len(selected) == 0:
        raise ValueError(
            f"No frames selected. Check start={start}, n_frames={n_frames}, step={step}."
        )

    original_indices = [idx for idx, _ in selected]
    files = [f for _, f in selected]
    times = [f.stem for f in files]

    print(f"Found {len(files_all)} total PNG files")
    print(f"Selected {len(files)} frames")
    print(f"First selected frame: original index {original_indices[0]} | {files[0].name}")
    print(f"Last selected frame:  original index {original_indices[-1]} | {files[-1].name}")

    # ------------------------------------------------------------
    # Run SAM3 only on selected frames
    # ------------------------------------------------------------
    rgb_sequence = []
    mask_sequence = []
    records = []

    for i, (original_index, f, t) in enumerate(zip(original_indices, files, times)):

        print(
            f"Processing selected frame {i + 1}/{len(files)} "
            f"(original index {original_index}): {f.name}"
        )

        rgb = iio.imread(f)

        # Ensure RGB
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)

        # Drop alpha channel if PNG is RGBA
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        rgb_sequence.append(rgb)

        raw_mask = sam3_predict_box(
            rgb=rgb,
            box_xyxy=box_xyxy,
            prompt=prompt,
            device=device,
        )

        mask = clean_mask(
            raw_mask,
            min_area_px=min_area_px,
            fill_holes=True,
        )

        mask_sequence.append(mask)

        rec = {
            "index": int(i),                         # selected-frame index
            "original_index": int(original_index),   # index in full PNG sequence
            "time": str(t),
            "file": str(f),
            "input_box": [float(v) for v in box_xyxy],
            "area_px": int(mask.sum()),
            "success": bool(mask.sum() > 0),
        }
        records.append(rec)

        tag = f"{i:04d}_orig{original_index:04d}_{t}"

        np.save(
            mask_dir / f"mask_{tag}.npy",
            mask.astype("uint8"),
        )

        if save_qc:
            plot_tracking_qc(
                rgb,
                mask,
                input_box=box_xyxy,
                output_box=None,
                prev_mask=None,
                title=(
                    f"{i:04d} | original={original_index:04d} | "
                    f"{t} | area={int(mask.sum())} px"
                ),
                out_png=qc_dir / f"qc_{tag}.png",
            )

    # Convert to arrays only after selected frames are loaded
    rgb_sequence = np.stack(rgb_sequence, axis=0)
    mask_sequence = np.stack(mask_sequence, axis=0)

    input_boxes = [box_xyxy for _ in range(len(rgb_sequence))]

    output_boxes = [
        mask_to_bbox(m, pad=0, image_shape=m.shape) or [np.nan] * 4
        for m in mask_sequence
    ]

    records_with_outputs = []

    for r in records:
        idx = r["index"]

        cx, cy = mask_centroid(mask_sequence[idx])

        records_with_outputs.append(
            {
                **r,
                "output_box": output_boxes[idx],
                "centroid_x": cx,
                "centroid_y": cy,
            }
        )

    ds_track = save_tracking_netcdf(
        out_nc=out_dir / "plume_tracking.nc",
        rgb_sequence=rgb_sequence,
        mask_sequence=mask_sequence,
        times=times,
        input_boxes=input_boxes,
        output_boxes=output_boxes,
        records=records_with_outputs,
        text_prompt=prompt,
        pad_px=0,
        min_area_px=min_area_px,
    )

    return ds_track, records_with_outputs