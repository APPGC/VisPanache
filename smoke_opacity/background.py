from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_dilation
from PIL import Image

from .io import load_png_sequence


from .io import ensure_dir


def create_clean_background(
    reference_rgb,
    reference_mask,
    dilation_iterations=50,
    inpaint_radius=60,
):
    """
    Create the cleaned terrain/background reference from one segmented frame.

    This corresponds to the notebook section:
    'Define terrain background'.
    """
    terrain_ref = reference_rgb.astype("float32")
    if terrain_ref.max() > 1.5:
        terrain_ref = terrain_ref / 255.0

    ref_mask = reference_mask.astype(bool)
    ref_mask_dil = binary_dilation(ref_mask, iterations=dilation_iterations)

    terrain_ref_u8 = np.clip(terrain_ref * 255, 0, 255).astype("uint8")

    terrain_clean_u8 = cv2.inpaint(
        terrain_ref_u8,
        (ref_mask_dil.astype("uint8") * 255),
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    terrain_clean = terrain_clean_u8.astype("float32") / 255.0
    return terrain_clean, ref_mask_dil


def create_clean_background_from_tracking(
    ds_track,
    reference_time_index=0,
    dilation_iterations=50,
    inpaint_radius=60,
    out_dir=None,
    save_qc=True,
):
    """
    Create a clean background frame directly from ds_track.

    Parameters
    ----------
    ds_track : xr.Dataset
        Output from run_sam3_primitive_tracking.
    reference_time_index : int
        Frame index used as reference/background source.

    Returns
    -------
    terrain_clean : np.ndarray
        Float32 RGB background, range 0..1.
    ds_bg : xr.Dataset
        Background diagnostics.
    """
    terrain_ref = ds_track.rgb.isel(time=reference_time_index).values.astype("float32") / 255.0
    ref_mask = ds_track.plume_mask.isel(time=reference_time_index).values.astype(bool)

    terrain_clean, ref_mask_dil = create_clean_background(
        terrain_ref,
        ref_mask,
        dilation_iterations=dilation_iterations,
        inpaint_radius=inpaint_radius,
    )

    ds_bg = xr.Dataset(
        data_vars={
            "terrain_ref": (("y", "x", "channel"), terrain_ref.astype("float32")),
            "ref_mask": (("y", "x"), ref_mask.astype("uint8")),
            "ref_mask_dilated": (("y", "x"), ref_mask_dil.astype("uint8")),
            "terrain_clean": (("y", "x", "channel"), terrain_clean.astype("float32")),
        },
        coords={
            "y": ds_track.y.values,
            "x": ds_track.x.values,
            "channel": ds_track.channel.values,
        },
        attrs={
            "description": "Cleaned terrain/background frame created by inpainting segmented plume",
            "reference_time_index": int(reference_time_index),
            "dilation_iterations": int(dilation_iterations),
            "inpaint_radius": float(inpaint_radius),
        },
    )

    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        ds_bg.to_netcdf(out_dir / "background_reference.nc")

        if save_qc:
            save_background_qc(
                terrain_ref,
                ref_mask,
                terrain_clean,
                out_dir / "background_reference_qc.png",
            )

    return terrain_clean, ds_bg


def save_background_qc(terrain_ref, ref_mask, terrain_clean, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(terrain_ref)
    axes[0].set_title("Reference frame")

    axes[1].imshow(ref_mask, cmap="gray")
    axes[1].set_title("Reference plume mask")

    axes[2].imshow(terrain_clean)
    axes[2].set_title("Cleaned terrain reference")

    for ax in axes:
        ax.axis("off")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)



def create_background_from_reference_folder(
    ref_png_dir,
    ref_index=0,
):
    """
    Load a clean background frame from another folder.

    Parameters
    ----------
    ref_png_dir : str or Path
        Folder containing reference/background imagery.

    ref_index : int
        Which frame to use.

    Returns
    -------
    bg : ndarray
        Float32 RGB image in range 0..1
    """

    rgb_ref, times_ref, files_ref = load_png_sequence(
        ref_png_dir,
        pattern="*.png",
    )

    bg = rgb_ref[ref_index].astype("float32") / 255.0

    return bg


def create_clean_background_from_reference_folder(
    ref_png_dir,
    first_box_xyxy,
    ref_index=0,
    pattern="*.png",
    prompt="smoke",
    min_area_px=300,
    dilation_iterations=50,
    inpaint_radius=60,
    device="cuda",
    out_dir=None,
    save_qc=True,
):
    from pathlib import Path
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from .io import load_png_sequence, ensure_dir
    from .segmentation import sam3_predict_box, clean_mask

    rgb_ref, times_ref, files_ref = load_png_sequence(
        ref_png_dir,
        pattern=pattern,
    )

    ref_rgb = rgb_ref[ref_index].astype("float32") / 255.0

    raw_mask = sam3_predict_box(
        rgb=rgb_ref[ref_index],
        box_xyxy=first_box_xyxy,
        prompt=prompt,
        device=device,
    )

    ref_mask = clean_mask(
        raw_mask,
        min_area_px=min_area_px,
        fill_holes=True,
    )

    ref_mask_dil = binary_dilation(
        ref_mask,
        iterations=dilation_iterations,
    )

    ref_u8 = np.clip(ref_rgb * 255, 0, 255).astype("uint8")
    mask_u8 = ref_mask_dil.astype("uint8") * 255

    bg_u8 = cv2.inpaint(
        ref_u8,
        mask_u8,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    bg = bg_u8.astype("float32") / 255.0

    if out_dir is not None:
        out_dir = ensure_dir(out_dir)

        np.save(out_dir / "background_ref_mask.npy", ref_mask.astype("uint8"))
        np.save(out_dir / "background_ref_mask_dilated.npy", ref_mask_dil.astype("uint8"))
        np.save(out_dir / "background_clean.npy", bg.astype("float32"))

        if save_qc:
            fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)

            axes[0].imshow(ref_rgb)
            axes[0].set_title("Reference frame")

            axes[1].imshow(ref_rgb)
            axes[1].imshow(np.where(ref_mask, 1, np.nan), alpha=0.45)
            axes[1].set_title("SAM3 plume mask")

            axes[2].imshow(ref_mask_dil, cmap="gray")
            axes[2].set_title("Dilated inpainting mask")

            axes[3].imshow(bg)
            axes[3].set_title("Cleaned background")

            for ax in axes:
                ax.axis("off")

            fig.savefig(out_dir / "background_reference_cleaning_qc.png", dpi=150)
            plt.close(fig)

    return bg, ref_mask, ref_mask_dil
