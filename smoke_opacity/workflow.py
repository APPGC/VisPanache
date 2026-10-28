from .segmentation import run_sam3_primitive_tracking
from .background import create_clean_background_from_tracking
from .opacity import retrieve_opacity_dataset


def run_segmentation_then_background(
    png_dir,
    out_dir,
    first_box_xyxy,
    source_xy,
    pattern="*.png",
    prompt="smoke",
    reference_time_index=0,
    **tracking_kwargs,
):
    """
    Convenience function for your first two required stages:
    1. segment plume
    2. create the cleaned background frame

    You can still call the two stages separately in the notebook.
    """
    ds_track, records = run_sam3_primitive_tracking(
        png_dir=png_dir,
        out_dir=out_dir,
        first_box_xyxy=first_box_xyxy,
        pattern=pattern,
        prompt=prompt,
        source_xy=source_xy,
        **tracking_kwargs,
    )

    bg, ds_bg = create_clean_background_from_tracking(
        ds_track,
        reference_time_index=reference_time_index,
        out_dir=out_dir,
    )

    return ds_track, records, bg, ds_bg


def run_full_case(
    png_dir,
    out_dir,
    first_box_xyxy,
    source_xy,
    pattern="*.png",
    prompt="smoke",
    reference_time_index=0,
    **tracking_kwargs,
):
    ds_track, records, bg, ds_bg = run_segmentation_then_background(
        png_dir=png_dir,
        out_dir=out_dir,
        first_box_xyxy=first_box_xyxy,
        source_xy=source_xy,
        pattern=pattern,
        prompt=prompt,
        reference_time_index=reference_time_index,
        **tracking_kwargs,
    )

    ds_opacity = retrieve_opacity_dataset(
        ds_track,
        bg,
        out_dir=out_dir,
        save_qc=True,
    )

    return ds_track, records, bg, ds_bg, ds_opacity
