from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.interpolate import griddata

from .io import ensure_dir

from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    maximum_filter,
    median_filter,
)



#def luminance(a):
#    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def smooth_inside_mask(field, mask, sigma=8):
    """
    Smooth field only inside mask.
    Keeps NaNs outside the mask.
    """
    valid = np.isfinite(field) & mask

    data = np.where(valid, field, 0.0)
    weights = valid.astype("float32")

    data_s = gaussian_filter(data, sigma=sigma)
    weights_s = gaussian_filter(weights, sigma=sigma)

    out = data_s / (weights_s + 1e-6)
    out[~mask] = np.nan
    return out


def normalize_by_region(alpha, region_mask, percentiles=(5, 95)):
    out = np.full_like(alpha, np.nan, dtype="float32")

    v = alpha[region_mask]
    v = v[np.isfinite(v)]

    if v.size > 20:
        lo, hi = np.nanpercentile(v, percentiles)
        out = np.clip((alpha - lo) / (hi - lo + 1e-6), 0, 1).astype("float32")
    else:
        lo, hi = np.nan, np.nan

    return out, float(lo), float(hi)


def bridge_horizon(alpha_norm_blend, plume, sky_weight, low=0.15, high=0.85, dilation_iterations=20):
    bridge_zone = plume & (sky_weight > low) & (sky_weight < high)
    bridge_zone = binary_dilation(bridge_zone, iterations=dilation_iterations) & plume

    known = plume & (~bridge_zone) & np.isfinite(alpha_norm_blend)

    H, W = alpha_norm_blend.shape
    yy, xx = np.mgrid[0:H, 0:W]

    alpha_bridge = alpha_norm_blend.copy()

    if known.sum() < 10 or bridge_zone.sum() == 0:
        alpha_bridge[~plume] = np.nan
        return alpha_bridge, bridge_zone

    points = np.column_stack([xx[known], yy[known]])
    values = alpha_norm_blend[known]

    target = bridge_zone

    interp_vals = griddata(
        points,
        values,
        (xx[target], yy[target]),
        method="linear",
    )

    bad = ~np.isfinite(interp_vals)
    if np.any(bad):
        interp_nearest = griddata(
            points,
            values,
            (xx[target], yy[target]),
            method="nearest",
        )
        interp_vals[bad] = interp_nearest[bad]

    alpha_bridge[target] = interp_vals
    alpha_bridge[~plume] = np.nan

    return alpha_bridge, bridge_zone


def retrieve_opacity_frame(
    img,
    bg,
    plume,
    smoke_rgb=(0.8, 0.8, 0.8),
    sky_threshold=0.62,
    sky_weight_sigma=50,
    sky_smooth_sigma=2,
    terrain_sky_split=0.5,
    percentiles=(5, 95),
    horizon_low=0.15,
    horizon_high=0.85,
    horizon_dilation_iterations=20,
    horizon_weight_cap=0.4,
    final_smooth_sigma=8,
):
    """
    Retrieve visual smoke opacity index for one frame.

    This is the packaged version of the notebook section:
    'Trying to find the right one' + normalization + horizon bridge.

    Important:
    - all normalization is same-frame only
    - no cross-frame empirical normalization is used
    """
    img = img.astype("float32")
    bg = bg.astype("float32")

    if img.max() > 1.5:
        img = img / 255.0
    if bg.max() > 1.5:
        bg = bg / 255.0

    plume = plume.astype(bool)

    L_img = luminance(img)
    L_bg = luminance(bg)

    # Terrain / sky separation from background luminance
    sky_weight = gaussian_filter((L_bg > sky_threshold).astype("float32"), sigma=sky_weight_sigma)
    sky_weight = np.clip(sky_weight, 0, 1)

    terrain_part = plume & (sky_weight < terrain_sky_split)
    sky_part = plume & (sky_weight >= terrain_sky_split)

    # Terrain: smoke brightening model
    smoke_rgb = np.array(smoke_rgb, dtype="float32")
    diff_obs = img - bg
    diff_smoke = smoke_rgb[None, None, :] - bg

    den = np.sum(diff_smoke**2, axis=-1)
    alpha_terrain = np.sum(diff_obs * diff_smoke, axis=-1) / (den + 1e-6)
    alpha_terrain = np.clip(alpha_terrain, 0, 1).astype("float32")

    # Sky: simple attenuation model
    alpha_sky = 1.0 - L_img / (L_bg + 1e-6)
    alpha_sky = np.clip(alpha_sky, 0, 1).astype("float32")

    if sky_smooth_sigma and sky_smooth_sigma > 0:
        alpha_sky = gaussian_filter(alpha_sky, sigma=sky_smooth_sigma).astype("float32")

    # Raw combined diagnostic
    alpha_raw = np.full_like(L_img, np.nan, dtype="float32")
    alpha_raw[terrain_part] = alpha_terrain[terrain_part]
    alpha_raw[sky_part] = alpha_sky[sky_part]

    # Normalize terrain retrieval inside same-frame terrain plume pixels
    alpha_terrain_norm, lo_t, hi_t = normalize_by_region(
        alpha_terrain,
        terrain_part,
        percentiles=percentiles,
    )

    # Horizon transition zone and sky core, as in notebook
    horizon_band = (sky_weight > 0.2) & (sky_weight < 0.8)
    horizon_band = binary_dilation(horizon_band, iterations=20)

    sky_core = plume & (sky_weight > 0.85) & (~horizon_band)

    # If sky_core is too small, fall back to sky_part
    sky_norm_region = sky_core if sky_core.sum() > 20 else sky_part

    alpha_sky_norm, lo_s, hi_s = normalize_by_region(
        alpha_sky,
        sky_norm_region,
        percentiles=percentiles,
    )

    sky_weight2 = sky_weight.copy()
    sky_weight2[horizon_band] = np.minimum(sky_weight2[horizon_band], horizon_weight_cap)

    alpha_norm_blend = (
        (1.0 - sky_weight2) * alpha_terrain_norm
        + sky_weight2 * alpha_sky_norm
    ).astype("float32")
    alpha_norm_blend[~plume] = np.nan

    # Bridge/interpolate horizon transition
    alpha_bridge, bridge_zone = bridge_horizon(
        alpha_norm_blend,
        plume,
        sky_weight,
        low=horizon_low,
        high=horizon_high,
        dilation_iterations=horizon_dilation_iterations,
    )

    # Smooth only the bridge zone, as in the notebook
    if final_smooth_sigma and final_smooth_sigma > 0:
        alpha_bridge_smooth = smooth_inside_mask(alpha_bridge, plume, sigma=final_smooth_sigma)
        alpha_vis = alpha_bridge.copy()
        alpha_vis[bridge_zone] = alpha_bridge_smooth[bridge_zone]
    else:
        alpha_vis = alpha_bridge.copy()

    alpha_vis[~plume] = np.nan

    return {
        "alpha_vis": alpha_vis.astype("float32"),
        "alpha_raw": alpha_raw.astype("float32"),
        "alpha_terrain": alpha_terrain.astype("float32"),
        "alpha_sky": alpha_sky.astype("float32"),
        "alpha_terrain_norm": alpha_terrain_norm.astype("float32"),
        "alpha_sky_norm": alpha_sky_norm.astype("float32"),
        "alpha_norm_blend": alpha_norm_blend.astype("float32"),
        "alpha_bridge": alpha_bridge.astype("float32"),
        "sky_weight": sky_weight.astype("float32"),
        "sky_weight2": sky_weight2.astype("float32"),
        "terrain_part": terrain_part.astype("uint8"),
        "sky_part": sky_part.astype("uint8"),
        "sky_core": sky_core.astype("uint8"),
        "horizon_band": horizon_band.astype("uint8"),
        "bridge_zone": bridge_zone.astype("uint8"),
        "lo_t": lo_t,
        "hi_t": hi_t,
        "lo_s": lo_s,
        "hi_s": hi_s,
    }


def retrieve_opacity_dataset(
    ds_track,
    bg,
    out_dir=None,
    save_qc=True,
    every_n_frames=1,
    **kwargs,
):
    """
    Run opacity retrieval for all frames in ds_track.

    Parameters
    ----------
    ds_track : xr.Dataset
        Output from run_sam3_primitive_tracking.
    bg : np.ndarray
        Clean background RGB, range 0..1 or 0..255.
    out_dir : path-like, optional
        If given, save smoke_opacity.nc and QC PNGs.

    Returns
    -------
    ds_opacity : xr.Dataset
    """
    nt = ds_track.sizes["time"]
    y = ds_track.y.values
    x = ds_track.x.values

    results = []

    for i in range(nt):
        print(f"Opacity {i+1}/{nt}")

        img = ds_track.rgb.isel(time=i).values.astype("float32") / 255.0
        plume = ds_track.plume_mask.isel(time=i).values.astype(bool)

        res = retrieve_opacity_frame(img, bg, plume, **kwargs)
        results.append(res)

        if out_dir is not None and save_qc and (i % every_n_frames == 0):
            out_dir_p = ensure_dir(out_dir)
            qc_dir = ensure_dir(out_dir_p / "opacity_qc")
            save_opacity_qc(
                img,
                bg,
                res["alpha_vis"],
                qc_dir / f"opacity_qc_{i:04d}.png",
                title=f"Frame {i}",
            )

    stack_names = [
        "alpha_vis",
        "alpha_raw",
        "alpha_terrain",
        "alpha_sky",
        "alpha_terrain_norm",
        "alpha_sky_norm",
        "alpha_norm_blend",
        "alpha_bridge",
        "sky_weight",
        "sky_weight2",
        "terrain_part",
        "sky_part",
        "sky_core",
        "horizon_band",
        "bridge_zone",
    ]

    data_vars = {}
    for name in stack_names:
        arr = np.stack([r[name] for r in results], axis=0)
        data_vars[name] = (("time", "y", "x"), arr)

    for scalar in ["lo_t", "hi_t", "lo_s", "hi_s"]:
        data_vars[scalar] = (("time",), np.asarray([r[scalar] for r in results], dtype="float32"))

    ds_opacity = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ds_track.time.values,
            "y": y,
            "x": x,
        },
        attrs={
            "description": "Visual smoke opacity retrieval from segmented timelapse frames",
            "normalization": "same-frame terrain/sky robust percentiles only",
        },
    )

    if out_dir is not None:
        out_dir_p = ensure_dir(out_dir)
        ds_opacity.to_netcdf(out_dir_p / "smoke_opacity.nc")

    return ds_opacity


def save_opacity_qc(img, bg, alpha_vis, out_png, title=None):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    axes[0].imshow(bg)
    axes[0].set_title("Background reference")

    axes[1].imshow(img)
    axes[1].set_title("Current frame")

    axes[2].imshow(img)
    im = axes[2].imshow(
        alpha_vis,
        cmap="inferno_r",
        vmin=0,
        vmax=1,
        alpha=0.10 + 0.80 * np.nan_to_num(alpha_vis),
    )
    axes[2].set_title("Visual smoke opacity index" if title is None else title)

    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized smoke opacity index")

    for ax in axes:
        ax.axis("off")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    

def luminance(img):
    """
    RGB luminance in [0, 1].

    Accepts uint8 [0, 255] or float [0, 1].
    """
    img = img.astype("float32")

    if np.nanmax(img) > 2:
        img = img / 255.0

    return (
        0.2126 * img[..., 0]
        + 0.7152 * img[..., 1]
        + 0.0722 * img[..., 2]
    ).astype("float32")



def nan_gaussian_mean(values, valid, sigma):
    """
    Gaussian-weighted local mean, ignoring invalid pixels.
    """
    values = values.astype("float32")
    valid = valid.astype(bool) & np.isfinite(values)

    num = np.where(valid, values, 0.0)
    den = np.where(valid, 1.0, 0.0)

    num_f = gaussian_filter(num, sigma=sigma)
    den_f = gaussian_filter(den, sigma=sigma)

    out = np.full_like(values, np.nan, dtype="float32")
    good = den_f > 1e-6
    out[good] = num_f[good] / den_f[good]

    return out

# ============================================================
# Hybrid opacity retrieval with interface-aware cross-calibration
# ============================================================

def retrieve_hybrid_opacity(
    img,
    bg,
    plume,
    sky_weight,
    smoke_lum=0.95,
    gain_terrain=1.0,
    gain_sky=1.0,
    sky_weight_smooth_sigma=2.0,
    smoothstep=True,
    # ------------------------------------------------------------
    # Interface-aware cross-calibration parameters
    # ------------------------------------------------------------
    interface_threshold=0.5,
    support_inner_px=3,
    support_outer_px=15,
    terrain_conf_max_w=0.25,
    sky_conf_min_w=0.75,
    local_scale_sigma=20,
    min_support_pixels=20,
    scale_clip=(0.25, 4.0),
    calibration_strength=1.0,
    # ------------------------------------------------------------
    # Seam-repair parameters
    # ------------------------------------------------------------
    seam_repair=True,
    seam_filter_size=5,
    seam_gap_fraction=0.45,
    seam_min_local_max=0.05,
    return_diagnostics=False,
):
    """
    Hybrid smoke opacity retrieval with terrain--sky interface-aware
    cross-calibration.

    The retrieval first estimates two opacity branches:

    terrain branch:
        alpha_t = (L_img - L_bg) / (L_smoke - L_bg)

    sky branch:
        alpha_s = (L_bg - L_img) / L_bg

    These branches are then locally harmonised near the terrain--sky
    interface using support bands just below and just above the interface.
    The locally scaled sky branch is blended with the terrain branch using
    the smoothed sky-weight field.
    """

    # ------------------------------------------------------------
    # Luminance fields
    # ------------------------------------------------------------
    L_img = luminance(img)
    L_bg = luminance(bg)

    plume = plume.astype(bool)

    # ------------------------------------------------------------
    # Smooth only the terrain/sky blending weight.
    # Do NOT smooth the opacity retrieval itself.
    # ------------------------------------------------------------
    w_raw = np.clip(sky_weight.astype("float32"), 0, 1)
    w = w_raw.copy()

    if sky_weight_smooth_sigma is not None and sky_weight_smooth_sigma > 0:
        w = gaussian_filter(w, sigma=sky_weight_smooth_sigma)

    w = np.clip(w, 0, 1)

    if smoothstep:
        # Smooth transition between terrain and sky models.
        w = w * w * (3.0 - 2.0 * w)

    # ------------------------------------------------------------
    # Terrain brightening model
    # Smoke over darker terrain often makes pixels brighter.
    # ------------------------------------------------------------
    denom_terrain = smoke_lum - L_bg

    alpha_terrain = (L_img - L_bg) / (denom_terrain + 1e-6)

    # Avoid unstable retrieval where background is already too bright
    # relative to the assumed smoke luminance.
    alpha_terrain = np.where(
        denom_terrain > 0.02,
        alpha_terrain,
        0.0,
    )

    # ------------------------------------------------------------
    # Sky attenuation model
    # Smoke over bright sky often makes pixels darker.
    # ------------------------------------------------------------
    alpha_sky = (L_bg - L_img) / (L_bg + 1e-6)

    alpha_terrain = np.clip(alpha_terrain, 0, 1).astype("float32")
    alpha_sky = np.clip(alpha_sky, 0, 1).astype("float32")

    # ------------------------------------------------------------
    # Apply gains
    # ------------------------------------------------------------
    alpha_terrain_g = np.clip(gain_terrain * alpha_terrain, 0, 1).astype("float32")
    alpha_sky_g = np.clip(gain_sky * alpha_sky, 0, 1).astype("float32")

    # ------------------------------------------------------------
    # Signed distance to the terrain/sky interface
    #
    # signed_dist > 0 : sky side
    # signed_dist < 0 : terrain side
    # ------------------------------------------------------------
    sky_binary = w >= interface_threshold

    dist_to_terrain = distance_transform_edt(sky_binary)
    dist_to_sky = distance_transform_edt(~sky_binary)

    signed_dist = (dist_to_terrain - dist_to_sky).astype("float32")

    # ------------------------------------------------------------
    # Support bands just below and above the interface.
    # These exclude the immediate boundary, which is the unstable zone.
    # ------------------------------------------------------------
    terrain_support = (
        plume
        & (signed_dist <= -support_inner_px)
        & (signed_dist >= -support_outer_px)
        & (w <= terrain_conf_max_w)
        & np.isfinite(alpha_terrain_g)
        & (alpha_terrain_g > 0)
    )

    sky_support = (
        plume
        & (signed_dist >= support_inner_px)
        & (signed_dist <= support_outer_px)
        & (w >= sky_conf_min_w)
        & np.isfinite(alpha_sky_g)
        & (alpha_sky_g > 0)
    )

    n_terrain_support = int(np.sum(terrain_support))
    n_sky_support = int(np.sum(sky_support))

    # ------------------------------------------------------------
    # Robust fallback scales in case support pixels are sparse.
    # ------------------------------------------------------------
    terrain_fallback_mask = plume & (w < 0.35) & np.isfinite(alpha_terrain_g)
    sky_fallback_mask = plume & (w > 0.65) & np.isfinite(alpha_sky_g)

    if n_terrain_support >= min_support_pixels:
        terrain_global = np.nanmedian(alpha_terrain_g[terrain_support])
    elif np.any(terrain_fallback_mask):
        terrain_global = np.nanmedian(alpha_terrain_g[terrain_fallback_mask])
    else:
        terrain_global = np.nanmedian(alpha_terrain_g[plume])

    if n_sky_support >= min_support_pixels:
        sky_global = np.nanmedian(alpha_sky_g[sky_support])
    elif np.any(sky_fallback_mask):
        sky_global = np.nanmedian(alpha_sky_g[sky_fallback_mask])
    else:
        sky_global = np.nanmedian(alpha_sky_g[plume])

    if not np.isfinite(terrain_global):
        terrain_global = 0.0

    if not np.isfinite(sky_global) or sky_global <= 1e-6:
        sky_global = 1e-6

    global_scale = float(np.clip(
        terrain_global / (sky_global + 1e-6),
        scale_clip[0],
        scale_clip[1],
    ))

    # ------------------------------------------------------------
    # Local opacity scales propagated from support bands.
    # ------------------------------------------------------------
    terrain_local = nan_gaussian_mean(
        alpha_terrain_g,
        valid=terrain_support,
        sigma=local_scale_sigma,
    )

    sky_local = nan_gaussian_mean(
        alpha_sky_g,
        valid=sky_support,
        sigma=local_scale_sigma,
    )

    terrain_ref = np.where(
        np.isfinite(terrain_local),
        terrain_local,
        terrain_global,
    ).astype("float32")

    sky_ref = np.where(
        np.isfinite(sky_local) & (sky_local > 1e-6),
        sky_local,
        sky_global,
    ).astype("float32")

    # ------------------------------------------------------------
    # Local sky-to-terrain scale factor.
    # ------------------------------------------------------------
    sky_to_terrain_scale = terrain_ref / (sky_ref + 1e-6)
    sky_to_terrain_scale = np.clip(
        sky_to_terrain_scale,
        scale_clip[0],
        scale_clip[1],
    ).astype("float32")

    sky_to_terrain_scale = np.where(
        np.isfinite(sky_to_terrain_scale),
        sky_to_terrain_scale,
        global_scale,
    ).astype("float32")

    alpha_sky_calibrated = np.clip(
        alpha_sky_g * sky_to_terrain_scale,
        0,
        1,
    ).astype("float32")

    # ------------------------------------------------------------
    # Interface calibration strength.
    # 4*w*(1-w) is strongest at w=0.5 and zero at w=0 or w=1.
    # ------------------------------------------------------------
    lambda_interface = 4.0 * w * (1.0 - w)
    lambda_interface = np.clip(
        calibration_strength * lambda_interface,
        0,
        1,
    ).astype("float32")
    lambda_interface = np.where(plume, lambda_interface, 0.0).astype("float32")

    # Effective sky branch used in the blend:
    # - raw sky branch away from the interface
    # - cross-calibrated sky branch near the interface
    alpha_sky_eff = (
        (1.0 - lambda_interface) * alpha_sky_g
        + lambda_interface * alpha_sky_calibrated
    ).astype("float32")

    # ------------------------------------------------------------
    # Hybrid blend after cross-calibration.
    # ------------------------------------------------------------
    alpha_linear = (
        (1.0 - w) * alpha_terrain_g
        + w * alpha_sky_eff
    ).astype("float32")

    alpha = alpha_linear.copy()

    # ------------------------------------------------------------
    # Sign-aware fallback near terrain/sky interface.
    #
    # If L_img > L_bg, the sky attenuation model gives zero after clipping.
    # Near the horizon, this can wrongly suppress real opacity.
    # ------------------------------------------------------------
    diff = L_img - L_bg

    brightening = diff > 0
    transition = plume & (w > 0.05) & (w < 0.98)

    use_fallback = transition & brightening

    alpha[use_fallback] = np.maximum(
        alpha_linear[use_fallback],
        alpha_terrain_g[use_fallback],
    )

    alpha = np.clip(alpha, 0, 1).astype("float32")

    # ------------------------------------------------------------
    # Targeted seam repair.
    # This should now be less important, but remains a safety step.
    # ------------------------------------------------------------
    seam_gap = np.zeros_like(plume, dtype=bool)

    if seam_repair:
        if np.any(transition):
            alpha_for_filter = np.where(np.isfinite(alpha), alpha, 0.0)

            alpha_local_max = maximum_filter(
                alpha_for_filter,
                size=seam_filter_size,
            )

            alpha_local_med = median_filter(
                alpha_for_filter,
                size=seam_filter_size,
            )

            seam_gap = (
                transition
                & np.isfinite(alpha)
                & (alpha < seam_gap_fraction * alpha_local_max)
                & (alpha_local_max > seam_min_local_max)
            )

            alpha_repair = 0.5 * alpha_local_med + 0.5 * alpha_local_max

            alpha[seam_gap] = np.maximum(
                alpha[seam_gap],
                alpha_repair[seam_gap],
            )

    alpha = np.clip(alpha, 0, 1).astype("float32")

    # ------------------------------------------------------------
    # Mask outside plume.
    # ------------------------------------------------------------
    alpha_out = alpha.copy()
    alpha_terrain_out = alpha_terrain.copy()
    alpha_sky_out = alpha_sky.copy()

    alpha_out[~plume] = np.nan
    alpha_terrain_out[~plume] = np.nan
    alpha_sky_out[~plume] = np.nan

    if not return_diagnostics:
        return alpha_out, alpha_terrain_out, alpha_sky_out

    diagnostics = {
        "L_img": L_img,
        "L_bg": L_bg,
        "L_diff": diff,
        "plume": plume,
        "sky_weight_raw": w_raw,
        "sky_weight": w,
        "sky_binary": sky_binary,
        "signed_dist": signed_dist,
        "terrain_support": terrain_support,
        "sky_support": sky_support,
        "n_terrain_support": n_terrain_support,
        "n_sky_support": n_sky_support,
        "alpha_terrain_raw": alpha_terrain_out,
        "alpha_sky_raw": alpha_sky_out,
        "alpha_terrain_g": np.where(plume, alpha_terrain_g, np.nan),
        "alpha_sky_g": np.where(plume, alpha_sky_g, np.nan),
        "terrain_local": terrain_local,
        "sky_local": sky_local,
        "terrain_ref": terrain_ref,
        "sky_ref": sky_ref,
        "terrain_global": terrain_global,
        "sky_global": sky_global,
        "global_scale": global_scale,
        "sky_to_terrain_scale": sky_to_terrain_scale,
        "alpha_sky_calibrated": np.where(plume, alpha_sky_calibrated, np.nan),
        "lambda_interface": lambda_interface,
        "alpha_sky_eff": np.where(plume, alpha_sky_eff, np.nan),
        "alpha_linear": np.where(plume, alpha_linear, np.nan),
        "use_fallback": use_fallback,
        "seam_gap": seam_gap,
        "alpha_final": alpha_out,
    }

    return alpha_out, alpha_terrain_out, alpha_sky_out, diagnostics

