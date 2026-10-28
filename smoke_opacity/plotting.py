

def plot_hybrid_opacity_method_summary(
    img,
    alpha,
    diagnostics,
    out_png=None,
    figsize=(16, 9),
    cmap_alpha="inferno_r",
    dpi=200,
):
    """
    Cleaner method-summary figure for presentations/paper drafts.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plume = diagnostics["plume"]
    w = diagnostics["sky_weight"]
    terrain_support = diagnostics["terrain_support"]
    sky_support = diagnostics["sky_support"]

    support_rgb = np.zeros((*plume.shape, 3), dtype="float32")
    support_rgb[terrain_support] = [1.0, 0.45, 0.05]
    support_rgb[sky_support] = [0.05, 0.45, 1.0]

    panels = [
        ("(a) Observed image", img, None, None, None),
        ("(b) Terrain--sky weight", w, "viridis", 0, 1),
        ("(c) Support bands", None, None, None, None),
        ("(d) Terrain branch", diagnostics["alpha_terrain_g"], cmap_alpha, 0, 1),
        ("(e) Sky branch", diagnostics["alpha_sky_g"], cmap_alpha, 0, 1),
        ("(f) Sky-to-terrain scale", np.where(plume, diagnostics["sky_to_terrain_scale"], np.nan), "magma", None, None),
        ("(g) Final hybrid opacity", alpha, cmap_alpha, 0, 1),
    ]

    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    for i, (title, arr, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[i]

        if title.startswith("(a)"):
            ax.imshow(img)
        elif title.startswith("(c)"):
            ax.imshow(img)
            ax.imshow(support_rgb, alpha=0.65)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].axis("off")

    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, axes

# updated function:
def plot_hybrid_opacity_crosscal_diagnostics(
    img,
    bg,
    alpha,
    diagnostics,
    figsize=(20, 18),
    vmin_alpha=0,
    vmax_alpha=1,
    cmap_alpha="inferno_r",
    cmap_div="coolwarm",
    savepath=None,
    out_png=None,
    dpi=150,
):
    """
    Diagnostic figure for the interface-aware cross-calibrated
    hybrid smoke opacity retrieval.

    This function does not recompute opacity. It only visualises outputs from
    retrieve_hybrid_opacity_crosscal().

    Panels include:
    - observed image and background
    - plume mask and sky weight
    - signed distance to terrain/sky interface
    - terrain and sky support bands
    - raw terrain and sky opacity branches
    - local terrain and sky opacity references
    - sky-to-terrain scale factor
    - calibration strength
    - sky branch after local scaling
    - effective sky branch used in the blend
    - hybrid opacity before seam repair
    - final opacity without seam overlay
    - final opacity with seam contour
    - seam-repaired pixels only
    - opacity added by seam repair
    - numerical summary

    Parameters
    ----------
    img : ndarray
        Observed RGB image.
    bg : ndarray
        Background RGB image.
    alpha : ndarray
        Final opacity field.
    diagnostics : dict
        Diagnostics returned by retrieve_hybrid_opacity_crosscal().
    figsize : tuple
        Figure size.
    vmin_alpha, vmax_alpha : float
        Opacity colour scale limits.
    cmap_alpha : str
        Opacity colormap. Default is "inferno_r" so low opacity is light/yellow
        and high opacity is dark/purple/black.
    cmap_div : str
        Diverging colormap for signed distance.
    savepath, out_png : str or Path, optional
        Output path. `out_png` is kept for backward compatibility.
    dpi : int
        Figure resolution.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Backward-compatible output keyword
    # ------------------------------------------------------------
    if savepath is None and out_png is not None:
        savepath = out_png

    # ------------------------------------------------------------
    # Retrieve diagnostic fields
    # ------------------------------------------------------------
    plume = diagnostics["plume"]
    w = diagnostics["sky_weight"]
    signed_dist = diagnostics["signed_dist"]

    terrain_support = diagnostics["terrain_support"]
    sky_support = diagnostics["sky_support"]

    alpha_terrain = diagnostics["alpha_terrain_g"]
    alpha_sky = diagnostics["alpha_sky_g"]
    alpha_sky_cal = diagnostics["alpha_sky_calibrated"]
    alpha_sky_eff = diagnostics["alpha_sky_eff"]
    scale = diagnostics["sky_to_terrain_scale"]
    lambda_interface = diagnostics["lambda_interface"]
    alpha_linear = diagnostics["alpha_linear"]
    seam_gap = diagnostics["seam_gap"]

    # ------------------------------------------------------------
    # Support-band overlay image
    # ------------------------------------------------------------
    support_rgb = np.zeros((*plume.shape, 3), dtype="float32")
    support_rgb[terrain_support] = [1.0, 0.45, 0.05]  # orange
    support_rgb[sky_support] = [0.05, 0.45, 1.0]      # blue

    masked_signed_dist = np.where(plume, signed_dist, np.nan)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        5,
        4,
        figsize=figsize,
        constrained_layout=True,
    )
    axes = axes.ravel()

    # ------------------------------------------------------------
    # 1. Observed image
    # ------------------------------------------------------------
    axes[0].imshow(img)
    axes[0].set_title("Observed image")

    # ------------------------------------------------------------
    # 2. Background image
    # ------------------------------------------------------------
    axes[1].imshow(bg)
    axes[1].set_title("Background image")

    # ------------------------------------------------------------
    # 3. Plume mask
    # ------------------------------------------------------------
    im = axes[2].imshow(plume, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Plume mask")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    # ------------------------------------------------------------
    # 4. Sky weight
    # ------------------------------------------------------------
    im = axes[3].imshow(w, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Smoothed sky weight\n0=terrain, 1=sky")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    # ------------------------------------------------------------
    # 5. Signed distance to interface
    # ------------------------------------------------------------
    if np.any(np.isfinite(masked_signed_dist)):
        vmax_dist = np.nanpercentile(np.abs(masked_signed_dist), 98)
        if not np.isfinite(vmax_dist) or vmax_dist <= 0:
            vmax_dist = 1
    else:
        vmax_dist = 1

    im = axes[4].imshow(
        masked_signed_dist,
        cmap=cmap_div,
        vmin=-vmax_dist,
        vmax=vmax_dist,
    )
    axes[4].set_title("Signed distance to interface\nnegative=terrain, positive=sky")
    fig.colorbar(im, ax=axes[4], fraction=0.046)

    # ------------------------------------------------------------
    # 6. Support bands
    # ------------------------------------------------------------
    axes[5].imshow(img)
    axes[5].imshow(support_rgb, alpha=0.65)
    axes[5].set_title("Support bands\norange=terrain, blue=sky")

    # ------------------------------------------------------------
    # 7. Terrain branch before cross-calibration
    # ------------------------------------------------------------
    im = axes[6].imshow(
        alpha_terrain,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[6].set_title("Terrain opacity branch\nbefore cross-calibration")
    fig.colorbar(im, ax=axes[6], fraction=0.046)

    # ------------------------------------------------------------
    # 8. Sky branch before cross-calibration
    # ------------------------------------------------------------
    im = axes[7].imshow(
        alpha_sky,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[7].set_title("Sky opacity branch\nbefore cross-calibration")
    fig.colorbar(im, ax=axes[7], fraction=0.046)

    # ------------------------------------------------------------
    # 9. Local terrain reference
    # ------------------------------------------------------------
    im = axes[8].imshow(
        np.where(plume, diagnostics["terrain_ref"], np.nan),
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[8].set_title("Local terrain opacity reference")
    fig.colorbar(im, ax=axes[8], fraction=0.046)

    # ------------------------------------------------------------
    # 10. Local sky reference
    # ------------------------------------------------------------
    im = axes[9].imshow(
        np.where(plume, diagnostics["sky_ref"], np.nan),
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[9].set_title("Local sky opacity reference")
    fig.colorbar(im, ax=axes[9], fraction=0.046)

    # ------------------------------------------------------------
    # 11. Sky-to-terrain scale factor
    # ------------------------------------------------------------
    im = axes[10].imshow(
        np.where(plume, scale, np.nan),
        cmap="magma",
    )
    axes[10].set_title(
        "Sky-to-terrain scale factor\n"
        f"global fallback={diagnostics['global_scale']:.2f}"
    )
    fig.colorbar(im, ax=axes[10], fraction=0.046)

    # ------------------------------------------------------------
    # 12. Calibration strength
    # ------------------------------------------------------------
    im = axes[11].imshow(
        np.where(plume, lambda_interface, np.nan),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[11].set_title("Calibration strength\nstrongest near interface")
    fig.colorbar(im, ax=axes[11], fraction=0.046)

    # ------------------------------------------------------------
    # 13. Sky branch after local scaling
    # ------------------------------------------------------------
    im = axes[12].imshow(
        alpha_sky_cal,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[12].set_title("Sky opacity branch\nafter local scaling")
    fig.colorbar(im, ax=axes[12], fraction=0.046)

    # ------------------------------------------------------------
    # 14. Effective sky branch used in blend
    # ------------------------------------------------------------
    im = axes[13].imshow(
        alpha_sky_eff,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[13].set_title("Effective sky branch\nused in blend")
    fig.colorbar(im, ax=axes[13], fraction=0.046)

    # ------------------------------------------------------------
    # 15. Hybrid blend before seam repair
    # ------------------------------------------------------------
    im = axes[14].imshow(
        alpha_linear,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[14].set_title("Hybrid blend\nbefore seam repair")
    fig.colorbar(im, ax=axes[14], fraction=0.046)

    # ------------------------------------------------------------
    # 16. Final opacity only
    # ------------------------------------------------------------
    im = axes[15].imshow(
        alpha,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )
    axes[15].set_title("Final opacity\nno seam overlay")
    fig.colorbar(im, ax=axes[15], fraction=0.046)

    # ------------------------------------------------------------
    # 17. Final opacity with seam repair contour
    # ------------------------------------------------------------
    im = axes[16].imshow(
        alpha,
        cmap=cmap_alpha,
        vmin=vmin_alpha,
        vmax=vmax_alpha,
    )

    if np.any(seam_gap):
        axes[16].contour(
            seam_gap.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=0.8,
        )

    axes[16].set_title("Final opacity + seam contour\ncyan contour = repaired zone")
    fig.colorbar(im, ax=axes[16], fraction=0.046)

    # ------------------------------------------------------------
    # 18. Seam-repaired pixels only
    # ------------------------------------------------------------
    im = axes[17].imshow(
        seam_gap,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[17].set_title(
        f"Seam-repaired pixels only\nn={int(np.sum(seam_gap))}"
    )
    fig.colorbar(im, ax=axes[17], fraction=0.046)

    # ------------------------------------------------------------
    # 19. Opacity added by seam repair
    #
    # Show only repaired pixels, otherwise the panel is dominated by
    # zero/no-change values over the full plume.
    # ------------------------------------------------------------
    seam_delta = alpha - alpha_linear
    seam_delta = np.where(seam_gap, seam_delta, np.nan)

    if np.any(np.isfinite(seam_delta)):
        vmax_delta = np.nanpercentile(seam_delta, 99)
        vmax_delta = max(vmax_delta, 0.02)
    else:
        vmax_delta = 0.02

    im = axes[18].imshow(
        seam_delta,
        cmap="magma_r",
        vmin=0,
        vmax=vmax_delta,
    )
    axes[18].set_title("Opacity added by seam repair\nshown only on repaired pixels")
    fig.colorbar(im, ax=axes[18], fraction=0.046)

    # ------------------------------------------------------------
    # 20. Numerical summary
    # ------------------------------------------------------------
    axes[19].axis("off")

    valid_scale = scale[plume & np.isfinite(scale)]
    if valid_scale.size > 0:
        scale_p05 = np.nanpercentile(valid_scale, 5)
        scale_p50 = np.nanpercentile(valid_scale, 50)
        scale_p95 = np.nanpercentile(valid_scale, 95)
    else:
        scale_p05 = np.nan
        scale_p50 = np.nan
        scale_p95 = np.nan

    valid_alpha = alpha[plume & np.isfinite(alpha)]
    if valid_alpha.size > 0:
        alpha_p50 = np.nanpercentile(valid_alpha, 50)
        alpha_p90 = np.nanpercentile(valid_alpha, 90)
        alpha_max = np.nanmax(valid_alpha)
    else:
        alpha_p50 = np.nan
        alpha_p90 = np.nan
        alpha_max = np.nan

    axes[19].text(
        0.02,
        0.95,
        "Cross-calibration summary\n"
        "-------------------------\n"
        f"Terrain support pixels : {diagnostics['n_terrain_support']}\n"
        f"Sky support pixels     : {diagnostics['n_sky_support']}\n"
        f"Terrain global ref     : {diagnostics['terrain_global']:.3f}\n"
        f"Sky global ref         : {diagnostics['sky_global']:.3f}\n"
        f"Global scale           : {diagnostics['global_scale']:.3f}\n"
        "\n"
        "Scale inside plume\n"
        f"p05 / p50 / p95        : {scale_p05:.3f} / {scale_p50:.3f} / {scale_p95:.3f}\n"
        "\n"
        "Final opacity inside plume\n"
        f"p50 / p90 / max        : {alpha_p50:.3f} / {alpha_p90:.3f} / {alpha_max:.3f}\n"
        "\n"
        f"Seam pixels            : {int(np.sum(seam_gap))}",
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    # ------------------------------------------------------------
    # Cosmetic cleanup
    # ------------------------------------------------------------
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # ------------------------------------------------------------
    # Save if requested
    # ------------------------------------------------------------
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, axes

def plot_opacity_overlay_frame(
    img,
    alpha,
    out_png,
    dpi=150,
):
    import numpy as np
    import matplotlib.pyplot as plt

    alpha_plot = np.where(np.isfinite(alpha), alpha, np.nan)

    # Transparency cannot contain NaNs
    alpha_transparency = np.where(
        np.isfinite(alpha),
        np.clip(alpha, 0, 1),
        0.0,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 6),
        constrained_layout=True,
    )

    axes[0].imshow(img)
    axes[0].set_title("Visible image")
    axes[0].axis("off")

    axes[1].imshow(img)

    im = axes[1].imshow(
        alpha_plot,
        cmap="inferno_r",
        vmin=0,
        vmax=1,
        alpha=alpha_transparency,
    )

    axes[1].set_title("Opacity overlay")
    axes[1].axis("off")

    cbar = plt.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Opacity")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)