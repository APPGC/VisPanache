"""Exact original notebook overlay renderer; source provenance in docs/source_lineage.json."""
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
