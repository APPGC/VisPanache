VisPanache paper analysis companion for frozen v1.0.0, commit 98b556fe2b1993c03621d5973ca18522eff66190.

Run with the recorded CPU environment:
python reproduce_checks.py --source /path/to/frozen/source --data /path/to/extracted/inputs --output /path/to/new/checks

Verifies every archived input and exact numerical module SHA256, then recomputes three bottom ablations, 24 controlled reference offsets on six camera/frame configurations, and 81 threshold configurations on five bottom and four top original-mask frames. Top reference-offset cases reuse full-guided masks. All relevant metrics must match the archived expected values exactly in the observed environment. Other dependency versions may require explicit floating-point comparison review. No source modification, GPU, segmentation rerun, download, installation or output overwrite.

Generate all primary baseline/refinement NetCDF products separately using reproduction/run.py from the frozen source. No new algorithm is introduced by this companion.

To render the paper figures after the full saved-data run:
python render_figures.py --data /path/to/inputs --bottom-products /path/to/results/bottom --top-original-products /path/to/results/top_original --top-guided-products /path/to/results/top_guided --refined-products /path/to/results/bottom --checks /path/to/checks --output /path/to/new/figures

The full saved-data run needs --refine-seam-edges for companion fields in bottom products. Original-style video rendering is supplied separately in frozen source reproduction/render_movies.py.

Artwork revision 04 (current paper): render_figures.py preserves the historical artwork and numbering. Replace workflow and component artwork with:
python render_artwork_revision04.py --source /path/to/repository --data /path/to/extracted/inputs --baseline-products /path/to/results/bottom --checks /path/to/checks --workflow /path/to/repository/assets/original_workflow.png --output /path/to/new/artwork

This preserves full-frame numerical calculation and crops only the displayed panels. All 20 components, plume/support/seam pixels remain visible. The source original workflow illustration is retained. The revision04 paper omits old Figure4 ablation; former Figures5-8 are Figures4-7. Historical plots are not automatically renamed or deleted. All parameters should be supplied explicitly.
