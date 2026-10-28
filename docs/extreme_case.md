# Proposed manuscript wording: terrain–sky continuity limitation

## Methods / diagnostic experiment

We evaluated an optional post-segmentation continuity prior for a difficult top-camera sequence in which the author identified a continuous plume but SAM3 omitted support near the terrain–sky boundary. Raw predictions were retained. Candidate additions were restricted to a narrow boundary band and required substantial smoke detections on both the terrain and sky sides, bounded vertical gap and corridor extent, an added-area limit, and consistency with adjacent-frame detections or independently generated candidates. Adjacent-frame agreement is a prior-consistency check and does not provide independent segmentation truth. All inferred pixels were identified by a separate mask and quality bit (131072, continuity_inferred). They were assigned missing opacity, invalid retrieval regime and zero retrieval confidence. Existing opacity values and retrieval/calibration support were preserved exactly.

## Results / scope

For the 22-frame top-camera window (frame identifiers 00355–00376), dominant terrain and sky supports were initially connected in 10 frames. Gap limits of 40, 60 and 80 original-image pixels produced connected supports in 15, 22 and 20 frames, respectively. The 60-pixel experiment added 266,706 pixel-frames, equivalent to 1.40% of the original 19,028,582 detected plume pixel-frames; the largest per-frame addition was 2.47%. The 80-pixel experiment triggered the 5% added-area safeguard in two frames, for which the proposal was withheld. These figures measure geometric consistency and sensitivity, not segmentation accuracy. The thresholds are exploratory and case specific.

## Limitations and interpretation

Visual inspection reveals that narrow inferred bridges can establish connectivity while leaving much of the ambiguous strip unresolved. The prototype therefore remains optional and is not adopted as a validated replacement for raw SAM3 masks. Nearly equal smoke and sky RGB can prevent image-only segmentation from uniquely identifying the connecting region; connectivity assumptions may also fail for detached smoke, occlusion or overlapping clouds. Geometric repair does not resolve radiometric non-identifiability or justify interpolation of opacity. The case should be reported as an extreme low-contrast failure mode, with raw masks, inferred pixels and missing-opacity regions displayed separately.

## Reporting notes

These are proposed passages, not edits to the Desktop manuscript. Resolve the 28/29 April dataset-date discrepancy before assigning a physical date. No independent pixel labels, optical-depth reference or calibrated probability is available. Lower-positive prompt ablation is a sensitivity diagnostic and is not evidence that the upper region is background cloud. The bottom-camera pipeline and original retrieval source remain unchanged.

## Revised framing after author review (supersedes any claim of a solved continuity problem)

This sequence is retained as an extreme-case failure study. Frame367 illustrates that a narrow geometric bridge can satisfy a connectivity metric while leaving much of the apparent plume corridor unresolved. A broader, locally bounded spatial and temporal continuity prior is proposed for future evaluation, with raw and inferred support retained separately. The present patch is not adopted as a validated correction.

Frame376 illustrates a separate limitation: SAM3 already provides connected dominant plume supports, but radiometric QC removes valid-opacity support along the connecting strip. Pixels that brighten the reconstructed sky or have near-neutral contrast are withheld under the current attenuation-based screening criteria. Thus mask continuity does not guarantee a retrievable opacity field. Across this 22-frame diagnostic,10 raw masks have connected dominant terrain/sky supports, whereas none of the QC-valid opacity fields do under the same connectivity diagnostic. These figures characterize software-stage behaviour, not independent accuracy. The background estimate, regime assumptions and threshold conservatism may all contribute and remain unvalidated. Do not describe the additional opacity split as a SAM3 mask failure or imply that geometric infilling recovers optical information.
