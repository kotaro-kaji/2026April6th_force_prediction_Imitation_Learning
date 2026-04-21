# Wiki Log

## [2026-04-17] ingest | Research Theme 1 Force Prediction Material Imitation

- Added the original research memo to `raw/`.
- Added a reusable summary page to `wiki/`.
- Initialized `wiki/index.md` and `wiki/log.md`.

## [2026-04-17] query | Hidden-Physics Task Candidates

- Added a task-selection note to the research theme page.
- Added a ranked comparison page for alternative tasks beyond box standing.
- Linked the new task-design page from `wiki/index.md`.

## [2026-04-19] query | Clarified weakness of dual-arm box uprighting

- Updated the research theme page with the current premise that dual-arm teleoperation replay may already solve box uprighting.
- Recorded that unseen center of mass and mass may be largely irrelevant in the current uprighting task.
- Narrowed the task-candidate page so unseen-physics PushT is the main remaining candidate.

## [2026-04-19] ingest | Discussion note on task direction

- Added a raw discussion note capturing the current premise about dual-arm box uprighting and replay.
- Updated the research theme page to reflect that wiping with different sponge hardness is now a promising direction alongside unseen-physics PushT.
- Revised the task-candidate page so wiping is no longer treated as a poor fit by default.

## [2026-04-19] ingest | Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers

- Read `raw/202430182_en.pdf` and created a reusable wiki summary page.
- Added the paper note to `wiki/index.md`.
- Recorded the paper as a precedent for feeding predicted future task-relevant state into an ACT-style imitation learning policy.

## [2026-04-21] query | Overlap with ACP UMI-FT CompACT ForceMimic

- Added a comparison note distinguishing compliant or force-aware imitation from the object-specific latent direction.
- Linked the comparison page from `wiki/index.md`.
- Recorded that the latent-identification framing remains the clearest way to separate the project from recent related work.

## [2026-04-21] query | Dynamical-Metalearning vs Li-san face prediction

- Added a comparison note between `raw/dynamical-metalearning` and the facial motion prediction paper.
- Recorded that the system-identification repository is architecturally closer to hidden-physics prediction than the facial predictor.
- Clarified that the best fit for the project is still a hybrid: history encoder plus explicit object/material latent.

## [2026-04-21] ingest | EXI-Net modernization context

- Added a raw context note clarifying that the predictor and latent design originally came from EXI-Net.
- Updated the research theme page so the current method discussion is framed as a modernization of EXI-Net rather than a separate line of work.
- Recorded that the live design question is how to modernize latent inference while preserving deployment-time adaptation of the hidden-dynamics representation.
