# Wiki Log

## [2026-06-16] maintenance | Mark binary contact-state gating as legacy

- Updated the proposed-method, research-theme, regrasping, internal-property-memory, and index pages so explicit binary tactile/contact-state gating is no longer presented as part of the current method.
- Kept `wiki/binary_tactile_contact_signal.md` as a legacy page for traceability.
- Current framing should avoid saying the method performs or depends on contact/no-contact classification.

## [2026-06-13] ingest | Internal-property memory for regrasping and pouch insertion

- Read `thoughts/接触リッチ模倣学習 2.md` as the latest project-direction note.
- Added a wiki page reframing the project around human-like retention of internal object properties for regrasping, rather than method complexity added only for AdaWorldPolicy differentiation.
- Recorded the current concrete task candidate: thin pouch edge pinch, reorientation from a flat table pose, and insertion into a slot or flexible opening.
- Recorded the advisor critique that long-object unknown-center-of-mass lifting may be too rule-solvable to justify imitation learning as the main task.
- Updated the regrasping, proposed-method, and task-candidate pages to prioritize pouch pinch/insertion and demote long-object lifting to a secondary diagnostic benchmark.

## [2026-06-11] ingest | Binary tactile contact signal as core interaction evidence

- Read `thoughts/バイナリ型触覚の重要性。6月11日に気がついたこと。.md`.
- Added a high-importance wiki page recording the insight that contact occurrence should not be inferred from vision alone.
- Framed low-resolution or binary tactile contact as a core signal for hidden-physics identification, complementary to force/torque.
- Updated the main research theme, proposed-method page, and regrasping task page to include binary contact state as a gating/context signal for latent updates.
- Linked the new page from `wiki/index.md`.

## [2026-06-10] ingest | Regrasping as the core force-imagination task

- Read `thoughts/接触リッチ模倣学習.md` as the latest project-direction note.
- Added a reusable wiki page reframing the method around carrying force-derived hidden-physics information into regrasping or lifting phases where direct force information is absent or insufficient.
- Updated the proposed-method page so regrasping, long-object lifting, and dual-arm non-prehensile box lifting are the latest main task candidates.
- Updated the task-candidate page to demote sponge/brush surface interaction and foreground one-handed unknown-CoM lifting plus dual-arm non-prehensile box lifting.
- Linked the new page from `wiki/index.md`.

## [2026-05-21] query | Add brush sweeping and unknown-CoM prism lift task candidates

- Updated the strict task-search page with two revived candidates from the user.
- Added sweeping coffee beans with brushes of different stiffness as a feasible xArm7 lab task in the same surface-interaction family as sponge wiping.
- Added single-arm grasp-and-lift of a long rectangular prism with unknown center of mass as a simulation-friendly task that returns to the hidden-CoM story.
- Revised the previous "only sponge wiping survives" conclusion into a narrower but stronger task family: visually ambiguous contact-quality tasks, plus a simulation-only hidden-CoM benchmark.

## [2026-05-21] query | xArm7 lab constraint narrows force-imagination task candidates

- Updated the task-search page after the user rejected scraping/polishing, tape/lamination, and brushing as unrealistic for imitation learning with an xArm7 gripper in an ordinary lab setup.
- Recorded that sponge wiping is now effectively the only surviving practical task under the strict visual-ambiguity and hardware constraints.
- Clarified that broader surface-interaction framing can remain conceptual, but the first empirical paper should likely center on sponge wiping.

## [2026-05-21] query | Task search beyond sponge wiping for force imagination

- Added a self-critical task-search page evaluating alternatives to sponge wiping under a stricter criterion.
- Recorded that most candidate tasks are weak because the relevant hidden property becomes visible through motion, deformation, task progress, or failure.
- Narrowed the strongest domain to visually ambiguous contact-quality tasks, especially wiping/scraping/polishing with hidden compliance.
- Linked the new task-search page from `wiki/index.md`.

## [2026-05-21] ingest | Prior-art difference between visual world models and force imagination

- Added the user's original memo to `thoughts/2026-05-21_prior_art_difference_world_model_vs_force_imagination.md`.
- Added a wiki interpretation page clarifying the distinction from visual world-model-based imitation learning.
- Recorded the current experimental positioning: PushT is a controlled simulation benchmark, while wiping with different sponge softness is the stronger main task for showing the value of force imagination.
- Linked the new page from `wiki/index.md`.

## [2026-05-17] query | Li-san cross-attention landmark improvement

- Updated the Li-san facial motion prediction note with the simple-fusion vs cross-attention landmark-error comparison.
- Recorded that cross-attention reduces normalized mouth-center mean errors by about `53%` and variance by about `79-81%`.
- Clarified that SSIM improves from `0.818` to `0.928`, while the image-only baseline lacks usable landmark-error numbers.

## [2026-05-14] query | Proposed force-imagination online-correctable imitation method

- Added a reusable proposed-method page centered on force imagination and online-correctable imitation learning.
- Framed measured-vs-imagined force error as the signal for adapting only a compact hidden-physics latent.
- Linked the page from `wiki/index.md`.

## [2026-05-08] query | Add CAVIA context-adaptation prior

- Added a CAVIA paper note as a precedent for adapting only low-dimensional context parameters while shared network weights remain fixed.
- Clarified that MAML is the 2017 reference, while CAVIA is arXiv 2018 / ICML 2019.
- Linked CAVIA to the current force-estimation material-latent framing and prior-art page.

## [2026-05-08] maintenance | Shift wiki framing to force estimation

- Updated the main research theme from force prediction to force estimation.
- Recorded that the current direction is force-estimation-error-driven material-latent adaptation, not future force prediction.
- Updated related pages so force prediction is treated as older context or prior-art comparison rather than the central project framing.
- Renamed the main wiki page to `research_theme_1_force_estimation_material_imitation.md`.

## [2026-05-08] query | Preventing hidden-context latent collapse

- Added a reusable design note on why PB-like variables or context latents can be ignored by a predictor.
- Connected the failure mode to EXI-Net, IIDA, DyWA, and the force-estimation latent-adaptation framing.
- Recorded practical diagnostics and design adjustments for making hidden-physics latents affect inference.

## [2026-04-29] query | DyWA adaptation embedding size

- Updated the DyWA paper note with implementation-level adaptation embedding details from the paper.
- Recorded that DyWA reports `history length = 5`, `History Decoder = Conv1d + MaxPool`, `History Decoder channel = 128`, and `FiLM block Num = 3`.
- Added `raw/DyWA` as a git submodule from `https://github.com/jiangranlv/DyWA`.
- Confirmed from the released implementation that `z_t` / `cond` has shape `[B, 1, 128]` and is flattened to `[B, 128]` before FiLM conditioning.
- Clarified that this is a 128-dimensional history-conditioned adaptation representation rather than a small explicit per-object latent like EXI-Net's `d_i` or IIDA's `z_e`.

## [2026-04-29] query | EXI-Net d_e and d_i dimensions

- Added a reusable EXI-Net note recording the dimensions of explicit and implicit dynamics parameters.
- Recorded that standard EXI-Net uses `d_e = 4` for mass, friction, CoGx, and CoGy, and `d_i = 5` for implicit parametric biases.
- Noted the ablation detail that `EXI w/o d_e` uses `d_i = 9` to replace the missing explicit parameters.
- Linked the new note from `wiki/index.md`.

## [2026-04-29] query | IIDA z_e dimension and context points

- Added a reusable paper note for `Context is Everything: Implicit Identification for Dynamics Adaptation`.
- Recorded that IIDA's environment latent `z_e` is 8-dimensional across average-pooling, RNN, and Transformer context summarizers.
- Clarified that a context point is one transition tuple `(s, a, s')`, and that context size is the number of such transition examples.
- Linked the new note from `wiki/index.md`.

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

- Read `raw/Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers - student report.pdf` and created a reusable wiki summary page.
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

## [2026-04-24] ingest | DyWA paper note

- Read `raw/DyWA Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation.pdf` and added a reusable wiki summary page for DyWA.
- Recorded the method structure around world-action modeling, history-based dynamics adaptation, and FiLM conditioning.
- Added the DyWA paper note to `wiki/index.md`.

## [2026-04-24] query | DyWA vs explicit hidden-physics latent

- Added a comparison note clarifying that DyWA is strong under mixed randomized dynamics but does not cleanly isolate same-geometry hidden-physics identification.
- Recorded the key result numbers most relevant to the project, including the single-view unknown-state benchmark and the non-uniform-mass real-world cases.
- Documented the controlled-dynamics evaluation gap that could make this project's thesis sharper than DyWA.

## [2026-04-27] query | Force/wrench preprocessing in local papers

- Scanned the current local paper collection for explicit force or wrench preprocessing such as low-pass filters, Butterworth filters, moving-average smoothing, and FFT/Fourier transforms.
- Added a reusable note identifying `Adaptive Wiping` and `Adaptive Compliance Policy` as the clear positives.
- Linked the new note from `wiki/index.md`.

## [2026-04-28] query | AdaWorldPolicy force preprocessing

- Checked the AdaWorldPolicy arXiv HTML for explicit force preprocessing details.
- Added a reusable note recording that the paper does not appear to use explicit temporal force filtering, but does describe quantile-based normalization for force data.
- Linked the new note from `wiki/index.md`.

## [2026-04-28] query | AdaWorldPolicy force preprocessing detail update

- Re-checked the newly added local raw text for AdaWorldPolicy.
- Updated the force-preprocessing note with the newly confirmed detail that quantile-based normalization uses the `1st` and `99th` percentiles.
- Recorded that clipping is still not explicitly specified in the text.

## [2026-04-28] query | AdaWorldPolicy same-visual hidden-physics gap

- Added a comparison note recording the user assessment that AdaWorldPolicy is mainly evaluated on visual OOD and mixed deployment shifts.
- Clarified that it does not yet cleanly answer how far force-based feedback alone can adapt to same-looking objects with different hidden physical properties.
- Linked the new note from `wiki/index.md`.

## [2026-04-28] query | AdaWorldPolicy same-looking hidden-physics ambiguity update

- Updated the comparison note with the stronger user assessment that AdaWorldPolicy may face a more fundamental ambiguity when visually identical objects differ only in hidden physical properties.
- Recorded that the model does not appear to maintain an explicit long-term hidden-physics identity, so it may need to keep re-identifying the object from only very short-horizon force observations.
- Noted boxes with visually hidden internal state as a representative example where this can matter.
- Added the user’s original wording verbatim because it may be easier for the user to understand than a normalized summary.

## [2026-05-07] query | Force estimation and latent adaptation prior art

- Searched for prior work around force estimation from vision, force-aware imitation learning, and force-error-based online adaptation.
- Added a reusable note identifying ForceMapping as the closest threat for force-estimator-assisted imitation learning.
- Recorded that the remaining plausible gap is not force estimation itself, but measured-vs-estimated force error used to update only a compact object/material latent for imitation-policy conditioning.

## [2026-05-17] maintenance | Rename raw paper files

- Renamed unclear raw paper filenames such as arXiv IDs, hashes, and placeholders to title-based filenames.
- Updated existing raw/wiki references that pointed to the old filenames.
- Marked the duplicate UMI-FT raw Markdown copy explicitly as a duplicate copy in its filename.
