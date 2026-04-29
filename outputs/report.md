## [2026-04-17 14:00] maintenance | Refine raw source retention rule

- changed files: `AGENTS.md`, `outputs/report.md`
- result: Added a stricter rule for `raw/` so web content is stored only when it is clearly useful and costly to re-access later, reducing noise in the raw source layer.

## [2026-04-17 14:10] ingest | Research Theme 1 memo

- changed files: `raw/research_theme_1_force_prediction_material_imitation.md`, `wiki/research_theme_1_force_prediction_material_imitation.md`, `wiki/index.md`, `wiki/log.md`, `outputs/report.md`
- result: Stored the user's research-theme memo as a raw source and added a short reusable wiki summary with index and log entries.

## [2026-04-17 14:25] query | Hidden-physics task candidates

- changed files: `wiki/research_theme_1_force_prediction_material_imitation.md`, `wiki/task_candidates_for_hidden_com_material_latent.md`, `wiki/index.md`, `wiki/log.md`, `outputs/report.md`
- result: Added a ranked comparison of task candidates centered on hidden CoM / hidden physics, clarifying why constrained unknown-CoM pushing and edge-pivoting are stronger fits than box standing, wiping, or insertion.

## [2026-04-17 14:35] maintenance | Correction on reporting and task interpretation

- changed files: `outputs/report.md`
- result: Recorded that conversational progress updates exceeded the AGENTS.md reporting preference and that the earlier task ranking overvalued unknown-CoM edge pivoting despite the user's observation that trajectory replay already succeeds there.

## [2026-04-17 14:45] query | Community view on missing force or wrench sensing

- changed files: `outputs/report.md`
- result: Investigated current IL / VLA literature on failure modes caused by lacking force or tactile sensing, focusing on contact-rich manipulation and benchmark/task implications.

## [2026-04-27 14:35] query | Force/wrench preprocessing in local papers

- changed files: `outputs/report_force_wrench_preprocessing.md`, `outputs/report.md`
- result: Saved a task-specific report listing local papers that explicitly preprocess force/wrench signals with Butterworth low-pass filtering, moving-average smoothing, or FFT-based encoding, and separated them from papers that only use raw wrench or non-force filtering.

## [2026-04-28 10:26] query | Opinion on whether past wrench should be an input

- changed files: `outputs/report_past_wrench_input_opinion.md`, `outputs/report.md`
- result: Wrote a task-specific Japanese memo arguing that past wrench should usually be excluded from the main representation-learning input when the goal is hidden-physics latent acquisition, while noting limited helper uses and recommending a history-based estimator/identifier design instead.
