# DyWA vs Explicit Hidden-Physics Latent: The Controlled-Dynamics Gap

## Question

What is the important difference between DyWA and the current hidden-physics latent direction for this project?

## Short Answer

DyWA is strong evidence that history-based dynamics adaptation helps under randomized mass, friction, scale, and partial observability.
But it does **not** appear to provide the cleanest controlled test of whether a method can identify and adapt to hidden physical properties when geometry is held effectively fixed.

That gap matters if the thesis claim is specifically about hidden-physics identification rather than broad robustness under mixed variation.

## What DyWA Explicitly Evaluates

The paper explicitly reports:

- simulation training with randomized mass, scale, friction, and restitution,
- simulation testing on `10` unseen geometries with `5` scales each,
- real-world evaluation on unseen objects including slippery objects and objects with non-uniform mass distribution,
- a real-world ablation on different surface-friction conditions.

In the most realistic simulation track, unknown state plus single-view observation, DyWA reports:

- seen `82.2%`
- unseen `75.0%`

while CORN(PN++) reports:

- seen `50.7%`
- unseen `49.4%`

In real-world non-uniform-mass examples, the paper reports:

- half-filled bottle: `4/5`
- coffee jar: `3/5`

These are strong results.

## What Is Not Cleanly Isolated

The paper does **not explicitly present** a controlled experiment of the form:

- same mesh,
- same visual point cloud or nearly identical appearance,
- different mass,
- different center-of-mass offset,
- different friction.

This is an inference from the benchmark description, task setup, and result tables in the paper.
The reported experiments mix geometry variation with dynamics variation much of the time, and the real-world evaluation also changes object identity across trials.

So the paper supports the claim:

"the method is robust under randomized dynamics and realistic observation constraints."

It does **not** cleanly isolate the stronger claim:

"the method can identify hidden physical differences when geometry is controlled and appearance gives little help."

## Why This Matters for Your Method

Your project becomes much easier to distinguish if you evaluate exactly the case that DyWA leaves partially open:

- same geometry,
- nearly unchanged appearance,
- action requirements change because of hidden dynamics only.

Examples include:

- same object shape with different mass,
- same object shape with center-of-mass shift,
- same object shape with friction change,
- same tool geometry with different compliance.

If your method succeeds there, the contribution is more specifically about hidden-physics inference rather than general robustness.

## Important Difference in Representation

DyWA uses a history-conditioned adaptation embedding and FiLM modulation inside the world-action model.
That is adaptive, but the hidden physics is represented implicitly.

Your preferred direction is different if it stays centered on:

- an explicit latent for hidden object/material properties,
- using prediction as a supervisory tool to shape that latent,
- feeding that latent into a downstream imitation policy such as Diffusion Policy,
- and potentially adapting only that latent at test time.

So the representation claim is sharper in your project if the latent is explicit and reusable outside the predictor itself.

## What Would Be a Stronger Evaluation Than DyWA

If the goal is to make a stronger hidden-physics claim, the key simulation benchmark would be something like:

- same mesh / same visual point cloud,
- different mass,
- different center-of-mass offset,
- different friction,
- all other variables controlled as much as possible.

Then the important analyses would be:

- success rate as a function of history length,
- performance when latent adaptation is disabled,
- whether the inferred embedding or latent clusters by physical property,
- whether nearby embeddings correspond to nearby dynamics,
- whether a fixed policy plus adapted latent is enough to recover performance.

This would test identification more directly than DyWA's current benchmark.

## Bottom Line

DyWA is a strong paper for:

- single-view non-prehensile manipulation,
- history-based dynamics adaptation,
- FiLM-conditioned policy modulation,
- and robustness to mixed geometry-and-dynamics variation.

Its weaker point, relative to your thesis direction, is that it does not cleanly isolate the same-geometry hidden-dynamics setting.
That gap is exactly where your project can make a sharper contribution.

## Sources

- [raw/2503.16806v2.pdf](../raw/2503.16806v2.pdf)
- [Research Theme 1: Force Estimation and Material Latent for Imitation Learning](research_theme_1_force_estimation_material_imitation.md)
- [2026-04-23 Message: DP latent methods 2 and 3 reframing](../raw/2026-04-23_message_dp_latent_methods_2_and_3_reframing.md)
