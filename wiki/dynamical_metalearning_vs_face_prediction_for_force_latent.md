# Dynamical-Metalearning vs Face Prediction for Force/Material Latent

## Summary

The `dynamical-metalearning` repository is not a close implementation match for Li-san's face keypoint prediction project.
Li-san's paper uses a task-specific facial motion predictor that combines RGB images and MediaPipe landmarks, then feeds predicted future landmarks into an ACT-style controller.
By contrast, `dynamical-metalearning` is a system-identification repository for robot dynamics that predicts future trajectories from context windows of past observations and controls.

For your project, `dynamical-metalearning` is architecturally closer than the face-prediction paper to the hidden-physics problem itself.
However, it still does not match your preferred formulation exactly, because it identifies dynamics implicitly from history rather than learning an explicit per-object latent such as `d_i`.

## What Li-san's Face Prediction System Does

The face-prediction paper uses:

- RGB facial video
- 468 MediaPipe landmarks per frame
- a cross-attention Transformer for short-horizon future facial landmark prediction
- an ACT-style policy that receives current and predicted future mouth landmarks

So its core pattern is:

1. predict a task-relevant future variable
2. inject that prediction into the downstream imitation policy

This was useful under the older force-prediction framing as a structural precedent for predicting future task-relevant variables before control.
Under the current force-estimation framing, it is mainly a weaker analogy rather than the main implementation template.
But it is not a system-identification method for hidden physical properties.

## What Dynamical-Metalearning Does

The repository is organized around synthetic data generation plus `sys_identification`.
Its main models take a context window of past control and state trajectories and predict future trajectories.

The Transformer path specifically embeds context `(y, u)` pairs, encodes them, and decodes future outputs conditioned on future inputs.
The repository also randomizes masses, centers of mass, and inertias during data generation, so hidden physical variation is central to the benchmark.

Its core pattern is:

1. randomize hidden dynamics across environments
2. infer those dynamics implicitly from trajectory context
3. predict future behavior under new control inputs

This is much closer to your hidden-physics direction than facial landmark prediction.

## Key Difference Relative to Your Current Research Direction

The older preferred framing was:

- learn a force or state predictor
- condition it on image, force, action, and an explicit object/material latent
- adapt only that latent at test time

As of 2026-05-08, the current framing is sharper:

- learn a force estimator supervised by measured force/torque,
- avoid giving the policy recent measured force history as a shortcut,
- update only a compact object/material latent from measured-vs-estimated force error,
- use estimated force and/or the adapted latent for imitation-policy conditioning.

`dynamical-metalearning` is close on the first half but different on the second half.
It uses context-conditioned sequence prediction, but it does not make the environment code the explicit main object of the method.
In other words, the hidden physics is represented implicitly in the history embedding rather than explicitly as a named latent that can be analyzed, stored per object, or updated online as a compact belief.

That makes it a better architectural reference than the face-prediction paper, but not yet the cleanest reference for your thesis claim.

## Which Is Better for Your Project

If your immediate goal is to build a working predictor quickly, `dynamical-metalearning` is the better reference.
It already matches the important ingredients:

- context-conditioned dynamics prediction
- randomized hidden physical parameters
- Transformer-based history reading
- evaluation on generalization across changed dynamics

If your goal is to make the research claim crisp, neither should be copied directly.
The stronger direction is a hybrid:

1. Take the history-encoder intuition from `dynamical-metalearning` only where it helps identify hidden dynamics.
2. Compress the inferred hidden dynamics into an explicit latent `d_i` or `z_env`.
3. Use that latent inside a force estimator or policy.
4. Update only the latent online from force-estimation error at deployment.

That hybrid matches your current wiki direction better than either source alone.

## Practical Recommendation

For your research, the most defensible path is:

1. Do not treat Li-san's face predictor as the main implementation template.
2. Use `dynamical-metalearning` mainly as a reference for context-based hidden-dynamics inference.
3. Keep your own method centered on an explicit latent for object/material properties.
4. If you want a simple first version, implement a small encoder plus latent head plus force-estimation head, then test whether measured-vs-estimated force error can adapt only the latent.

That would preserve the practical strengths of `dynamical-metalearning` while keeping your method aligned with the explicit latent-identification story you want to tell.

## Sources

- [raw/dynamical-metalearning/README.md](../raw/dynamical-metalearning/README.md)
- [raw/dynamical-metalearning/sys_identification/architectures/transformer/transformer_sim.py](../raw/dynamical-metalearning/sys_identification/architectures/transformer/transformer_sim.py)
- [raw/dynamical-metalearning/sys_identification/train.py](../raw/dynamical-metalearning/sys_identification/train.py)
- [raw/dynamical-metalearning/data_generation/randomenvs.py](../raw/dynamical-metalearning/data_generation/randomenvs.py)
- [Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers](improving_robotic_imitation_learning_with_predicted_facial_motion_using_transformers.md)
- [Research Theme 1: Force Estimation and Material Latent for Imitation Learning](research_theme_1_force_estimation_material_imitation.md)
