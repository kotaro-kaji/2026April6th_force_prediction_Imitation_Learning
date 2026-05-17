# Proposed Method: Force-Imagination-Based Online-Correctable Imitation Learning

## Summary

The proposed method is **online-correctable imitation learning with force imagination**.
The core idea is to train a policy that does not merely replay demonstrated motion, but acts while internally estimating the contact force that should be present under the current object or material.

During demonstration collection, measured force/torque is available as supervision.
During deployment, the policy mainly uses deployable observations such as vision, proprioception, action history, and a compact object/material latent.
The model then produces an imagined or estimated force signal.
If force/torque measurement is available online, the mismatch between measured and imagined force is used to update only the compact latent, while the estimator and policy weights remain fixed.

## Method Structure

The method has three modules:

1. **Force imagination model**
   - Input: deployable observation, proprioception, action, and object/material latent `d_i`.
   - Output: estimated current interaction force or wrench.
   - Training signal: measured force/torque from demonstrations.

2. **Imitation policy**
   - Input: ordinary policy observation plus imagined force and/or adapted latent `d_i`.
   - Output: action, action sequence, or diffusion-policy trajectory.
   - Role: imitate expert behavior while conditioning on hidden physical state.

3. **Online latent correction**
   - Input: error between measured force and imagined force.
   - Updated variable: only `d_i`.
   - Frozen variables: force-imagination model weights and policy weights.

The deployment-time adaptation rule is:

`d_i <- d_i - alpha * grad_{d_i} L(F_measured, F_imagined(o, a, d_i))`

This follows the CAVIA-style principle of adapting only a low-dimensional context variable instead of updating the full model online.

## Why Force Imagination

Force imagination is useful because contact-rich manipulation often depends on hidden physical properties that are not directly visible.
Examples include mass, center of mass, friction, stiffness, hardness, and mechanism resistance.

If two objects look the same but require different actions, a pure visual imitation policy can collapse into trajectory replay.
Force imagination gives the policy a physically grounded internal signal:

- what force should be felt now,
- whether the current interaction matches the expected object/material,
- and how the policy should change when the hidden physics is different.

The important point is that the imagined force is not only an auxiliary prediction.
It is the error signal used to correct the latent that conditions later policy behavior.

## Difference From Ordinary Force-Aware Imitation

The contribution is not simply that the policy uses force.
Recent work already uses force/torque data, force prediction, or force-aware representations.

The sharper claim is:

- measured force is used as supervision and possibly as online correction,
- the policy is not allowed to solve the task by directly consuming recent measured force history as a shortcut,
- the model maintains a compact hidden-physics latent,
- online adaptation changes only that latent,
- the adapted latent and imagined force condition the imitation policy.

This makes the method closer to hidden-physics identification than ordinary multimodal imitation learning.

## Training Objective

A minimal training objective is:

`L = L_policy + lambda_F L_force + lambda_reg L_latent`

where:

- `L_policy` trains the imitation policy from demonstrations,
- `L_force` supervises imagined force with measured force/torque,
- `L_latent` keeps the object/material latent compact and stable.

For tasks where latent collapse is likely, multi-step force or dynamics consistency can be added so that the latent must explain persistent hidden physical properties rather than only one-frame noise.

## Evaluation Focus

The key evaluation should use same-appearance or nearly same-appearance objects whose hidden physical properties differ.
The task should fail or degrade under simple trajectory replay.

Strong candidates include:

- PushT-like manipulation with hidden mass, friction, or center of mass changes,
- wiping with visually similar sponges of different hardness or stiffness,
- object interaction where mechanism resistance changes while geometry remains similar.

The most important ablations are:

- no imagined force,
- imagined force without online latent correction,
- online correction of full network weights instead of latent only,
- direct measured-force-history policy input,
- randomized or fixed latent at deployment.

## Relation to Existing Notes

- [Research Theme 1](research_theme_1_force_estimation_material_imitation.md)
- [Force Estimation and Latent Adaptation Prior Art](force_estimation_latent_adaptation_prior_art.md)
- [Preventing Hidden-Context Latents From Being Ignored](preventing_hidden_context_latent_ignored.md)
- [CAVIA: Fast Context Adaptation via Meta-Learning](cavia_fast_context_adaptation_meta_learning.md)
