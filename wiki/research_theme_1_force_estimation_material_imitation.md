# Research Theme 1: Force Estimation and Material Latent for Imitation Learning

## Summary

The current direction is **force estimation**, not future force prediction.
The goal is to learn a force estimator supervised by measured force/torque during data collection, then use the estimated force or force-aware representation for imitation learning.

The strongest version of the idea is:

- train a force estimator from deployable observations such as vision, proprioception, action, and possibly a compact object/material latent `d_i`,
- avoid giving the downstream policy recent measured force history as an easy shortcut,
- use the mismatch between estimated and measured force to update only a compact object/material latent online,
- feed the estimated force and/or adapted latent into a policy such as Diffusion Policy.

This replaces the older "predict future force from current state/action" framing with a cleaner "estimate unavailable or hidden interaction force, then adapt a material latent from estimation error" framing.
Force prediction may still appear as a related auxiliary or prior-art comparison, but it is no longer the central research direction.
CAVIA is a useful meta-learning precedent for this logic because it freezes shared network weights and adapts only low-dimensional context parameters at test time.

## Motivation

The target benefit is higher success on unseen physical or material properties without relying on direct force sensing as a policy input at deployment.
Measured force/torque remains valuable as supervision and as an online adaptation signal, but the main representation should be a compact latent or estimated force signal that the policy can use.

This framing is more defensible against recent force-aware imitation and online world-model papers, because the contribution is not simply "use force" or "predict future force."
The intended contribution is force-estimation-error-driven adaptation of a compact hidden-physics/material latent.

## Task Selection Constraint

The current box-standing setup is not yet ideal as a core benchmark because, when the outer geometry is fixed, the task can still often be solved by replaying a successful trajectory.
The preferred next task should therefore make hidden physical properties matter to success even when appearance stays almost unchanged.
Unknown center of mass, mass, friction, or mechanism resistance are all valid hidden variables if they change the required action in a meaningful way.

## Current Premise

As of 2026-04-19, the central concern is that the dual-arm box-uprighting task may already be solvable by replaying a dual-arm teleoperation trajectory.
In that setup, unseen center of mass and total box mass do not currently appear to matter enough for success.
That makes the task weak as evidence for a hidden-physics latent, because success may come from replayed motion rather than identification of object dynamics.
Under the current discussion, one remaining candidate where unseen center of mass and mass are likely to matter is PushT with unseen physical properties.
Another promising direction is wiping with sponges of different stiffness or hardness, where compliance itself may be a causal hidden variable for successful control.
At the method level, the active design question is how to combine force estimation, compact material-latent adaptation, and downstream imitation learning without collapsing into ordinary force-aware policy learning.

## Open Questions

- Estimate current/interaction force only, or also estimate contact mode variables such as slip, sticking, or contact state.
- Whether measured-vs-estimated force error is sufficient to adapt `d_i` online when model weights are frozen.
- Whether the best implementation is explicit test-time latent optimization, encoder-based latent inference, or a hybrid of both.
- Whether uncertainty-conditioned action aggressiveness is tractable as an extension.
- Whether the project should stay centered on hidden mass / center of mass, or broaden toward hidden compliance through sponge hardness in wiping.

## Source

- [raw/research_theme_1_force_prediction_material_imitation.md](../raw/research_theme_1_force_prediction_material_imitation.md)
- [raw/2026-04-19_discussion_task_direction.md](../raw/2026-04-19_discussion_task_direction.md)
- [raw/2026-04-21_context_exi_net_modernization.md](../raw/2026-04-21_context_exi_net_modernization.md)
- [Force Estimation and Latent Adaptation Prior Art](force_estimation_latent_adaptation_prior_art.md)
- [CAVIA: Fast Context Adaptation via Meta-Learning](cavia_fast_context_adaptation_meta_learning.md)
- User clarification on 2026-04-17: the key problem is to avoid tasks that are still solvable by trajectory reproduction alone when geometry is unchanged.
- User clarification on 2026-04-19: dual-arm teleoperation replay may already solve box uprighting, so unseen center of mass and mass may be largely irrelevant there.
- User clarification on 2026-04-19: promising next directions currently include unseen-physics PushT and wiping with sponges of different hardness.
- User clarification on 2026-05-08: the project direction has shifted from force prediction to force estimation.
