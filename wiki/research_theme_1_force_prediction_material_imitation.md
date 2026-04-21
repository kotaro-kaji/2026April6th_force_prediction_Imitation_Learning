# Research Theme 1: Force Prediction and Material Latent for Imitation Learning

## Summary

Learn a force prediction model conditioned on image, current force, action, and a per-object latent material representation `d_i`.
Use boxes with identical geometry but varied internal mass and center of mass to induce physically different interaction dynamics.
After learning `f` and object-specific `d_i`, train a policy such as Diffusion Policy with `d_i` included in the observation.
At deployment, freeze `f` and adapt only `d_i` online from execution-time data.
This direction is best understood as a modernization of the EXI-Net-style idea: keep predictor conditioning on hidden dynamics, but replace older latent-inference machinery with a more modern learned history encoder where useful.

## Motivation

The target benefit is higher success on unseen physical properties, not just reconstruction of a demonstrated trajectory.
The long-term hope is that better identification of object physics enables faster and more open-loop-like dynamic manipulation.

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
At the method level, the active design question is not whether to keep EXI-Net unchanged, but how to preserve its deployment-time latent adaptation logic while updating the predictor and latent inference design to a more modern form.

## Open Questions

- Predict only future force, or also slip/contact mode variables.
- Whether repeated test-time latent optimization can recover from larger failures than small local corrections.
- Whether the best modernization of EXI-Net is explicit test-time latent optimization, encoder-based latent inference, or a hybrid of both.
- Whether uncertainty-conditioned action aggressiveness is tractable as an extension.
- Whether the project should stay centered on hidden mass / center of mass, or broaden toward hidden compliance through sponge hardness in wiping.

## Source

- [raw/research_theme_1_force_prediction_material_imitation.md](../raw/research_theme_1_force_prediction_material_imitation.md)
- [raw/2026-04-19_discussion_task_direction.md](../raw/2026-04-19_discussion_task_direction.md)
- [raw/2026-04-21_context_exi_net_modernization.md](../raw/2026-04-21_context_exi_net_modernization.md)
- User clarification on 2026-04-17: the key problem is to avoid tasks that are still solvable by trajectory reproduction alone when geometry is unchanged.
- User clarification on 2026-04-19: dual-arm teleoperation replay may already solve box uprighting, so unseen center of mass and mass may be largely irrelevant there.
- User clarification on 2026-04-19: promising next directions currently include unseen-physics PushT and wiping with sponges of different hardness.
