# Research Theme 1: Force Prediction and Material Latent for Imitation Learning

## Summary

Learn a force prediction model conditioned on image, current force, action, and a per-object latent material representation `d_i`.
Use boxes with identical geometry but varied internal mass and center of mass to induce physically different interaction dynamics.
After learning `f` and object-specific `d_i`, train a policy such as Diffusion Policy with `d_i` included in the observation.
At deployment, freeze `f` and adapt only `d_i` online from execution-time data.

## Motivation

The target benefit is higher success on unseen physical properties, not just reconstruction of a demonstrated trajectory.
The long-term hope is that better identification of object physics enables faster and more open-loop-like dynamic manipulation.

## Task Selection Constraint

The current box-standing setup is not yet ideal as a core benchmark because, when the outer geometry is fixed, the task can still often be solved by replaying a successful trajectory.
The preferred next task should therefore make hidden physical properties matter to success even when appearance stays almost unchanged.
Unknown center of mass, mass, friction, or mechanism resistance are all valid hidden variables if they change the required action in a meaningful way.

## Open Questions

- Predict only future force, or also slip/contact mode variables.
- Whether repeated test-time latent optimization can recover from larger failures than small local corrections.
- Whether uncertainty-conditioned action aggressiveness is tractable as an extension.

## Source

- [raw/research_theme_1_force_prediction_material_imitation.md](../raw/research_theme_1_force_prediction_material_imitation.md)
- User clarification on 2026-04-17: the key problem is to avoid tasks that are still solvable by trajectory reproduction alone when geometry is unchanged.
