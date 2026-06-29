# 2026-04-19 Discussion: task direction and current premise

## Context

This note records the user's current discussion-level premise and task-direction ideas.
These points are not yet final conclusions.

## Current Premise

- A dual-arm teleoperation trajectory may already be enough to solve the dual-arm box-uprighting task by replay.
- If that is true, then unseen center of mass and total box mass are not very important for success in the current uprighting setup.
- That makes the current box-uprighting task weak as evidence for a hidden-physics or material-latent method.

## Task Direction Discussed

At this point, the user has identified three task candidates:

1. Unseen-center-of-mass / unseen-mass PushT.
2. A wiping task that uses sponges with different stiffness or hardness.
3. A task that uses rollers ("korokoro") with different radii to pick up small paper pieces or similar debris.

Why these currently look useful:

- PushT keeps the hidden variable centered on mass and center of mass.
- Sponge wiping is attractive because sponge compliance may directly matter for contact behavior and task success, unlike the current box-uprighting setup where replay may already be sufficient.
- The roller / korokoro idea is also promising because, even if the tool geometry is visually observable, force feedback still seems important for successful interaction.
- More broadly, tasks that insert a tool between the robot and the environment may be good fits because contact state is harder to infer directly and the user's PB-style physical-property representation may become useful.

## Status

- These points are still discussion premises and not yet fixed research decisions.
