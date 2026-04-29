# DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation

## Summary

DyWA studies non-prehensile 6D object rearrangement under a realistic setting: single-view point cloud observation, no external pose tracker for the student policy, and randomized object dynamics.
Its key idea is to combine:

- a world-action model that jointly predicts actions and next task state,
- a dynamics adaptation module that reads a history of observation-action pairs,
- and FiLM conditioning that injects the inferred dynamics embedding into the policy network.

The paper shows strong gains over CORN and HACMan in simulation, especially in the hardest setting with unknown state and single-view observation.
It also reports real-world robustness to slippery objects, surface-friction changes, and some non-uniform mass cases such as a half-filled bottle.

## Method Structure

The student policy is trained in a teacher-student distillation framework.
The teacher has privileged access to full point cloud, task state, and physical parameters.
The student only receives observations realistic for deployment.

The core components are:

1. World-action model:
   jointly predicts the action and the next task state.
2. Dynamics adaptation module:
   encodes a history of observation-action pairs into an adaptation embedding.
3. FiLM conditioning:
   decodes the adaptation embedding into modulation parameters that condition early layers of the world-action model.

The history encoder is not an explicit per-object latent in the sense of a stored object code `d_i`.
Instead, dynamics are inferred online from recent trajectories and injected into the policy as an adaptation embedding.

## What Is Randomized and Evaluated

In simulation, the paper randomizes:

- object mass,
- object scale,
- object friction,
- restitution of object, table, and gripper.

The benchmark includes:

- 323 training objects,
- 10 unseen geometries for test,
- 5 scales per unseen geometry,
- three evaluation tracks:
  known state with 3-view observation,
  unknown state with 3-view observation,
  unknown state with 1-view observation.

In the real world, the paper evaluates 10 unseen objects, including slippery objects and objects with non-uniform mass distribution.
It also includes a surface-friction robustness experiment using four tablecloth friction levels.

## Main Results

### Simulation

In the hardest setting, unknown state with single-view observation:

- DyWA: seen `82.2%`, unseen `75.0%`
- CORN (PN++): seen `50.7%`, unseen `49.4%`
- CORN: seen `29.0%`, unseen `29.8%`

So the gap over the stronger CORN(PN++) baseline is large in the setting that most resembles realistic deployment.

### Ablation

In the same hardest setting:

- DAgger baseline: seen `59.9%`, unseen `57.5%`
- RMA-style dynamics adaptation only: seen `65.6%`, unseen `57.9%`
- world model + dynamics adaptation without FiLM: seen `73.3%`, unseen `59.4%`
- full DyWA: seen `82.2%`, unseen `75.0%`

This supports the paper's claim that next-state prediction, dynamics adaptation, and FiLM conditioning work best together.

### Real World

Across 10 unseen objects:

- DyWA: `34/50 = 68%`
- CORN with tracking: `18/50 = 36%`

For challenging non-uniform mass cases:

- half-filled bottle: `4/5`
- coffee jar: `3/5`

For surface-friction variation:

- DyWA with dynamics adaptation maintains `4/5` success across all four friction settings in the reported experiment,
- while the version without dynamics adaptation varies between `3/5` and `4/5` and shows much less stable execution time.

## Relevance to This Project

DyWA is important because it is a recent example of:

- history-based hidden-dynamics adaptation,
- explicit use of FiLM for dynamics conditioning,
- and strong evaluation under randomized physical properties.

However, its representation of hidden physics is still mainly **implicit** in a history-conditioned embedding.
That is different from your current thesis direction, where the central object is an explicit latent used downstream by imitation learning or diffusion policy.

So DyWA is a strong architectural and experimental reference for:

- history encoder design,
- dynamics-conditioned policy modulation,
- and benchmark style for randomized physics,

but not yet a direct match to an explicit object/material latent story.

## Sources

- [raw/2503.16806v2.pdf](../raw/2503.16806v2.pdf)
