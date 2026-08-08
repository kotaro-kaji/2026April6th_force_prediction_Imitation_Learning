# Decision Fork: Single-Arm Pouch vs Dual-Arm Box Lifting

## Current Question

As of 2026-07-30, the project is choosing between two tasks that require substantially different predictive representations:

1. **Single-arm pouch lifting**
   - The useful prediction target is primarily visual: future pouch shape, pose, folding, buckling, or task-relevant keypoints.
   - This direction is closer to visual predictive imitation learning and deformable-object manipulation.

2. **Dual-arm box lifting with minimal tilt**
   - The useful signal is primarily force/torque: load distribution, gravitational moment, and hidden center of mass.
   - This direction is closer to inertial-parameter estimation, wrench prediction, and force-aware cooperative control.

These should not be presented as two interchangeable benchmarks for one method.
They represent a choice between two different research theses.

## Current Recommendation

Under the present two-option choice, **advance the single-arm pouch task as the primary experimental direction**.

This recommendation implies an explicit pivot:

> The main contribution would become task-relevant visual future-state prediction for imitation learning, rather than force-imagination-based hidden-physics adaptation.

If retaining force imagination is non-negotiable, neither task is currently satisfactory:

- the pouch task does not clearly require force prediction;
- the box task is now strongly covered by an analytic online-estimation and wrench-optimization solution.

In that case, the correct action is to search for a third force-centered task rather than forcing the pouch and box into one method.

## Why the Dual-Arm Box Is Weak as the Main Learned Task

The box task has clean physical variables and metrics, but this is also its main research risk.
Mass and center of mass can be estimated from measured contact wrenches using rigid-body equilibrium, after which contact forces can be computed through optimization.

A particularly close 2026 study already:

- handles a box with unknown mass and center of mass using two arms;
- estimates mass and center of mass online from measured contact wrenches;
- computes friction-feasible contact forces and torques;
- evaluates orientation deviation, slip, drop, and excessive squeezing;
- demonstrates the method on a real dual-arm system with different center-of-mass configurations.

This creates a strong simple-baseline problem for a learned force predictor.
The paper question would become:

> Why learn force or a hidden latent when measured wrench plus quasi-static estimation and convex optimization already solves the task?

Changing the implementation from analytic estimation to imitation learning is not, by itself, a strong answer.

The box direction remains viable only if the task moves beyond the assumptions of the analytic solution, for example:

- dynamic rather than quasi-static lifting;
- deformable or shifting contents;
- uncertain contact geometry or changing contact modes;
- a later action decision that must use remembered interaction evidence;
- large model mismatch for which a learned residual is demonstrably necessary.

These changes would substantially increase the scope and should be treated as a redesigned task, not the current box-lifting task.

## Why the Pouch Is the Better Primary Direction

The pouch task has a stronger reason to use imitation learning:

- deformation is high-dimensional and difficult to model exactly;
- pinching and lifting can cause folding, buckling, dragging, or occlusion;
- successful action can depend on predicting the object's task-relevant visual state before a failure becomes unrecoverable;
- demonstrations can encode manipulation strategies that are difficult to express as a small analytic controller.

The prediction target should remain compact and task-relevant.
Predicting full future RGB frames is not automatically necessary.
Candidate targets include:

- pouch boundary or centerline keypoints;
- free-end position;
- pouch-plane orientation;
- fold or buckle state;
- future grasp-relative geometry;
- probability of reaching a lift-ready pose.

The contribution can then be tested by feeding this predicted future visual state into an imitation policy and measuring whether it improves action selection.

## Main Risk of the Pouch Direction

The pouch task is not automatically a strong prediction task.
A standard image-history policy or reactive visual servo may solve it without an explicit future predictor.

The task must contain a decision whose quality changes when future deformation is anticipated.
Examples include:

- choosing lift direction or wrist rotation after the initial pinch;
- choosing between continuing the lift and performing a corrective motion;
- selecting a regrasp point;
- avoiding a fold or table-drag state that becomes difficult to recover from.

If every pouch can be lifted by one slow conservative motion, future-state prediction has no causal role.

## Minimum Go/No-Go Experiment

Before building the full method:

1. Fix one pouch geometry and create a small controlled variation set.
2. Train or execute a plain current-observation imitation baseline.
3. Measure fold, drag, drop, final orientation, and lift success.
4. Test whether a short-horizon visual predictor can forecast failures or task-relevant keypoints before they are visible in the current observation.
5. Continue only if the predicted state changes the selected action and improves success on held-out variations.

This is the cheapest way to test whether the pouch is genuinely a predictive-imitation problem rather than only a visually reactive manipulation task.

## Role of the Box Task

The box should be retained as:

- a diagnostic force/torque benchmark;
- a baseline implementation for hidden-center-of-mass estimation;
- or a possible later task after introducing dynamic contents or other departures from quasi-static rigid-body assumptions.

It should not currently carry the main novelty claim.

## Sources

- User clarification on 2026-07-30: pouch lifting primarily requires visual prediction, whereas tilt-minimizing dual-arm box lifting primarily requires force-sensor prediction.
- [Industrial Dual-Arm Box Handling via Online Inertial Estimation and Convex Wrench Optimization](https://arxiv.org/abs/2605.22021)
- [Bimanual Deformable Bag Manipulation Using a Structure-of-Interest Based Neural Dynamics Model](https://arxiv.org/abs/2401.11432)
- [Internal-Property Memory for Regrasping](internal_property_memory_for_regrasping.md)
- [Regrasping as the Core Force-Imagination Task](regrasping_as_core_force_imagination_task.md)
