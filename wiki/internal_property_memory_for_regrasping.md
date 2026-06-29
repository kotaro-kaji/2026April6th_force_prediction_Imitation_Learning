# Internal-Property Memory for Regrasping

## Summary

The latest framing returns the project to the core problem:

> How can a robot behave like a human by identifying an object's internal physical property, keeping that information in memory, and using it for regrasping or repositioning?

This is more important than designing extra mechanisms only to differentiate from AdaWorldPolicy.
The project should not become complicated just to look different from prior online world-model adaptation methods.
The primary research value is the problem setting: **same or similar appearance, hidden internal property, and regrasping behavior that depends on remembering that property**.

## Main Lesson

The strongest task is not simply one where unknown physics exists.
It must be a task where unknown physics alone is not enough to justify learning if a simple rule-based or model-based search can solve it.

The advisor's critique is important:

- A long-object lifting task with unknown center of mass may be solved by rule-based probing or binary search over grasp positions.
- If the solution is essentially "try a grasp, observe tilt, move the grasp point," then the need for imitation learning or end-to-end policy learning becomes weak.
- The task should return to the reason imitation learning is useful: handling messy manipulation where precise modeling is difficult.

Therefore the task should combine hidden physical properties with a manipulation setting that is hard to solve cleanly by simple model-based rules.

## Current Concrete Task: Thin Pouch Pinch and Insertion

The most concrete current candidate is a thin pouch-like object.
The robot pinches an edge of a thin pouch on a table, lifts or stands it up, and inserts it into a slot, bag opening, backpack opening, or other compliant container.

The important details are:

- The pouch starts lying flat on the table.
- The robot should pinch an edge rather than grasp the whole 14 cm width.
- Pinching is compatible with the xArm7 / G1 gripper constraint better than trying to open the gripper around the full object.
- The task involves reorientation: from flat on the table to a vertical or insertion-ready pose.
- The final goal is insertion into a container or slot-like opening.

This task is stronger than a simple long-object center-of-mass lift because the difficult part is not only estimating a scalar hidden parameter.
The manipulation involves deformability, pinching, regrasping, contact transitions, and a target container that may itself be flexible.

## Why Flexible Containers Matter

The reason this task can resist simple model-based solutions is not just that the object is hard to grasp.
Grasping and regrasping can often be handled by model-based systems when geometry is clear and rigid.

The harder part is that the receiving container may be soft or deformable.
For example:

- a pouch being inserted into a soft bag,
- a paper bag opening that collapses,
- a backpack or fabric pocket whose opening changes shape,
- a flexible slot that deforms under contact.

In this setting, the robot needs to manage contact with deformable objects, partial observability, and uncertain contact state.
That makes the task closer to the original motivation for imitation learning than a clean rigid-body center-of-mass search problem.

## Single Arm vs Dual Arm

A single-arm version is simpler and cleaner:

- one arm pinches the pouch,
- lifts or stands it up,
- inserts it into a slot or opening.

A dual-arm version is more realistic and may not be too heavy experimentally:

- one arm pinches and manipulates the pouch,
- the other arm opens, stabilizes, or shapes the flexible container opening.

The dual-arm version better supports the "soft container" argument, because one arm can actively maintain the opening while the other performs insertion.
However, the first implementation should only choose dual arm if it is needed to make the container interaction meaningful.

## Rejected or Deprioritized Directions

### Long-object center-of-mass lifting

This remains conceptually related but is weaker as the main task.
The risk is that it becomes a rule-based search problem rather than an imitation-learning problem.

It can still be useful as a simulation or diagnostic benchmark, but it should not carry the main paper unless the task is made much less rule-solvable.

### Multiple same-appearance objects in one episode

The idea of storing separate latents for multiple visually identical objects in the same episode is currently not a good fit.
If the objects are visually identical, the system needs an object identity or spatial memory mechanism to know which latent belongs to which object.
The current method does not explicitly solve that kind of object-indexed memory problem.

If the goal is to remember that "the left object has one property and the right object has another," a method with explicit object memory or tracking may be needed.
AdaWorldPolicy-like methods may even be stronger in such a setting.

### Confidence / uncertainty as the core method

Object-property confidence or uncertainty may be useful, but it should not become the main story yet.
It risks making the method too complex before the problem setting is clean.

Possible confidence signals include:

- prediction error,
- accumulated contact time,
- closeness to an offline inferred latent,
- uncertainty over the object-property latent.

But the current priority is not to build a complex confidence-conditioned policy.
The priority is to define the task and problem setting clearly.

## Relationship to AdaWorldPolicy

The project should not be driven primarily by trying to look different from AdaWorldPolicy.
If the task, data quality, and research objective are different, the work can still be meaningful even if some method components look similar.

The current difference should be framed as:

- AdaWorldPolicy is a broad online world-model adaptation method.
- This project studies how robots can infer and retain internal object properties for regrasping or repositioning.
- The focus is the quality of the demonstration data, same/similar appearance under hidden internal differences, and manipulation phases where the robot must act using retained physical context.

The method still needs a clean minimal difference, but that difference should follow from the problem.
It should not be added only for related-work positioning.

## Updated Research Question

The current research question is:

> How can an imitation-learning policy use interaction-derived physical context to perform regrasping or repositioning when the object's internal property is not visible but affects how the object should be handled?

For the pouch task, this becomes:

> Can a robot learn to pinch, lift, reorient, and insert a thin object into a flexible opening while using interaction-derived context about object or container behavior, rather than relying only on visible pose replay?

## Practical Next Step

The next step should be task concretization, not method expansion.

The minimum task definition should specify:

- the exact pouch object,
- initial pose: flat on table,
- grasp mode: edge pinch,
- target: rigid slot or flexible opening,
- whether the container is fixed, passive flexible, or held by the second arm,
- success metric: inserted depth, final orientation, no drop, no severe folding or buckling,
- what hidden property varies: pouch stiffness, pouch thickness, weight distribution, container opening stiffness, or friction.

Only after this is fixed should the method decide whether it needs:

- force imagination,
- interaction residuals,
- legacy binary tactile/contact-state gating,
- confidence or uncertainty,
- explicit latent-only adaptation.

## Source

- [thoughts/接触リッチ模倣学習 2.md](../thoughts/%E6%8E%A5%E8%A7%A6%E3%83%AA%E3%83%83%E3%83%81%E6%A8%A1%E5%80%A3%E5%AD%A6%E7%BF%92%202.md)
