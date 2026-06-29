# Regrasping as the Core Force-Imagination Task

## Summary

The latest direction shifts the task story from continuous force regulation to **regrasping or repositioning after hidden-physics identification**.

As of 2026-06-13, the framing is sharpened further:
the project should focus on **human-like retention of internal object properties for regrasping**, not on adding complexity just to differentiate from AdaWorldPolicy.
The current concrete task direction is a thin pouch that is pinched, lifted or stood up, and inserted into a slot or flexible opening.

The key idea is:

> Force information is used during contact to identify hidden physics, but the important policy decision may happen later, during a period where direct force information is absent or insufficient.

This makes the method more necessary than tasks such as sponge wiping or brush sweeping, where a reactive force-aware controller can often adjust the applied force online without needing a persistent hidden-physics latent.

## Why This Shift Matters

In surface-interaction tasks, the robot can often observe force during the same phase in which it needs to act.
If the sponge is too soft or the brush is too stiff, a force-feedback or compliance-aware policy can modify pressure immediately.
That weakens the need for a method that stores hidden physics in a compact latent.

Regrasping is different.
The robot may first touch, probe, or partially lift the object and infer its hidden center of mass or mass distribution from force/torque.
Then it must decide where or how to hold the object next.
During this decision phase, the relevant force signal may no longer be directly available.
The policy therefore needs a persistent latent that carries the identified hidden physics forward in time.

As of 2026-06-16, the earlier binary tactile/contact-state premise is legacy.
The current method should not be described as relying on explicit contact occurrence classification.

This is a cleaner motivation for force-imagination-based latent adaptation:

- contact reveals hidden physics,
- force-estimation error updates the latent,
- the latent persists after the contact evidence is gone,
- the policy uses that latent to choose the next grasp, support point, or lifting strategy.

## Current Candidate Task: Thin Pouch Pinch and Insertion

The current leading candidate is a thin pouch-like object.
The robot pinches an edge of the pouch while it lies flat on the table, lifts or stands it up, and inserts it into a slot, bag opening, backpack opening, or flexible container.

Why it fits:

- It uses a pinch grasp rather than requiring the gripper to open around the full object width.
- It involves reorientation and possible regrasping, not only lifting.
- The flexible pouch or receiving container makes pure rigid-body model-based planning less convincing.
- It returns the task to a manipulation regime where imitation learning is useful because geometry, contact, and deformation are difficult to model exactly.

The exact implementation still needs to specify whether the container is rigid, passive flexible, or actively held open by the second arm.
Dual arm is plausible if one arm manipulates the pouch while the other opens or stabilizes the receiving container.

## Deprioritized Candidate: One-Handed Lift of a Long Rectangular Object

The previous candidate was a remote-control-like long rectangular prism with unknown center of mass.
The robot must lift it with one hand.

Why it fits:

- The external shape can be fixed while the internal center of mass changes.
- Correct grasp or support location depends directly on the hidden center of mass.
- Initial contact or a small probing lift can reveal torque information.
- The final lifting or regrasping decision benefits from carrying the inferred center-of-mass latent forward.

This task is now weaker as the main story because it may be solved by a simple rule-based probing or binary-search-like grasp-location strategy.
It remains useful as a diagnostic benchmark, but it should not carry the main imitation-learning motivation unless the task is made less rule-solvable.

Failure can be made concrete:

- object tilts beyond a threshold,
- object slips or rotates in the gripper,
- lift cannot be completed without regrasping,
- excessive wrist torque or grasp force is required,
- the robot chooses a poor grasp point compared with the latent-adapted policy.

The task should avoid becoming a simple reactive lifting controller.
A strong design is to allow only one short probing interaction before the final grasp or lift.

## Candidate Task 2: Dual-Arm Non-Prehensile Box Lifting

The second candidate is dual-arm non-prehensile lifting of a box, motivated by moving boxes in real settings.
The box may look like an ordinary cardboard box, but the internal center of mass can be shifted.
A good lifting strategy should place the two support/contact points so that the moment is balanced.

Why it fits conceptually:

- The object appearance can remain almost unchanged.
- Hidden center of mass changes the required support points.
- The task naturally involves using earlier contact/force information to decide how to hold or reposition.
- It connects to an intuitive real-world use case: moving boxes with unknown contents.

Main experimental risk:

On a fixed-base robot, the box may not obviously fail even when the support is suboptimal.
Unlike a mobile manipulator or human mover, the robot base is stable, so instability may not lead to a dramatic drop or whole-body failure.
This makes success-rate improvement harder to demonstrate unless the failure criterion is designed carefully.

Possible evaluation metrics:

- final box tilt angle during lift,
- maximum roll/pitch oscillation after lift,
- peak contact force or wrench,
- required correction motions after initial lift,
- time to stable lift,
- whether the box can be transported through a narrow pose/orientation tolerance,
- whether the box remains stable under a mild standardized disturbance,
- number of regrasp or reposition attempts.

The disturbance should be a controlled experimental protocol rather than an ad hoc human push.
For example, use a repeatable small lateral impulse, a commanded short acceleration, or a required hold under a specified end-effector trajectory.

## Preferred Framing

The updated framing should not be:

> Force imagination is useful because contact-rich tasks need force control.

That is too broad and overlaps strongly with force-aware imitation, compliance learning, and reactive force control.

The sharper framing is:

> Force imagination identifies hidden object physics during contact, and the adapted latent is useful for later action phases where direct force information is absent, delayed, or insufficient.

This makes regrasping and support-point selection a more central story than continuous wiping or sweeping.

## Implication for Existing Task Notes

Sponge wiping and brush sweeping remain useful examples of visually ambiguous contact-quality tasks, but they are no longer the strongest main task if the goal is to show why a persistent latent is necessary.
Likewise, long-object center-of-mass lifting is no longer the strongest main task if it collapses into rule-based search.

The current leading task direction is:

1. Thin pouch pinch, reorientation, and insertion into a slot or flexible opening.
2. Dual-arm version where one arm opens or stabilizes the flexible container while the other inserts the pouch.
3. One-handed long-object lift or regrasping with unknown center of mass as a secondary diagnostic benchmark.

The key experimental question is no longer only whether force helps during contact.
It is whether force-derived hidden-physics information remains useful after the direct force signal is no longer available.

Earlier notes paired this question with a minimal contact-state signal.
That idea is now deprecated; the current task framing should stand without explicit contact-state gating.

## Source

- [thoughts/接触リッチ模倣学習.md](../thoughts/%E6%8E%A5%E8%A7%A6%E3%83%AA%E3%83%83%E3%83%81%E6%A8%A1%E5%80%A3%E5%AD%A6%E7%BF%92.md)
- [Legacy: Binary Tactile Contact Signal](binary_tactile_contact_signal.md)
- [Internal-Property Memory for Regrasping](internal_property_memory_for_regrasping.md)
