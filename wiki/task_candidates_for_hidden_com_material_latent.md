# Task Candidates for Hidden-Physics Imitation Learning

## Goal

Select a task where the same visible geometry can induce meaningfully different manipulation dynamics through hidden physical or material properties.
The task should reward estimating interaction force and adapting a compact latent from force-estimation error, rather than merely replaying a demonstrated trajectory.

## Current Discussion Status

As of 2026-06-13, the newest direction returns to the problem setting rather than related-work differentiation:
the project is about making a robot retain internal object-property information and use it for regrasping, like a human.
The current concrete task candidate is a thin pouch that starts flat on the table, is pinched at an edge, lifted or stood up, and inserted into a slot or flexible opening.
This is preferred over a clean long-object center-of-mass lift if that lift can be solved by rule-based probing or binary-search-like grasp-location adjustment.

As of 2026-06-10, the latest direction shifts the main task story toward **regrasping and lifting after hidden-physics identification**.
The core argument is that force information is most valuable when it is used to infer hidden center of mass or mass distribution during contact, then carried forward into a later action phase where direct force information is absent or insufficient.
This makes long-object lifting with unknown center of mass and dual-arm non-prehensile box lifting the most live task candidates.

This page should currently be read as a narrowing note rather than a broad recommendation list.
As of 2026-04-19, the user's working premise is that dual-arm box uprighting is probably too easy to solve by replaying a dual-arm teleoperation trajectory.
In that task, unseen center of mass and total mass may be close to irrelevant in practice.
Under that premise, unseen-mass / unseen-CoM PushT still looks like the clearest option if the project remains centered on mass distribution.
At the same time, a wiping task with sponges of different stiffness or hardness has emerged as a promising alternative if the hidden variable can shift from mass distribution toward compliance.

## Evaluation Criteria

- Same or near-identical appearance, different dynamics.
- Trajectory replay alone should fail often enough.
- Force estimation and online latent adaptation should have a clear causal role.
- The task should still be practical for simulation data collection and later real-robot execution.

## Practical Ranking Under The Current Premise

### 1. Thin pouch pinch and insertion into a slot or flexible opening

This is currently the strongest concrete candidate.
The object is a thin pouch-like item lying flat on the table.
The robot pinches an edge, lifts or stands it up, and inserts it into a slot, bag opening, backpack opening, or other container.

Why it fits:

- Pinching avoids requiring the G1 gripper to open around the full object width.
- The task requires reorientation and possibly regrasping, not just lifting.
- A flexible pouch or flexible receiving container makes simple model-based rigid-body planning weaker.
- The task better matches the original purpose of imitation learning: handling manipulation where exact modeling is difficult.

Design recommendation:

- Start with a clearly defined pouch object and initial flat pose.
- Define whether the target opening is rigid, passive flexible, or held open by a second arm.
- Use success metrics such as inserted depth, final orientation, no drop, no severe folding, and no container collapse.
- Vary only one hidden factor at first, such as pouch stiffness, pouch thickness, weight distribution, or opening stiffness.

### 2. Dual-arm pouch insertion with flexible container opening

This is a stronger but more complex version of the pouch task.
One arm pinches and manipulates the pouch, while the other arm opens or stabilizes a flexible container.

Why it fits:

- The second arm gives a natural role beyond just making the task larger.
- The flexible opening creates a reason that pure model-based insertion is hard.
- The task can express human-like regrasping and object-property retention more clearly than rigid-body lifting.

Main weakness:

- The experimental setup and demonstrations are more complex.
- The container behavior must be controlled enough that failures are interpretable.

### 3. One-handed long-object lift with unknown center of mass

This is currently the cleanest candidate under the latest direction.
The object is a remote-control-like long rectangular prism with fixed external appearance but shifted internal center of mass.
The robot must use contact or a short probing lift to infer the center of mass, then choose a grasp, regrasp, or lifting strategy.

Why it fits:

- The hidden variable is clear and physically meaningful.
- The correct grasp or support point depends on the hidden center of mass.
- Force/torque during early contact is informative.
- The policy must carry the inferred latent into the later lift or regrasp phase.
- Failure can be measured by tilt, slip, excessive torque, failed lift, or poor grasp location.

Design recommendation:

- Allow only one short probing interaction before the final lift.
- Keep the external geometry and texture fixed.
- Penalize excessive tilt, slip, or wrist torque.
- Compare against policies that use current observations or recent force history without a persistent latent.

Latest caveat:

- This task may be too rule-solvable.
- If a binary-search-like probing strategy can find the grasp point, then the need for imitation learning becomes weak.
- It is better used as a diagnostic benchmark than as the main paper task unless redesigned.

### 4. Dual-arm non-prehensile box lifting with shifted internal center of mass

This is the most intuitive real-world candidate.
The robot lifts a visually similar box whose internal center of mass may be shifted, and must choose two contact/support points that balance the moment.

Why it fits:

- The task matches the motivating scenario of moving boxes with unknown contents.
- Same-looking boxes can require different support points.
- Force feedback from initial contact can identify the latent, while the later lift benefits from remembering it.

Main weakness:

- On a fixed-base robot, unstable lifting may not naturally create dramatic failures.
- Success-rate differences may be hard to show unless the evaluation makes instability consequential.

Design recommendation:

- Evaluate final tilt, oscillation, peak wrench, correction count, lift time, and stability under a standardized disturbance.
- Use a repeatable commanded acceleration or trajectory tolerance instead of an ad hoc human push.
- Treat this as promising but experimentally riskier than the one-handed long-object task.

### 5. Unknown-CoM PushT or constrained planar pushing

This is currently the clearest direction if the project stays centered on hidden mass and center of mass.
Keeping the external shape fixed while shifting internal mass directly changes how the object rotates during pushing.
That is well aligned with hidden latent estimation and avoids the ambiguity of geometry changes.

Why it fits:

- Unknown mass distribution already appears as a key source of variation in pushing literature and datasets such as Omnipush.
- Online adaptation during pushing has prior art, including adaptive controllers for unknown objects with different mass and friction distributions.
- It is simple to scale in simulation, easy to label, and likely feasible on hardware.

Main weakness:

- If the corridor is too wide or the target too forgiving, the task can collapse into slow closed-loop correction rather than rewarding good latent identification.

Design recommendation:

- Do not use free-space pushing only.
- Use a corridor, narrow goal region, or rotation-sensitive target so that incorrect early pushes cannot be cheaply repaired.

### 6. Unknown-CoM edge pivoting / tumbling to a target pose

This is no longer a preferred direction under the current premise.
Instead of merely pushing to a planar goal, the robot repeatedly tilts and pivots the same-looking box around edges or corners to reach a target upright or reoriented pose.

Why it fits:

- The success strongly depends on hidden center of mass and inertia, not only visible shape.
- The transition between sticking, slipping, and tipping is force-sensitive.
- Once the robot has identified the latent physics, the motion can plausibly become more ballistic or open-loop-like.

Why it was previously considered:

- The problem can be redesigned so that one memorized trajectory no longer works across CoM shifts.
- Repeated pivot steps create multiple chances where wrong latent estimates cause irreversible pose error.

Main weakness:

- It is harder than PushT to stabilize and evaluate cleanly.
- Contact mode switches can make demonstrations and simulation tuning more brittle.
- More importantly, it may still admit success by replaying a strong dual-arm demonstration, which weakens the case that hidden center of mass or mass are causally necessary.

Design recommendation:

- Use a target pose after one or more pivot steps, not just “somehow stand up once.”
- Randomize internal CoM while fixing outer shape and contact geometry.
- Make success depend on final orientation and position, not only eventual toppling outcome.

### 7. Constrained pushing through a slot, gate, or narrow corridor with unknown CoM

This remains relevant mainly as a variant of the PushT direction, not as a separate research story.
The object must be pushed through a geometry that leaves little room for correcting rotation bias.

Why it fits:

- Unknown CoM produces systematic drift and yaw under nearly identical visual input.
- Early force observations can inform the latent, and then the robot can commit to a better subsequent push.
- The task remains simple, non-prehensile, and compatible with your current box-like setup.

Main weakness:

- If the corridor is too long, the task may become mostly feedback correction.
- If it is too narrow, it may become a pure precision-control problem.

Design recommendation:

- Keep the horizon short and penalties sharp.
- Prefer one decisive push plus one recovery action over many tiny corrections.

### 8. Articulated object opening with hidden friction or resistance

Examples are drawer opening, door opening, and faucet turning.
These are standard contact-rich benchmarks and explicitly involve articulated mechanisms.

Why it fits:

- Hidden physical variables such as hinge friction, damping, backlash, or handle resistance matter even when appearance is similar.
- Force estimation can be meaningful because the required interaction force evolves over the motion.

Why it is only fourth:

- The latent is no longer mainly object inertial property but mechanism property.
- This drifts away from your current “same box, different internal mass/CoM” story.
- It is less natural for the “identify latent, then move more open-loop” narrative than hidden-CoM rigid-body tasks.

## Lower-Priority or Poor-Fit Tasks

### Insertion

Insertion is clearly contact-rich and literature emphasizes varying force requirements across phases.
However, it is a weaker fit for your current theme because failure is often dominated by geometric alignment and tight-clearance contact, not hidden inertial properties.
It is better as a later extension if you shift toward general force-aware imitation learning rather than hidden material latent for rigid objects.

### Wiping / scraping / polishing

These tasks strongly need force feedback and have good recent results with adaptive force control.
Previously this page treated wiping as a weaker fit.
That assessment should now be considered outdated for the current discussion.
If the hidden factor is sponge stiffness or hardness, wiping may become a strong benchmark because the material property directly changes contact mechanics and task success.
The main tradeoff is that this shifts the story away from hidden center of mass and mass distribution toward hidden compliance.

## Bottom Line

Under the latest working premise, the path is narrower and more specific than "contact-rich manipulation."
The main issue is to find a task where force-derived hidden-physics information must be remembered for a later regrasping, support selection, or lifting decision.

1. Thin pouch pinch, reorientation, and insertion into a slot or flexible opening.
2. Dual-arm pouch insertion where the second arm opens or stabilizes a flexible container.
3. One-handed long-object lift or regrasping with unknown center of mass as a secondary diagnostic benchmark.

At the moment, the two live directions are:

- thin-pouch pinch insertion, if the project should prioritize a task where imitation learning is naturally motivated;
- dual-arm flexible-container insertion, if the project should emphasize deformable container interaction and can keep the setup controlled.

## References

- Omnipush and mass-asymmetry-rich pushing dataset/report: [MIT News summary](https://news.mit.edu/2019/pushy-robots-learn-fundamentals-object-manipulation-1022)
- Adaptive pushing for unknown objects with different mass and friction distributions: [Pushing corridors for delivering unknown objects with a mobile robot](https://link.springer.com/article/10.1007/s10514-018-9804-8)
- Estimating mass distribution from non-prehensile interaction: [Estimating Mass Distribution of Articulated Objects Using Non-prehensile Manipulation](https://tml.stanford.edu/publications/2020/estimating-mass-distribution-articulated-objects-using-non-prehensile)
- Contact-rich insertion as a task family with nonlinear low-clearance trajectories and varying force requirements: [An Adaptive Imitation Learning Framework for Robotic Complex Contact-Rich Insertion Tasks](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.777363/full)
- Wiping with varying sponge properties and surface heights: [Adaptive Wiping](https://www.catalyzex.com/paper/adaptive-wiping-adaptive-contact-rich)
- Benchmark examples for articulated manipulation: [ManiSkill2 task suite](https://maniskill2.github.io/) and [OpenCabinetDrawer documentation](https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer/index.html)
- Latest direction note: [Regrasping as the Core Force-Imagination Task](regrasping_as_core_force_imagination_task.md)
- Latest June 13 direction: [Internal-Property Memory for Regrasping](internal_property_memory_for_regrasping.md)
