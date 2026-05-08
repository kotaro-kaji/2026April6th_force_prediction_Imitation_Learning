# Task Candidates for Hidden-Physics Imitation Learning

## Goal

Select a task where the same visible geometry can induce meaningfully different manipulation dynamics through hidden physical or material properties.
The task should reward estimating interaction force and adapting a compact latent from force-estimation error, rather than merely replaying a demonstrated trajectory.

## Current Discussion Status

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

### 1. Unknown-CoM PushT or constrained planar pushing

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

### 2. Unknown-CoM edge pivoting / tumbling to a target pose

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

### 3. Constrained pushing through a slot, gate, or narrow corridor with unknown CoM

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

### 4. Articulated object opening with hidden friction or resistance

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

Under the current working premise, the path is much narrower than this page originally assumed.
The main issue is not just to find a contact-rich task, but to find one where replaying a teleoperated dual-arm trajectory is not already enough.

1. Unseen-CoM / unseen-mass PushT with a stricter success geometry.
2. Wiping with sponges of different stiffness or hardness, if broadening from hidden mass toward hidden compliance is acceptable.
3. A constrained pushing variant only if it is clearly still part of the same pushing story.

At the moment, the two live directions are:

- PushT, if the project should keep center of mass and mass as the hidden variables.
- Wiping with varied sponge hardness, if the project can broaden toward hidden compliance as the latent factor.

## References

- Omnipush and mass-asymmetry-rich pushing dataset/report: [MIT News summary](https://news.mit.edu/2019/pushy-robots-learn-fundamentals-object-manipulation-1022)
- Adaptive pushing for unknown objects with different mass and friction distributions: [Pushing corridors for delivering unknown objects with a mobile robot](https://link.springer.com/article/10.1007/s10514-018-9804-8)
- Estimating mass distribution from non-prehensile interaction: [Estimating Mass Distribution of Articulated Objects Using Non-prehensile Manipulation](https://tml.stanford.edu/publications/2020/estimating-mass-distribution-articulated-objects-using-non-prehensile)
- Contact-rich insertion as a task family with nonlinear low-clearance trajectories and varying force requirements: [An Adaptive Imitation Learning Framework for Robotic Complex Contact-Rich Insertion Tasks](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.777363/full)
- Wiping with varying sponge properties and surface heights: [Adaptive Wiping](https://www.catalyzex.com/paper/adaptive-wiping-adaptive-contact-rich)
- Benchmark examples for articulated manipulation: [ManiSkill2 task suite](https://maniskill2.github.io/) and [OpenCabinetDrawer documentation](https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer/index.html)
