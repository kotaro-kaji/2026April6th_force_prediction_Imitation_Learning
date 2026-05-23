# Task Search Beyond Sponge Wiping for Force Imagination

## Question

Are there tasks other than wiping with sponges of different softness where force imagination is clearly better motivated than visual world-model prediction?

## Current Answer

The set is very small.
Most contact-rich manipulation tasks look promising at first, but many collapse under closer inspection because the relevant hidden property eventually appears as visible motion, deformation, task progress, or failure.

The current strongest real-world task family is **surface interaction with visually similar compliant tools**.
After adding the practical constraints of imitation learning, an xArm7 gripper, and ordinary laboratory execution, most candidates fail.
However, two important real-world candidates survive: wiping with sponges of different softness, and sweeping coffee beans with brushes of different stiffness.

A separate simulation-friendly candidate also survives: single-arm grasp-and-lift of a long rectangular prism with unknown center of mass.

> The important hidden property should change the force needed for success, while not being reliably observable from ordinary camera prediction.

## Stricter Criterion

A task is a good fit only if all of the following are true:

- Same or nearly same visual appearance.
- Different internal material or contact property.
- Correct action depends on that property.
- The property is expressed strongly in force.
- The property is not reliably exposed in ordinary future images.
- A visual world model could plausibly be insufficient, not merely more expensive.
- The task is realistically executable with imitation learning, an xArm7 gripper, and normal lab equipment.

This criterion rejects many superficially attractive tasks.

## Rejected Or Weak Candidates

### PushT With Hidden Mass, Friction, Or CoM

This is weak as the main differentiating task.
It is useful as a controlled simulation benchmark, but hidden mass, friction, and center of mass often become visible through object motion.
If the object rotates, slides less, or drifts differently, future image prediction can in principle learn the same latent factor.

The honest conclusion is that PushT can test whether force imagination helps, but it does not strongly prove that force imagination is necessary.

### Drawer, Door, Faucet, Or Articulated Opening

These tasks initially seem good because hinge friction, damping, and mechanism resistance are force-relevant.
However, task progress is visually obvious: the drawer opens, the door angle changes, the faucet rotates.
A visual world model can often use the mismatch between commanded motion and observed motion to infer resistance.

They may still be useful extensions, but they are not clean enough to be the central argument.

### Peg Insertion, Connector Insertion, And Assembly

These are contact-rich and force-sensitive, but failure is often dominated by geometric alignment, jamming, and clearance.
The hidden variable is usually not an internal material property of the manipulated object.
Force helps control, but the task does not naturally isolate "same-looking object, different hidden physical property."

This drifts toward general force-aware manipulation rather than the specific force-imagination/material-latent thesis.

### Grasping Or Lifting Unknown-Weight Objects

Unknown weight is force-relevant, but the solution can be simple reactive control: lift, measure load, grip harder.
If the visual task is just "pick it up," imitation learning may not need a rich internal material representation.
Also, once the object accelerates or fails to lift, the difference can become visible.

This is likely too simple or too close to standard force feedback.

### Pouring Or Scooping With Different Contents

The hidden content changes weight, inertia, sloshing, and resistance.
However, many important effects are visible through fluid motion, container motion, or task progress.
The task also introduces complex perception and evaluation issues that can distract from the core claim.

It may be interesting, but it is not a clean first paper task.

### Cutting Food Or Deformable Objects

Hardness and toughness are force-relevant and often not visible.
This seems conceptually strong.
The problem is experimental complexity: cutting involves sharp tools, irreversible state changes, difficult safety constraints, and messy success metrics.
It may be a strong future direction but is risky as the main validation.

### Pressing Buttons, Keys, Or Switches With Different Stiffness

This is too small.
Different stiffness is force-relevant, but the task success is often binary and easily solved by pressing harder or using impedance control.
It may not require a learned internal physical-property representation.

### Cloth, Rope, Or Elastic Object Manipulation

Material properties matter, but the deformation is usually highly visible.
If the object shape changes are in view, visual prediction can plausibly capture the relevant state.
These tasks also add high-dimensional geometry and tracking difficulty, weakening the clean argument for force imagination.

## Surviving Candidates

### 1. Wiping With Visually Similar Sponges Of Different Softness

This remains the best current task.
Softness changes contact area, normal-force distribution, friction, deformation, and cleaning efficiency.
The robot may need to choose different force or motion even when the camera view does not clearly show the sponge's internal compression.

Why it survives:

- The hidden property is material compliance.
- The property is expressed directly in force.
- The camera may not reliably observe the relevant compression or pressure distribution.
- Success depends on contact mechanics, not only visible object motion.
- It is feasible on real hardware.

Main risk:

- If the camera clearly sees sponge deformation or dirt removal progress, visual prediction may still be competitive.
The experimental setup should avoid making the relevant compression trivially visible from a side camera.

### 2. Sweeping Coffee Beans With Brushes Of Different Stiffness

This is a strong real-world candidate under the xArm7 lab constraint.
The robot holds visually similar brushes with different bristle stiffness and sweeps coffee beans to a target area.

Why it survives:

- It is feasible with an xArm7 gripper and ordinary lab materials.
- The manipulated objects, coffee beans, are easy to reset, count, and visually evaluate.
- Brush stiffness changes the relation between end-effector motion, contact force, bristle deformation, and bean motion.
- The same visible brush trajectory may produce different sweeping results depending on stiffness.
- The needed adjustment is force/contact-quality dependent, not only pose tracking.

Why it helps the paper:

- It is close to wiping as a surface-interaction task, but not identical.
- It shows that the method is not only for sponge wiping or dirt removal.
- It keeps the central idea: hidden compliance affects contact force and task success.

Main risks:

- If bean motion is fully visible and dense enough, a visual world model may still infer the effect from image prediction.
- If the task is too easy, all brush stiffnesses may work with the same conservative sweeping motion.
- The design should make stiffness matter by choosing bean density, target geometry, sweep speed, and brush contact angle carefully.

### 3. Single-Arm Grasp-And-Lift Of A Long Rectangular Prism With Unknown Center Of Mass

This is a promising simulation task.
The robot must grasp and lift a long rectangular prism whose external appearance is fixed but whose center of mass is unknown.
If the grasp point is wrong, the object tilts, slips, or produces torque that makes the lift unstable.

Why it survives:

- The hidden variable is physically clear: center of mass along the long axis.
- The correct grasp location depends on the hidden center of mass.
- The force/torque response during initial contact or slight lifting is directly informative.
- It can be simulated cleanly and scaled across many CoM values.
- It is better than generic unknown-weight lifting because the task requires choosing where to grasp, not only how hard to grip.

Why it helps the paper:

- It provides a non-surface-interaction simulation benchmark.
- It returns to the original hidden-mass / hidden-CoM story.
- It is less obviously solved by a single safe reactive policy than ordinary lifting, because the key is selecting the correct support point.

Main risks:

- Once the object tilts, the CoM error becomes visually obvious, so visual prediction may still be a strong baseline.
- If the policy is allowed to adjust after observing tilt, the task may become simple feedback correction.
- The task should reward choosing a good grasp point before or during the earliest lift phase, not repeated regrasping after visible failure.

Design recommendation:

- Use one grasp or one short probing-lift before the final lift.
- Fix external geometry and texture while varying only internal CoM.
- Penalize large tilt, slip, or table contact after lift begins.
- Compare against image-prediction world-model baselines because this task is more visually observable than sponge wiping or brush sweeping.

## Practically Rejected After xArm7 / Lab Constraint

### Scraping Or Polishing With Visually Similar Tools Of Different Compliance

This is a close relative of wiping.
Examples include scraping residue with rubber/silicone tools of different stiffness, polishing with pads of different compliance, or removing thin material from a surface.

Why it initially seemed plausible:

- Tool compliance changes the effective contact patch and required force.
- The visually observed tool motion may be almost identical while the contact pressure differs.
- Success depends on pressure and friction at the interface, not simply on reaching a visible pose.

Why it is rejected now:

- It requires a stable tool fixture or custom end-effector.
- It is hard to make demonstrations clean with an xArm7 gripper.
- Surface residue, scraping angle, tool wear, and evaluation can become uncontrolled.
- It is too close to wiping while being more experimentally annoying.

### Applying Adhesive, Tape, Or Lamination With Different Tool/Surface Compliance

Examples include pressing tape, applying a sticker, laminating a sheet, or using a roller on surfaces with different compliance.

Why it initially seemed plausible:

- The task depends on pressure distribution and contact quality.
- Failure can happen even when the visual trajectory looks correct.
- Force response is more directly tied to success than future image prediction.

Why it is rejected now:

- Evaluation can be annoying unless success is measured cleanly.
- Adhesion introduces material variability and one-shot irreversible effects.
- Demonstrations are likely messy and reset is costly.
- It is not a natural fit for a standard xArm7 gripper without additional tooling.
- The task complexity would distract from the force-imagination claim.

### Brushing Or Cleaning With Hidden Bristle Stiffness

This was initially rejected as too messy in the abstract.
That rejection was too broad.
The concrete variant **sweeping coffee beans with brushes of different stiffness** is viable and is now listed as a surviving candidate.

Generic brush cleaning remains weaker because the dirt-removal target, brush mounting, and bristle wear can become nuisance variables.

Why it initially seemed plausible:

- Bristle stiffness affects contact force and cleaning effectiveness.
- The bristle deformation may be visually hard to estimate from an overhead or wrist camera.

Why generic brush cleaning remains risky:

- If the bristles are visible, image prediction may capture deformation.
- The task may become visually messy because dirt removal itself is visible.
- Brush mounting, bristle wear, and cleaning-material consistency add nuisance variables.
- It is harder than sponge wiping unless the task is concretized as a clean, resettable sweeping setup such as coffee beans.

## Current Ranking

1. Wiping with visually similar sponges of different softness.
2. Sweeping coffee beans with brushes of different stiffness.
3. Single-arm grasp-and-lift of a long rectangular prism with unknown center of mass, mainly as a simulation task.

Everything else is either too visually observable, too geometrically dominated, too simple, experimentally too messy, or unrealistic for imitation learning with an xArm7 gripper in an ordinary lab setup.

## Important Self-Criticism

The real conclusion is narrower than broad "robot manipulation", but not limited to sponge wiping alone.
Under the current hardware and experimental constraints, the practical real-world task family is:

> surface interaction with visually similar compliant tools whose stiffness changes the required contact force and contact outcome.

This includes sponge wiping and coffee-bean sweeping with stiffness-varied brushes.

The broader conceptual family remains:

> contact-rich surface interaction where success depends on pressure, friction, compliance, or contact-area distribution that is not directly visible.

This is narrower than originally hoped, but it is also clearer.
The paper should probably lean into this narrowness instead of pretending that force imagination is universally needed.

If the task family is framed this way, sponge wiping and brush sweeping are representative instances of **visually ambiguous contact-quality tasks**.
The long-prism grasp-and-lift task can be used as a simulation-only complement for hidden CoM, but its relation to visual world models must be handled more carefully because tilt and motion can become visually observable.

## Paper Positioning Implication

The related-work claim should not be:

> Existing visual world models cannot solve hidden physics.

That is too strong.

The safer and sharper claim is:

> Visual world models adapt through predicted visible state changes, while this work targets contact-quality variables whose task relevance is expressed more directly in force than in ordinary images.

Under this framing, the value of the method is not just lower computational cost.
The value is that the auxiliary prediction target is better aligned with the hidden variable needed for successful contact-rich surface manipulation.

## Source

- User question on 2026-05-21: whether any task beyond sponge wiping really satisfies this criterion, with the concern that maybe there are almost none.
- Related page: [Prior-Art Difference: Visual World Models vs Force Imagination](prior_art_difference_visual_world_model_vs_force_imagination.md)
