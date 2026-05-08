# Overlap With ACP, UMI-FT, CompACT, and ForceMimic

## Question

Does the current research direction stay meaningfully distinct from recent work on compliant or force-aware imitation learning?

## Short Answer

Partly yes, but only if the project is framed around **object-specific latent identification for hidden physical properties**.
If the contribution is framed mainly as force-aware imitation or compliance-aware control, then the overlap with recent work is strong.

## Why It Can Be Distinct

The current project summary is centered on learning a force estimator and an object/material latent `d_i`, then adapting only `d_i` online at test time for unseen physical properties such as mass, center of mass, or compliance.
In that framing, measured force/torque is mainly supervision and online feedback for latent update, while the policy should rely on estimated force or the adapted latent rather than recent measured force history.
That differs from recent work whose main goal is usually better force modulation, stiffness prediction, or compliant execution for a task.

## Paper-by-Paper Assessment

### Adaptive Compliance Policy (ACP)

- ACP learns a policy that predicts pose, force-related signals, and spatial-temporal stiffness for compliant execution.
- Its main claim is adaptive compliance control for maintaining good contact modes under uncertainty.
- It does handle object and scene variation, but not through an explicit per-object latent that is identified and updated as the main representation.

Assessment:
This is **close in control style and task family**, but still not the same core thesis if your thesis is latent identification of hidden object physics.
If your paper reads like "we use force to improve contact-rich imitation," ACP is a direct threat.
If your paper reads like "we infer a compact object identity / hidden-physics latent and adapt it online," the separation is clearer.

### UMI-FT

- UMI-FT extends the ACP line with scalable in-the-wild multimodal data collection using per-finger force/torque sensing.
- The learned policy predicts position targets, grasp force, and stiffness for execution on standard compliance controllers.
- Its emphasis is scalable force-aware demonstration capture and compliant manipulation, not object-specific latent inference.

Assessment:
This is probably the **closest practical neighbor** if you move toward wiping with different sponge hardness, because it already covers wiping and force-sensitive manipulation with a modified ACP-style policy.
Still, the main representation is not "latent hidden physics of each object/tool."
So the distinction survives, but only if you foreground latent inference rather than compliant control.

### CompACT

- CompACT learns variable compliance control from a few demonstrations using ACT-style action chunking.
- It is strongly aligned with the idea that demonstrations should teach when the robot should be softer or stiffer.
- It does not appear to focus on identifying persistent object-specific hidden parameters as an explicit latent state.

Assessment:
This is **conceptually close** because it learns variable compliance from demonstrations.
But it is still mainly a compliance-learning paper, not a hidden-object-representation paper.
Compared with ACP and UMI-FT, it is less dangerous to your exact framing, but still dangerous if your contribution is presented too generically.

### ForceMimic

- ForceMimic is force-centric imitation learning with a dedicated force-motion capture system and hybrid force-position execution.
- It predicts wrench-position behavior and uses force information to improve contact-rich execution.
- Its emphasis is realistic force demonstration and force-centric policy learning, especially for tasks like peeling.

Assessment:
This is **nearby but more separable**.
It is much more about force trajectory capture and reproduction than about inferring a reusable latent for hidden physical properties across object instances.
Among the four, this is the easiest one to argue against if your framing stays representation-centric.

## Bottom Line

You can still say the topic is distinct, but only in a narrow and disciplined way:

- Not: learning force-aware or compliance-aware imitation for contact-rich manipulation.
- Yes: learning a force estimator supervised by measured force/torque, then using force-estimation error to adapt a compact object/material latent online while keeping the policy and estimator fixed.

## Implication For Task Choice

The distinction is stronger when:

- visual geometry is nearly unchanged across object instances,
- success depends on hidden physics,
- the same visible scene can require different actions because of object-specific latent properties.

That is exactly why unseen mass / center-of-mass variants or sponge-hardness variants are better than tasks that can be solved by replaying one good trajectory.

## Safer Claim Style

Safer paper claim:
"We study force-estimation-error-driven identification of object/material properties, and use the adapted latent or estimated force to condition imitation policies for unseen objects."

Less safe claim:
"We improve contact-rich imitation learning using force information."

## Sources

- Local project summary: [Research Theme 1: Force Estimation and Material Latent for Imitation Learning](research_theme_1_force_estimation_material_imitation.md)
- ACP project page: https://adaptive-compliance.github.io/
- UMI-FT project page: https://umi-ft.github.io/
- CompACT project page: https://omron-sinicx.github.io/CompACT/
- ForceMimic project page: https://forcemimic.github.io/

## Date

Checked against public project pages and linked paper metadata on 2026-04-21.
