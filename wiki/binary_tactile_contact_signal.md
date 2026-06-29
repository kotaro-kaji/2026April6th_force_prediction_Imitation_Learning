# Legacy: Binary Tactile Contact Signal

## Summary

This page records a **legacy idea** from 2026-06-11.
As of 2026-06-16, explicit binary tactile/contact-state gating is no longer part of the current proposed method.
The current research theme should not be described as requiring contact/no-contact classification.

The old idea was that contact occurrence should not be inferred from images and should instead be provided by a minimal tactile or binary contact signal.
That framing has been superseded.

In the legacy framing, the project would treat tactile sensing, even in a very low-resolution or binary form, as a core signal.
The tactile signal does not need to be rich.
The minimum useful form is:

`contact = 0 or 1`

This binary contact signal was proposed as a potentially reliable way to decide whether interaction evidence is available.

## Legacy Motivation

A vision system can sometimes infer contact from object motion, deformation, or very close camera views.
But using images to decide whether contact has occurred can require unusually high resolution, careful camera placement, or extreme close-up observation.
That makes contact detection fragile.

In contrast, a skin-like tactile surface can reduce the problem to a simple binary event:

- not touching,
- touching.

In the legacy framing, this changed the sensor-design premise:
the method would not rely only on force/torque sensing or visual force imagination, and would instead use **tactile contact state plus force/torque** as evidence for object-physics identification.

## Relation to Hidden-Physics Identification

For hidden-physics identification, the model needs to know when an interaction actually happened.
Force, torque, slip, object motion, or deformation are only meaningful as evidence when contact is present.

In that older design, the binary tactile signal could act as:

- a contact-state observation,
- a mask for when force/torque residuals should update the latent,
- a boundary signal for probing and regrasping phases,
- a way to avoid interpreting non-contact motion as physical interaction.

This was previously connected to the regrasping direction.
That connection is now deprecated, because the current method should not assume an explicit contact classifier or binary contact gate.

## Force and Tactile Are Complementary

The legacy argument treated force/torque sensing and tactile contact sensing as complementary rather than substitutes.
They were understood as answering different questions.

Force/torque answers:

- how large the interaction force or moment is,
- in what direction it acts,
- how the object responds dynamically.

Binary tactile sensing answers:

- whether contact exists at all,
- which contact region is active if the skin has multiple binary cells,
- when contact starts or ends.

The old intuition was that humans likely use both.
This is no longer a current method requirement.

## Legacy Design Implication

The deprecated design would have considered adding a tactile-equivalent input even if it was extremely simple.
Examples included:

- one binary contact switch at the fingertip,
- multiple binary contact pads on the gripper,
- thresholded tactile array values,
- contact/no-contact labels from a simple skin or bumper,
- contact state derived from a reliable non-visual sensor.

The old point was not high-resolution tactile perception, but robust contact detection.

The old model input would have included something like:

`observation = vision + proprioception + action history + force/torque + binary contact state`

and the latent update would have been gated or interpreted by contact state:

`update d_i only when contact = 1`

or, more generally:

`use contact state to decide when interaction residuals are physically meaningful`.

## Superseded Positioning

This positioning is now superseded.
The current project should not be framed as requiring minimal tactile contact state or contact/no-contact classification.

Old claim:

> Contact-rich imitation requires knowing not only the magnitude of interaction forces, but also whether and when contact is actually occurring. A low-resolution or binary tactile signal can provide this contact state more reliably than vision, and can gate force/residual-based updates of a persistent hidden-physics latent.

This claim is retained only for historical traceability.

## Source

- [thoughts/バイナリ型触覚の重要性。6月11日に気がついたこと。.md](../thoughts/%E3%83%90%E3%82%A4%E3%83%8A%E3%83%AA%E5%9E%8B%E8%A7%A6%E8%A6%9A%E3%81%AE%E9%87%8D%E8%A6%81%E6%80%A7%E3%80%826%E6%9C%8811%E6%97%A5%E3%81%AB%E6%B0%97%E3%81%8C%E3%81%A4%E3%81%84%E3%81%9F%E3%81%93%E3%81%A8%E3%80%82.md)
