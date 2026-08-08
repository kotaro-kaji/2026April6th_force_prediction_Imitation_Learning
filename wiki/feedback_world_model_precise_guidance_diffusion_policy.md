# Feedback World Model Enables Precise Guidance of Diffusion Policy

## Summary

The paper predicts the next image-derived latent feature from the current visual feature, proprioception, and an action chunk. During deployment, the difference between the predicted and actually observed latent updates a lightweight feedback state while the world model and diffusion policy remain frozen. The corrected future latent then guides diffusion-policy sampling toward expert-like outcomes.

## Relation to This Research

This is close to the project's future-image-feature prediction direction because the predicted feature directly influences policy inference rather than serving only as an auxiliary loss.

The main difference is that this paper corrects transient visual-state prediction error. It does not use force/torque error, maintain a persistent object/material latent, or evaluate visually identical objects with different hidden physical properties. It is therefore a useful visual-only baseline for the proposed hidden-physics and force-imagination direction.

## Source

- Tuo An et al., *Feedback World Model Enables Precise Guidance of Diffusion Policy*, arXiv:2605.15705v1, 2026-05-15.
- [Local PDF](../raw/Feedback%20World%20Model%20Enables%20Precise%20Guidance%20of%20Diffusion%20Policy.pdf)
