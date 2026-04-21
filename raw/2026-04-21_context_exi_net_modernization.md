# 2026-04-21 Context: EXI-Net modernization as the background for predictor and latent design

## Context

The background for the recent discussion is that the original plan for the predictor and physical-property latent was based on EXI-Net.
The goal was not to copy EXI-Net unchanged, but to preserve its useful core idea while replacing older components with a more modern implementation.

## Original EXI-Net role in the project

- EXI-Net was the original reference point for both the predictor and the latent physical-property representation.
- The appealing part was the idea of using a predictor conditioned on dynamics-related variables, then adapting only the dynamics-related part at deployment.
- The policy itself was not the main point of interest here; the focus was on the latent representation of hidden physical properties and the predictor that uses it.

## Why modernization is being considered

- EXI-Net feels somewhat old as an implementation template.
- The current aim is therefore to keep the underlying idea while replacing the architecture with something more modern and easier to extend.
- In particular, the recent discussion has been about what should replace the older latent-inference mechanism and predictor design.

## Current modernization direction

- Keep the EXI-Net spirit: freeze the main predictor weights at deployment and adapt only the dynamics or material representation.
- Replace optimization-heavy or older latent handling with a more modern learned inference approach where appropriate.
- Likely use a short-history encoder, potentially Transformer-based, to infer a task-relevant latent from recent observation-action history.
- Use that latent inside a force predictor or state-transition predictor rather than treating policy design as the primary research object.

## Relevance to recent comparison work

- The comparison with Li-san's face keypoint prediction was meant only as a reference for prediction-before-control, not as the main implementation template.
- The comparison with `dynamical-metalearning` was meant to check whether it is a better modern reference for hidden-dynamics inference than the facial prediction work.
- So the thread of discussion has been: EXI-Net as the conceptual starting point, then search for a more modern replacement for the predictor and latent mechanism.
