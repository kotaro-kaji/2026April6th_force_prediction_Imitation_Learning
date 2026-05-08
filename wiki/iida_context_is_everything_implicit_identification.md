# IIDA: Context Is Everything for Implicit Identification

## Summary

`Context is Everything: Implicit Identification for Dynamics Adaptation` proposes IIDA, a dynamics-adaptation method that infers an implicit environment factor from contextual transition data.
The method does not require ground-truth physical parameters such as mass, friction, or center of mass.
Instead, it learns a context encoder `g_phi` that maps context transitions from the same environment into a latent `z_e`, then conditions a dynamics predictor `f_theta(s, a; z_e)` on that latent.

## What a Context Point Means

A context point is one transition tuple:

`(s_i, a_i, s'_i)`

The context set is:

`{(s_i, a_i, s'_i)}_{i=1}^N`

So `N` is the number of transition examples from the current environment, not the dimension of the latent.
Each context point is encoded by `g_phi`, and the resulting vectors are summarized into a single environment latent `z_e`.

## Dimension of `z_e`

The paper explicitly uses:

`z_e in R^8`

The training details state that the latent dimension is 8 for all IIDA models.
This applies across the average-pooling, RNN, and Transformer context summarizers.

Important distinction:

- `z_e` dimension: 8
- RNN hidden size: 256
- Transformer attention size: 120 with 5 heads
- Dynamics model hidden size: 256

The larger hidden sizes are internal network widths.
They are not the final environment latent dimension passed into the dynamics predictor.

## Context Summarizers

IIDA compares three order-invariant or context-aggregation choices:

- Average pooling: encode each transition with the same feed-forward encoder, then average the vectors.
- RNN: treat randomly ordered context tuples as a sequence and project the last LSTM hidden state into `z_e`.
- Transformer: omit positional encoding so the model can attend over context points without imposing temporal order, then project to the required latent dimension.

## Relevance to This Project

IIDA is directly relevant because it cleanly separates:

- context data from the current environment,
- an inferred compact environment latent,
- and a dynamics predictor conditioned on that latent.

This matches the project direction better than pure domain randomization, because the predictor changes behavior based on identified hidden dynamics.
It also supports the current thesis direction that an explicit compact latent can be the main object of study.

For the force-estimation/material-latent project, the analogous design would be:

`visual/proprio/action context -> g_phi -> z_env -> force estimator or policy conditioning`

The most useful concrete reference point is that IIDA uses a very small latent, `8` dimensions, even when the true environment variation includes multiple hidden physical parameters.

## Sources

- [raw/2203.05549v1 (1).pdf](../raw/2203.05549v1%20(1).pdf)
- [Research Theme 1: Force Estimation and Material Latent for Imitation Learning](research_theme_1_force_estimation_material_imitation.md)
