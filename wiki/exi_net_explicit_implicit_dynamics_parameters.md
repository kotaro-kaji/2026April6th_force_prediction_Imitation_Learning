# EXI-Net Explicit and Implicit Dynamics Parameters

## Summary

`EXI-Net: EXplicitly/Implicitly Conditioned Network for Multiple Environment Sim-to-Real Transfer` learns a predictive dynamics model conditioned on two groups of dynamics parameters:

`s_{t+1} = f(s_t, a_t, d_e, d_i)`

In the object-pushing experiments, the state can be moved to the object-centered frame, so the model is simplified to:

`Delta s_{t+1} = f(a_t, d_e, d_i)`

## Dimensions

For the standard EXI-Net method in the paper:

- `d_e`: 4 dimensions
- `d_i`: 5 dimensions
- combined `d = [d_e, d_i]`: 9 dimensions

The explicit dynamics parameters `d_e` are:

- object mass
- friction coefficient
- center of gravity x position
- center of gravity y position

The implicit dynamics parameters `d_i` are learned as parametric biases and are intended to represent factors that are hard to quantify directly, especially object shape and unmodeled effects.

## Important Ablation Detail

The paper also evaluates `EXI w/o d_e`, where explicit parameters are removed.
In that ablation, `d_i` is made 9-dimensional to replace the four missing explicit parameters.

So the dimensions are:

- standard `EXI`: `d_e = 4`, `d_i = 5`
- `EXI w/o d_e` ablation: `d_i = 9`

This distinction matters because the paper argues that separating explicit and implicit dynamics is beneficial: learning all dynamics as implicit parametric biases from scratch becomes harder as the number of parameters increases.

## Network Architecture

EXI-Net's predictive model is not a large or special architecture.
In the reported experiments, the predictive model is a plain fully connected MLP:

- 3 hidden layers
- 100 fully connected units per hidden layer
- ReLU activations
- mean squared error loss
- Adam optimizer
- 100 epochs
- batch size 128

The recurrent baselines use 100 LSTM units.
The paper reports that adding recurrent memory to EXI gave little positive impact, while explicit/implicit dynamics conditioning outperformed the recurrent model without dynamics parameters.

For the object-pushing setup, the general model is:

`s_{t+1} = f(s_t, a_t, d_e, d_i)`

But because the state is represented in the object-centered frame, the practical simplified model is closer to:

`Delta s_{t+1} = f(a_t, d_e, d_i)`

So the learned dynamics parameters are not competing with a very high-dimensional observation encoder.
They are appended directly to a low-dimensional MLP input and are needed to explain different outcomes under the same or similar pushing actions.

## Relation to This Project

For the current force-estimation/material-latent direction, EXI-Net supports the idea of keeping a compact latent that is optimized online at deployment.
A close analogue would be:

`force_t = estimator(image_t, proprio_t, action_t, d_i)`

or, if explicit physical parameters are available:

`force_t = estimator(image_t, proprio_t, action_t, d_e, d_i)`

The EXI-Net reference point suggests that the implicit latent does not need to be large.
In their object-pushing setup, the implicit part is only 5 dimensions when explicit physical parameters are separated out.

## Sources

- [EXI-Net: EXplicitly/Implicitly Conditioned Network for Multiple Environment Sim-to-Real Transfer](https://proceedings.mlr.press/v155/murooka21a/murooka21a.pdf)
- [OMRON TECHNICS article on EXI-Net](https://www.omron.com/jp/ja/technology/omrontechnics/2021/20211119-hamaya.html)
- [Research Theme 1: Force Estimation and Material Latent for Imitation Learning](research_theme_1_force_estimation_material_imitation.md)
