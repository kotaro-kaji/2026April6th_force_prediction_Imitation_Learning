# CAVIA: Fast Context Adaptation via Meta-Learning

## Summary

**CAVIA** stands for **Fast Context Adaptation via Meta-Learning**.
It is a meta-learning method by Zintgraf et al., published at ICML 2019 after an arXiv version in 2018.
It is closely related to MAML, but instead of adapting all network weights for each new task, it adapts a small set of **context parameters** that are given as additional input to the network.

The basic split is:

- shared parameters: meta-trained across tasks and kept fixed during task adaptation,
- context parameters: task-specific low-dimensional variables updated by gradient descent.

At test time, only the context parameters are updated.
This makes CAVIA a useful precedent for "freeze the main network, adapt only a compact task/environment latent."

## Relation to MAML

MAML was introduced in 2017 and adapts the model parameters themselves from a learned initialization.
CAVIA keeps the MAML-like inner-loop adaptation idea, but restricts adaptation to context variables.

This is important because:

- it reduces overfitting risk compared with updating all weights,
- it is easier to interpret as task/context identification,
- it aligns well with low-dimensional latent adaptation,
- it is closer to PB-style adaptation than full model fine-tuning.

So if the remembered date was "around 2017," that likely refers to MAML.
CAVIA itself is better dated as arXiv 2018 / ICML 2019.

## Relevance to This Project

CAVIA is highly relevant to the current force-estimation/material-latent direction.
The project's desired deployment logic is:

`freeze estimator and policy weights, update only d_i from measured-vs-estimated force error`

This is structurally similar to CAVIA:

`freeze shared network weights, update only context parameters from task-specific loss`

The main difference is the task loss.
CAVIA is a general meta-learning algorithm, while this project would use force-estimation error as the adaptation signal.

## Why It Matters

CAVIA gives a clean prior-art language for the current method:

- `d_i` can be described as a context parameter or task/environment latent,
- online adaptation can be framed as inner-loop optimization over context only,
- the main estimator/policy weights can remain fixed,
- the method avoids the stronger and messier claim of full test-time network adaptation.

This is especially useful for distinguishing the project from AdaWorldPolicy-style LoRA or parameter-efficient online updates.
The claim becomes latent-only adaptation rather than network-parameter adaptation.

## Caution

CAVIA does not solve the "latent is ignored" problem by itself.
If the model can minimize force-estimation loss without using `d_i`, the context parameters may still have little effect.
The data and loss must make hidden physical/material properties necessary for accurate estimation or downstream control.

## Sources

- Zintgraf et al., [Fast Context Adaptation via Meta-Learning](https://proceedings.mlr.press/v97/zintgraf19a.html), ICML 2019.
- Finn et al., [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html), ICML 2017.
- [Research Theme 1: Force Estimation and Material Latent for Imitation Learning](research_theme_1_force_estimation_material_imitation.md)
- [Preventing Hidden-Context Latents From Being Ignored](preventing_hidden_context_latent_ignored.md)
