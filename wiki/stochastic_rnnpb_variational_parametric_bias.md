# Stochastic RNNPB and Variational Parametric Bias

## Bibliographic Information

- Title: *A Novel Framework for Learning Stochastic Representations for Sequence Generation and Recognition*
- Authors: Jungsik Hwang and Ahmadreza Ahmadi
- arXiv: `2501.00076`
- Submitted: 2024-12-30
- Length: 14 pages
- Local source: [PDF](../raw/A%20Novel%20Framework%20for%20Learning%20Stochastic%20Representations%20for%20Sequence%20Generation%20and%20Recognition.pdf)
- Official source: [arXiv](https://arxiv.org/abs/2501.00076)

## Core Method

The paper extends deterministic Recurrent Neural Networks with Parametric Biases (RNNPB) by representing each sequence-level PB as a diagonal Gaussian:

```text
q(PB_i) = N(mu_i, diag(sigma_i^2))
PB_i = mu_i + sigma_i * epsilon
epsilon ~ N(0, I)
```

The sampled PB is held constant across a generated sequence and is supplied to the LSTM at every time step. It therefore represents uncertainty in a sequence-level internal belief, rather than independent output noise at each step.

Training jointly optimizes the network weights `theta` and the per-sequence distribution parameters `mu_i` and `sigma_i`. The objective is:

```text
L_train = L_reconstruction + beta * KL(q(PB_i) || N(0, I))
```

The reconstruction term is implemented as mean squared error. The KL term regularizes the PB distribution toward a unit Gaussian. The paper evaluates four-dimensional PBs with a 256-unit LSTM on 72 Pepper robot motion sequences.

## Recognition of a Novel Sequence

At recognition time, the network weights `theta` are frozen. Only `mu` and `sigma` are iteratively updated to reconstruct the observed part of a novel sequence:

```text
mu, sigma <- argmin L_reconstruction
theta remains fixed
```

This is called recognition through prediction-error minimization or recognition by reconstruction. The paper deliberately omits the KL term during this recognition optimization, even though the training objective includes it.

This detail is important for interpreting the method:

- the paper provides a precedent for adapting only a compact probabilistic PB while retaining long-term knowledge in the fixed network;
- it does not establish that KL regularization should also be used during deployment-time PB adaptation.

## Reported Findings

- Stochastic PBs form a smoother latent landscape than deterministic point-estimate PBs.
- The stochastic model is less sensitive to PB initialization when recognizing novel sequences.
- `sigma` changes during recognition and generally narrows as the model becomes more confident in a PB explanation.
- Larger `beta` produces stronger regularization and smoother representations but reduces reconstruction fidelity.
- Setting `beta = 0` makes learned PB variances nearly deterministic and performs worse than using a positive KL weight, although it still outperforms the explicitly deterministic model in the reported tasks.

## Naming Status

The paper calls the overall method **stochastic RNNPB** and descriptively refers to a **stochastic PB** or **stochastic PB layer**. It does not define `Variational Parametric Bias` or the acronym `VPB`.

For this project, the following is a clear project-defined term:

> **Variational Parametric Bias (VPB): a PB represented by a variational distribution and trained using reparameterized sampling and KL regularization.**

This label preserves continuity with Parametric Bias while making the probabilistic construction explicit. It must be introduced as terminology defined by this project, not as the name used by Hwang and Ahmadi.

Possible distinctions are:

- `VPB`: reparameterization plus a variational objective containing KL regularization;
- `PPB`: a generic probabilistic PB without committing to a variational objective;
- `SPB`: a generic stochastic PB, closest to the wording of the source paper.

`VPB` is the most precise name for the proposed implementation if it actually uses both reparameterization and KL regularization.

## Relation to the Material-Latent Project

The architectural role can be generalized conceptually from a sequence-pattern PB to a material or hidden-physics PB:

```text
q(p_material) = N(mu_material, diag(sigma_material^2))
world model or policy = f(observation, action, p_material)
```

A direct adaptation of the paper's inference pattern would be:

1. train the shared force estimator or world model together with a distributional material PB;
2. freeze the shared model at deployment;
3. update only `mu_material` and `sigma_material` from measured-versus-estimated force error;
4. condition the world model or policy on a sample or deterministic statistic of the adapted distribution.

This provides a stronger precedent than merely attaching uncertainty to a latent: it demonstrates iterative posterior-like optimization of only the PB distribution while the shared network remains fixed.

## Scope and Caveats

- The source method is evaluated on motion-sequence identity, not mass, stiffness, friction, center of mass, or other physical properties.
- PB is integrated with an LSTM-based RNNPB. Applying the same construction to a Transformer, diffusion policy, or world model is a project extension, not a result established by the paper.
- Each training sequence has its own learnable `mu_i` and `sigma_i`; the paper does not train an encoder that directly infers PB from a new sequence.
- Recognition uses reconstruction loss only. Updating a material VPB using force-estimation error, or retaining KL during online adaptation, requires separate validation.
- A distributional PB does not by itself prevent latent collapse. The dataset and prediction target must still make hidden physics necessary.

