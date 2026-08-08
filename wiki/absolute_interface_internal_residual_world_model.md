# Absolute Interface With an Internal Residual World Model

## Question

Should the future object state or image latent be predicted as an absolute value or as a change from the latest observation when the main research goal is to learn a compact hidden-physics representation `z_phys`?

## Recommendation

Use one public formulation:

```text
inputs  = past observations, robot state, future actions, z_phys
outputs = absolute future observations, absolute future wrench
```

Parameterize only the observation/state prediction head internally as a residual:

```text
r_k = residual_head(...)
z_hat[t+k] = z[t] + r_k
loss_state = MSE(z_hat[t+k], z_target[t+k])
```

The dataset and the returned prediction should remain absolute. The residual is an internal coordinate choice, not a second data format or a second public model mode.

For a future VAE-based model, use a fixed encoder and its deterministic posterior mean:

```text
z[t] = mu_encoder(I[t])
```

Do not form the dynamics target from independently sampled VAE latents.

## Why This Is Not a Compromise Between Incompatible Objectives

With a fixed latest observation `z[t]`, absolute-space MSE on a residual output is exactly equal to delta-space MSE:

```text
||z[t] + r_k - z[t+k]||^2
=
||r_k - (z[t+k] - z[t])||^2
```

Thus, keeping absolute targets does not remove the optimization benefit of residual prediction. It only moves subtraction and addition inside the model boundary.

The skip path supplies the identity component exactly, while the learned head models only action- and physics-dependent change. This is an architectural inductive bias. It does not require the residual vector to have a human-interpretable physical meaning.

The two latent variables must be named and interpreted separately:

- `z_phys`: the persistent representation intended to encode hidden object properties;
- `r_img` or `r_state`: a horizon-specific prediction residual used only to reconstruct an absolute future observation.

No claim should be made that individual dimensions of `r_img` are physical quantities.

## Evidence From the Local Literature

| Source | Prediction parameterization | Relevant lesson |
|---|---|---|
| Pedestrian Trajectory Prediction with CNNs | Compares raw absolute coordinates, first-observation-relative coordinates, latest-observation-relative coordinates, and consecutive relative displacements using both Conv1D and LSTM predictors. The latest-observation origin gives the lowest average displacement error for both architectures. | This is a controlled precedent for predicting every future horizon relative to one latest observation. Its arbitrary multi-scene coordinate origins and lack of actions, wrench, image latents, or hidden physics limit direct transfer. |
| How to Select and Use Tools? | A separately trained convolutional AutoEncoder produces 15-dimensional image features, and an MTRNN predicts the next absolute multimodal sample including image feature, force, tactile, motor, and grip state. Its slow-context initial value is optimized from prediction error. | A fixed deterministic image feature, absolute next-state semantics, multimodal force prediction, and a compact adaptable dynamics variable can coexist. This does not compare direct absolute and residual heads. |
| HADYNET | Predicts the next absolute multimodal sensor state, including contact force, while parametric bias represents grasped-object and hand-dynamics differences. Some temporal differences appear in the input, not in the output target. | Input parameterization, output parameterization, and compact dynamics adaptation can be chosen independently. |
| EXI-Net | Introduces the general absolute model `s[t+1] = f(s[t], a[t], d_e, d_i)`, then changes to `Delta s[t+1] = f(a[t], d_e, d_i)` after moving the experiment into an object-centered frame. | The parameterization follows the chosen state coordinates; it is not a method-level commitment. |
| EDO-Net | Conditions graph dynamics on a learned physical-property latent `z_i` and predicts graph displacement `delta G`. | A displacement target is directly compatible with learning a hidden-physics representation. |
| IIDA | Predicts the absolute next state from `(s, a, z_e)` using reconstruction loss. | Absolute next-state prediction can also learn a compact environment latent; absolute prediction is not inherently incompatible with identification. |
| DyWA | Predicts the next task state, including absolute translation and a structured rotation representation, while conditioning internal computation on inferred dynamics. | A task-centric structured absolute target avoids spending capacity on raw visual detail. This is not evidence about raw VAE-latent differences. |
| DPMPB | Predicts the next absolute sensor state with MSE. Images are compressed by a convolutional AutoEncoder trained in advance, and the resulting latent is used as a sensor state; only the compact PB is updated online. | This is the closest local precedent for fixed deterministic image features plus absolute predictive modeling and latent-only adaptation. |
| DSSP | Predicts the next learned observation representation from history context and action, using a stop-gradient future target and cosine loss. | This is a modern precedent for absolute next-latent supervision and suggests a target-encoder/stop-gradient option if the visual encoder is later updated. It does not compare against a residual skip. |
| Adaptive Wiping | Uses absolute base-frame `(x, y)` trajectories and next-step vertical displacement `Delta h` in different branches of one method. | A clean system need not force every output quantity into the same parameterization. |
| AdaWorldPolicy and Unified World Models | Model future observations/VAE tokens as future generative targets. Internally they learn flow-matching vector fields or diffusion noise rather than a direct absolute value or the temporal latent difference. | Absolute future semantics are natural when the final world model becomes stochastic or multimodal. These works do not answer the direct-absolute versus internal-residual regression comparison. |

The local sources therefore do not support a universal choice between Absolute and Delta. They support choosing an internal parameterization appropriate to the representation and task while keeping downstream semantics explicit.

No local paper provides a controlled comparison of direct-absolute and internal-residual prediction on the same fixed VAE features. The current project's state-based result is therefore the most direct evidence for its own architecture.

A complete audit of all 34 distinct top-level research works and the four bundled code repositories is recorded in [Comprehensive Raw Audit: Absolute Targets and Internal Residual Prediction](comprehensive_raw_audit_absolute_vs_residual_prediction.md).

## What Changes With a VAE

### Fixed posterior mean

For a fixed encoder, `mu(I[t+k]) - mu(I[t])` is a valid vector in one stable Euclidean coordinate system. It need not be human-interpretable. The decoder and downstream modules consume the reconstructed absolute latent `mu(I[t]) + r_k`, not the residual alone.

The absolute latent is itself non-identifiable up to changes of latent coordinates, so lack of physical meaning is not a problem unique to Delta prediction.

### Independently sampled latents

If

```text
z[t] ~ N(mu[t], sigma[t]^2)
z[t+k] ~ N(mu[t+k], sigma[t+k]^2)
```

are sampled independently, the residual noise has variance

```text
Var(z[t+k] - z[t]) = sigma[t+k]^2 + sigma[t]^2.
```

Even two encodings of an unchanged image can therefore produce a noisy nonzero residual. This is avoidable target noise, not useful dynamics uncertainty. Use posterior means for dynamics prediction. If stochastic future prediction is later required, model uncertainty in the future distribution explicitly rather than injecting it through independent encoder samples.

### Updating the encoder

Jointly updating the encoder makes both absolute and residual latent targets move. The residual skip additionally couples the current encoding directly to the prediction. A latent-only dynamics loss may then admit undesirable shortcuts or representation collapse.

The simplest first system should:

1. train the image AutoEncoder or VAE separately;
2. freeze it;
3. use posterior means;
4. train the residual world model and `z_phys`;
5. store the encoder and normalization statistics as part of the checkpoint contract.

End-to-end encoder fine-tuning should be a later experiment with an explicit reconstruction or perceptual constraint and, if needed, a frozen or slowly updated target encoder.

## Important Boundaries

### Keep wrench absolute

The residual skip is naturally justified for a state or image feature that should persist when nothing happens. Future wrench does not generally equal the latest wrench plus a small correction, especially across contact transitions. Keep the wrench head and wrench loss absolute unless a separate experiment demonstrates a better physically grounded parameterization.

### Use manifold-aware pose residuals

For current pose-based experiments, translation can use addition. Rotation should use a valid group operation or a representation with an explicit projection:

```text
q_hat[t+k] = delta_q[k] * q[t]
```

for quaternions, rather than component-wise quaternion addition. A VAE latent is Euclidean and can use ordinary addition.

### Avoid unintended rollout accumulation

For direct multi-horizon prediction, predict every residual relative to the same latest observation:

```text
z_hat[t+k] = z[t] + r_k,  k = 1, ..., H
```

This does not accumulate residual errors across horizons. Recursive integration of one-step residuals is a different model and should only be used intentionally.

## Evaluation Must Follow the Research Goal

Lower state or image prediction error alone does not establish that `z_phys` learned object physics. Compare the direct-absolute and residual parameterizations using:

- identical inputs, normalization, model width, data splits, and optimization budget;
- a latest-observation persistence baseline and improvement over that baseline;
- absolute future-state/image error and future-wrench error;
- separate results for windows with meaningful motion, contact, and contact transitions so static windows do not dominate;
- held-out-object adaptation by optimizing or inferring only `z_phys`;
- `z_phys` zero, shuffle, and wrong-object ablations;
- output sensitivity for fixed observation/action and different `z_phys`;
- same-appearance, same-action examples where only mass, center of mass, inertia, friction, or compliance changes;
- clustering or probing of `z_phys` against known physical parameters, treated as analysis rather than direct supervision.

The mainline implementation can remain residual-only internally. If a direct-absolute ablation is needed for a paper, it can be maintained as a short-lived experimental branch while preserving the same absolute external interface.

## Decision

Do not return to a direct-absolute head merely to prepare for a future VAE. Keep:

```text
absolute data -> internal residual head -> absolute prediction -> absolute loss
```

This preserves the empirically useful optimization bias, removes duplicate dataset and evaluation semantics, and remains valid after migration to fixed posterior-mean image features.

Reconsider this decision only if the future objective changes from deterministic MSE prediction to genuinely multimodal future generation. In that case, a probabilistic absolute future model such as video/latent diffusion is a different architectural regime, not an argument against the current residual parameterization.

## Sources

- [EXI-Net: EXplicitly/Implicitly Conditioned Network for Multiple Environment Sim-to-Real Transfer](../raw/EXI-Net%20EXplicitly%20Implicitly%20Conditioned%20Network%20for%20Multiple%20Environment%20Sim-to-Real%20Transfer.pdf)
- [EDO-Net: Learning Elastic Properties of Deformable Objects from Graph Dynamics](../raw/EDO-Net%20Learning%20Elastic%20Properties%20of%20Deformable%20Objects%20from%20Graph%20Dynamics.pdf)
- [Context is Everything: Implicit Identification for Dynamics Adaptation](../raw/Context%20is%20Everything%20Implicit%20Identification%20for%20Dynamics%20Adaptation.pdf)
- [DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation](../raw/DyWA%20Dynamics-adaptive%20World%20Action%20Model%20for%20Generalizable%20Non-prehensile%20Manipulation.pdf)
- [Deep Predictive Model Learning with Parametric Bias Handling Modeling Difficulties and Temporal Model Changes](../raw/Deep%20Predictive%20Model%20Learning%20with%20Parametric%20Bias%20Handling%20Modeling%20Difficulties%20and%20Temporal%20Model%20Changes.pdf)
- [Adaptive Wiping: Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations](../raw/Adaptive%20Wiping%20Adaptive%20contact-rich%20manipulation%20through%20few-shot%20imitation%20learning%20with%20Force-Torque%20feedback%20and%20pre-trained%20object%20representations.md)
- [AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation](../raw/AdaWorldPolicy%20World-Model-Driven%20Diffusion%20Policy%20with%20Online%20Adaptive%20Learning%20for%20Robotic%20Manipulation.md)
- [Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets](../raw/Unified%20World%20Models%20Coupling%20Video%20and%20Action%20Diffusion%20for%20Pretraining%20on%20Large%20Robotic%20Datasets.md)
- [A Novel Framework for Learning Stochastic Representations for Sequence Generation and Recognition](../raw/A%20Novel%20Framework%20for%20Learning%20Stochastic%20Representations%20for%20Sequence%20Generation%20and%20Recognition.pdf)
- [How to Select and Use Tools? Active Perception of Target Objects Using Multimodal Deep Learning](../raw/2106.02445v1.pdf)
- [Object Recognition, Dynamic Contact Simulation, Detection, and Control of the Flexible Musculoskeletal Hand Using a Recurrent Neural Network with Parametric Bias](../raw/Object%20Recognition,%20Dynamic%20Contact%20Simulation,%20Detection,%20and%20Control%20of%20the%20Flexible%20Musculoskeletal%20Hand%20Using%20a%20Recurrent%20Neural%20Network%20with%20Parametric%20Bias.md)
- [DSSP: Diffusion State Space Policy with Full-History Encoding](../raw/DSSP%20Diffusion%20State%20Space%20Policy%20with%20Full-History%20Encoding.md)
- [Pedestrian Trajectory Prediction with Convolutional Neural Networks](<../raw/Pedestrian Trajectory Prediction with Convolutional Neural Networks.pdf>)
