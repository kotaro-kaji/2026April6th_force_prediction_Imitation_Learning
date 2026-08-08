# Comprehensive Raw Audit: Absolute Targets and Internal Residual Prediction

## Scope

This audit covers every top-level file in `raw/` and the four source-code repositories stored below it.

- There are 40 top-level files.
- `UMI-FT` and `MTIL` each have an exact duplicate.
- `DPMPB` is stored in both PDF and Markdown form.
- The two facial-prediction PDFs describe the same research project.
- `STLファイル編集方法.md` is a technical note rather than a research paper.

After consolidating these cases, the top-level corpus contains 35 distinct research works. The code audit additionally covers `dynamical-metalearning`, `RoboMorph`, `sysid-transformers`, and `DyWA`.

The audit asks three separate questions for each work:

1. What is the semantic prediction target: a future observation/state, an action, a physical parameter, or something else?
2. If a future state is predicted, is it represented as an absolute value or a temporal change?
3. Is the internal optimization target a direct regression value, diffusion noise, or a flow-matching vector field?

These levels must not be collapsed into one `Absolute` versus `Delta` label.

## Main Finding

The corpus does not establish a literature consensus that direct absolute regression is preferable to residual regression, or vice versa.

It instead supports the following separation:

```text
semantic contract:
    predict an absolute future observation and absolute future wrench

deterministic observation head:
    predict a residual relative to the latest observation

reconstruction:
    predicted_absolute = latest_observation + residual

supervision and evaluation:
    compare predicted_absolute with target_absolute
```

This is not an awkward compromise. It separates the meaning of the model output from the numerical parameterization used to learn it.

AdaWorldPolicy and Unified World Models strengthen the case for an absolute *future-outcome contract*. They do not show that a deterministic network should directly regress the absolute latent. Their internal targets are flow-matching vector fields or diffusion noise, not the temporal difference between current and future image latents.

The project's own state-based experiment, where Delta improved translation and rotation prediction, remains the most directly task-matched evidence. The pedestrian-trajectory paper now in `raw/` supplies an independent controlled coordinate comparison: for both a Conv1D and an LSTM, representing all future positions relative to the latest observation gives the lowest average trajectory error among raw absolute coordinates, first-observation-relative coordinates, latest-observation-relative coordinates, and consecutive displacements.

It is not a complete substitute for the required controlled comparison between:

```text
direct absolute regression
vs.
latest-latent skip + learned residual
```

on the same fixed image latent representation.

## AdaWorldPolicy's Exact Relevance

AdaWorldPolicy is one of the closest references to the intended final system, but it answers a different architectural question from the current Absolute-versus-Delta decision.

| Axis | AdaWorldPolicy | Current research direction |
|---|---|---|
| Current observations | Multi-view image history and force/torque history | State history now; image latent, robot state, and wrench later |
| Action conditioning | Known executed action in Future Imagination mode | Given future action command sequence |
| Predicted outcomes | Future visual observations and future force/torque | Future object/image state and future wrench |
| Semantic future target | Absolute future outcome | Absolute future outcome |
| Internal training target | Flow-matching vector field for a noised future observation | Deterministic residual for the observation branch |
| Persistent object-physics representation | None is explicitly maintained | Explicit low-dimensional `z_phys` |
| Online adaptation variable | LoRA matrices across the connected modules | Preferably only `z_phys` |
| Clean same-appearance/different-physics identification | Not isolated as the central evaluation | Intended central evaluation |

The appropriate conclusion is therefore:

- AdaWorldPolicy is a Tier-1 reference for the *final multimodal input/output problem*.
- It is not evidence that a direct absolute MSE head is superior to an internal residual head.
- Its absence of a persistent object-specific `z_phys` and its LoRA-based adaptation leave a clear distinction for this research.

## Strongest Direct Precedents

### Fixed visual features and compact dynamics variables

Two older predictive-robotics works are especially close to the proposed deterministic formulation.

- [How to Select and Use Tools? Active Perception of Target Objects Using Multimodal Deep Learning](<../raw/2106.02445v1.pdf>) first trains a deterministic convolutional AutoEncoder, compresses each image to 15 dimensions, and uses an MTRNN to predict the next absolute 33-dimensional multimodal sample: image feature, motor state, tactile signal, force, and gripper state. A slow-context initial state is optimized from sensory prediction error and acts as a latent for ingredient characteristics and tool-object-action relations.
- [Deep Predictive Model Learning with Parametric Bias](<../raw/Deep Predictive Model Learning with Parametric Bias Handling Modeling Difficulties and Temporal Model Changes.md>) predicts the next absolute sensor state with MSE and conditions it on a low-dimensional parametric bias. Some experiments include an image compressed by a separately trained AutoEncoder, force-related state, and a PB representing object, cloth material, shoes, or other changing dynamics.

These works show that fixed deterministic image features, absolute next-state prediction, force sensing, and latent-only adaptation can coexist. They do not compare against an internal residual head.

### Absolute next-latent prediction with a trainable encoder

[DSSP](<../raw/DSSP Diffusion State Space Policy with Full-History Encoding.md>) uses

```text
z[t+1]     = E_obs(o[t+1])
z_hat[t+1] = g(context[t], action[t])
```

and a cosine loss to `stop_gradient(z[t+1])`. This is a useful modern precedent for predicting a learned next-observation representation without requiring a physically interpretable latent difference.

Its lesson is narrower than “use Absolute”:

- future latent targets can be stabilized with stop-gradient;
- a history representation can be shaped by a dynamics-aware auxiliary loss;
- the work does not compare direct absolute prediction with a residual skip;
- its history context represents task progress, not a persistent object-physics variable.

For the first VAE experiment, freezing the encoder is still simpler and more controlled than adopting end-to-end representation learning. If later fine-tuning the encoder, DSSP motivates a stop-gradient or target-encoder design.

### Physics-conditioned Delta prediction

- [Pedestrian Trajectory Prediction with Convolutional Neural Networks](<../raw/Pedestrian Trajectory Prediction with Convolutional Neural Networks.pdf>) compares four coordinate representations with both convolutional and recurrent models. Future positions expressed relative to the latest observation give the best average displacement error. This is equivalent to predicting all horizon residuals from one latest-observation base.
- [EDO-Net](<../raw/EDO-Net Learning Elastic Properties of Deformable Objects from Graph Dynamics.pdf>) infers a latent representation of elastic properties from exploratory graph/force observations and predicts graph displacement `delta G`.
- [EXI-Net](<../raw/EXI-Net EXplicitly Implicitly Conditioned Network for Multiple Environment Sim-to-Real Transfer.pdf>) begins with the general absolute formulation `s[t+1] = f(s[t], a[t], d_e, d_i)`, then uses `Delta s[t+1] = f(a[t], d_e, d_i)` after moving to an object-centered experimental coordinate system.
- [Adaptive Wiping](<../raw/Adaptive Wiping Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations.md>) deliberately mixes absolute planar trajectories with next-step vertical displacement.

These works show that a physical-property latent does not require an absolute regression head. They also show that output parameterization should be chosen per quantity and coordinate system.

## Complete Top-Level Corpus Audit

### A. Works that directly inform future-state parameterization

| Work | Learned prediction | Absolute/Delta relevance |
|---|---|---|
| [AdaWorldPolicy](<../raw/AdaWorldPolicy World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation.md>) | Future video observations, force/torque, and actions | Absolute future outcomes; internal flow-matching vector fields, not direct absolute or temporal-delta regression |
| [Unified World Models](<../raw/Unified World Models Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets.md>) | Actions and next/future images using a frozen SDXL VAE | Absolute future-image distribution; internal diffusion-noise prediction |
| [Pedestrian Trajectory Prediction with CNNs](<../raw/Pedestrian Trajectory Prediction with Convolutional Neural Networks.pdf>) | Twelve future planar positions from eight observed positions | Controlled comparison favors expressing all future positions relative to the latest observation by average displacement error |
| [How to Select and Use Tools?](<../raw/2106.02445v1.pdf>) | Next image feature, robot state, tactile signal, force, and grip state | Direct absolute next-step multimodal prediction from fixed AutoEncoder features |
| [Deep Predictive Model with Parametric Bias](<../raw/Deep Predictive Model Learning with Parametric Bias Handling Modeling Difficulties and Temporal Model Changes.md>) | Next sensor state and, for CTM, next control | Direct absolute next-state prediction; PB captures changing body/object/environment dynamics |
| [HADYNET](<../raw/Object Recognition, Dynamic Contact Simulation, Detection, and Control of the Flexible Musculoskeletal Hand Using a Recurrent Neural Network with Parametric Bias.md>) | Next muscle length, tension, contact force, and joint angle | Absolute outputs; differences are included only as some inputs and commands |
| [EDO-Net](<../raw/EDO-Net Learning Elastic Properties of Deformable Objects from Graph Dynamics.pdf>) | Future deformable-object graph displacement conditioned on a physical latent | Direct Delta precedent |
| [EXI-Net](<../raw/EXI-Net EXplicitly Implicitly Conditioned Network for Multiple Environment Sim-to-Real Transfer.pdf>) | Next object state conditioned on explicit and implicit dynamics | General model is absolute; object-centered experiment predicts Delta |
| [IIDA](<../raw/Context is Everything Implicit Identification for Dynamics Adaptation.pdf>) | Next state conditioned on an inferred environment context | Direct absolute next-state reconstruction |
| [DyWA](<../raw/DyWA Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation.pdf>) | Joint action and next task state, with history-inferred dynamics conditioning | Absolute next translation and structured rotation; action itself is an end-effector residual |
| [DSSP](<../raw/DSSP Diffusion State Space Policy with Full-History Encoding.md>) | Next learned observation representation as an auxiliary objective | Direct absolute next-latent target with stop-gradient |
| [From System Models to Class Models](<../raw/From system models to class models An in-context learning paradigm.md>) | Future system outputs from past input/output context and future inputs | Absolute future system-output sequence |
| [Diffuser](<../raw/2205.09991v2.pdf>) | Full state-action trajectories | Absolute trajectory distribution; internal diffusion-noise prediction |
| [Ada-Diffuser](<../raw/Ada-Diffuser Latent-Aware Adaptive Diffusion for Decision-Making.pdf>) | Latent-conditioned state/action trajectories | Absolute observable trajectories plus a time-varying inferred context; not a residual comparison |
| [Facial-motion prediction](<../raw/Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers - manuscript.pdf>) | Future facial landmarks and RGB frames | Direct absolute future landmark/image regression |

### B. Works that mainly inform `z_phys`, system identification, or latent adaptation

| Work | Main contribution | Relevance |
|---|---|---|
| [Few-Shot Learning of Force-Based Motions](<../raw/2309.04640v1.pdf>) | A VAE encodes exploratory force trajectories into a physical-property representation used to generate a motion trajectory | Strong precedent for force-derived `z_phys`, a frozen encoder, and downstream use; not a future-state model |
| [Adaptive Wiping](<../raw/Adaptive Wiping Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations.md>) | A VAE encodes sponge properties from force/torque exploration | Strong object-property latent precedent; also demonstrates mixed absolute/Delta outputs |
| [Stochastic RNNPB](<../raw/A Novel Framework for Learning Stochastic Representations for Sequence Generation and Recognition.pdf>) | A sequence-level PB is sampled once and held fixed throughout autoregressive generation; only its posterior parameters are adapted during recognition | Relevant to probabilistic `z_phys` and latent-only adaptation, not to sampled image-latent differencing |
| [CAVIA](<../raw/Fast Context Adaptation via Meta-Learning.pdf>) | Only a low-dimensional context parameter is updated in the inner loop and at test time | General precedent for adapting only `z_phys`; no specific state-output parameterization |
| [Dynamics as Prompts / CAPTURE](<../raw/Dynamics as Prompts In-Context Learning for Sim-to-Real System Identifications.md>) | Predicts explicit environment parameters from interaction history for sim-to-real adaptation | Relevant contrast between explicit system identification and the proposed implicit persistent latent |

### C. Works that provide task or policy context but do not decide the state-target question

| Work | Why it does not decide Absolute versus residual state prediction |
|---|---|
| [Adaptive stiffness control under intention uncertainty](<../raw/2A2-V04.pdf>) | A CVAE predicts future robot joint trajectories for stiffness control, not future observations conditioned on hidden object physics |
| [Adaptive Compliance Policy](<../raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md>) | Predicts pose/compliance action trajectories, not the future object or image state |
| [Bi-HIL](<../raw/Bi-HIL Bilateral Control-Based Multimodal Hierarchical Imitation Learning via Subtask-Level Progress Rate and Keyframe Memory for Long-Horizon Contact-Rich Robotic Manipulation.md>) | Predicts hierarchical progress, keyframes, and control outputs rather than an action-conditioned future observation |
| [Humanoid Nursing-Care Pouring](<../raw/Development of a Humanoid Nursing Care Robot and Realization of Quantitative Pouring Operation.md>) | Uses force sensing for quantitative pouring rather than learning a latent-conditioned world model |
| [Flow with the Force Field](<../raw/Flow with the Force Field Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data.md>) | Learns a compliant action policy by flow matching; its vector field is not a temporal observation Delta |
| [ForceMimic](<../raw/ForceMimic Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation.md>) | Generates pose/wrench-related action trajectories from demonstrations rather than predicting environmental outcomes |
| [Flexible Tool Manipulation](<../raw/Imitation Learning System Design with Small Training Data for Flexible Tool Manipulation.pdf>) | Maps multimodal observations to target end-effector velocity; no hidden-physics future-state head |
| [UMI-FT](<../raw/In-the-Wild Compliant Manipulation with UMI-FT.md>) | Predicts compliant robot actions including pose, virtual target, stiffness, and grasp force; no future-observation model |
| [Industrial Dual-Arm Box Handling](<../raw/Industrial Dual-Arm Box Handling via Online Inertial Estimation and Convex Wrench Optimization.md>) | Analytically estimates mass and center of mass from wrench measurements and optimizes contact wrenches |
| [Pivoting Manipulation](<../raw/Learning Pivoting Manipulation with Force and Vision Feedback Using Optimization-based Demonstrations.md>) | Learns a force/vision feedback policy from optimized demonstrations; quaternion perturbation `delta` is data augmentation, not the prediction target in question |
| [Variable Compliance Control / Comp-ACT](<../raw/Learning Variable Compliance Control From a Few Demonstrations for Bimanual Robot with Haptic Feedback Teleoperation System.pdf>) | Predicts future Cartesian pose and stiffness action chunks |
| [High-Quality Robotic Wiping](<../raw/Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum.md>) | Uses reinforcement learning, reward analysis, and curriculum adaptation rather than predictive state modeling |
| [MTIL](<../raw/MTIL Encoding Full History with Mamba for Temporal Imitation Learning.md>) | Encodes full history to predict action chunks; its recurrent memory is relevant to persistence but is not an explicit object-physics latent or future-state predictor |
| [Tactile Pouring thesis](<../raw/R014469.pdf>) | Estimates or controls poured amount from tactile/force information without the proposed learned world-model interface |
| [SCCRUB](<../raw/SCCRUB Surface Cleaning Compliant Robot Utilizing Bristles.pdf>) | Learns a static inverse-kinematics/elasticity mapping from desired pose and tendon force to tendon lengths, not temporal future dynamics |

## Code Repository Audit

The code agrees with the papers and does not reveal a hidden temporal-Delta convention.

| Repository | Checked behavior |
|---|---|
| [`sysid-transformers`](../raw/sysid-transformers/) | Encoder-decoder and one-step models regress future absolute system outputs. |
| [`RoboMorph`](../raw/RoboMorph/) | The Transformer returns `y_new_sim`; training compares it directly with `batch_y_new` by MSE, MAE, or Huber loss. There is no subtraction of the last context output. |
| [`dynamical-metalearning`](../raw/dynamical-metalearning/) | RoboMorph predicts the absolute future response. Its diffusion variants model or condition the absolute action-state trajectory while internally predicting noise or `x0`. |
| [`DyWA`](../raw/DyWA/) | The released implementation corresponds to joint action/absolute next-task-state prediction. Uses of “delta” elsewhere include action residuals, geometric errors, and control updates, not a future-state regression target. |

The preprint stored inside `dynamical-metalearning` is particularly useful for distinguishing semantic outcomes from diffusion parameterization. It writes the deterministic model as

```text
y_hat[m:N] = M(u[m:N], context)
```

and trains it against the absolute future response, while the diffusion versions estimate noise on absolute trajectories.

## Implications for the VAE Transition

### A latent difference does not need human physical meaning

For a fixed encoder, both

```text
z_future
```

and

```text
z_future - z_current
```

are coordinate-dependent quantities. Ada-Diffuser's identifiability result also emphasizes that a learned latent may only be identifiable up to an invertible transformation. Absolute latent coordinates therefore do not possess a privileged physical meaning that their differences lack.

The engineering requirement is not semantic interpretability of the residual. It is coordinate stability:

- use one fixed encoder;
- use the deterministic posterior mean;
- preserve encoder and normalization versions in the checkpoint contract;
- do not compute targets from independent posterior samples.

### Encoder updates affect both formulations

If the encoder changes, absolute targets also move. Delta is not uniquely unstable; it exposes the movement through both endpoints.

The first image-based experiment should therefore freeze the encoder. If end-to-end fine-tuning is later necessary:

- use a stop-gradient future target or a slowly updated target encoder;
- retain reconstruction/perceptual constraints;
- examine whether gradients through the current-latent skip encourage an overly static representation;
- do not infer from DSSP that a trainable residual skip is automatically stable, because DSSP predicts the next latent directly.

### Generative video prediction is a later architectural regime

If the project eventually replaces deterministic MSE prediction with a video diffusion or flow-matching model, the temporal residual head may disappear. The public contract still remains:

```text
current observations + actions + physics context
    -> distribution over absolute future observations and wrench
```

This future possibility is a reason to keep the external interface absolute. It is not a reason to discard the currently effective residual parameterization.

## Central Caution: Residual Prediction Does Not Guarantee `z_phys`

The residual head removes the need for the learned branch to reproduce unchanged state components. However, it also makes the persistence solution exact and easy:

```text
residual = 0
prediction = latest_observation
```

A direct-absolute network can also learn this shortcut by copying its input, so this is not a unique failure of residual prediction. The skip connection merely makes the shortcut explicit.

This matters when most training windows contain little motion or weak contact response. A low average loss can then be achieved without using the action or `z_phys`. The architecture decision and the representation-learning claim must therefore be evaluated separately.

At minimum:

- include a latest-observation persistence baseline;
- report improvement over persistence, not only raw prediction error;
- sample or report separately on windows with meaningful action-induced motion and contact transitions;
- compare correct `z_phys` with shuffled or wrong-object `z_phys` under the same observation and action;
- verify that the wrench branch also changes appropriately with `z_phys`.

A useful descriptive score is

```text
prediction_skill = 1 - model_MSE / persistence_MSE
```

when `persistence_MSE` is nonzero. This score should be reported by horizon and interaction regime rather than used as a replacement training loss in the first implementation.

## Recommended Implementation Contract

Use one non-configurable mainline semantic interface:

```text
state_or_image_prediction = absolute
wrench_prediction         = absolute
```

Inside the state/image branch:

```text
base = latest_observation
residual[k] = observation_head(features, horizon=k)
prediction[k] = base + residual[k]
loss = loss_fn(prediction[k], absolute_target[k])
```

For direct multi-horizon prediction, every horizon should use the same `base`. Do not recursively add horizon residuals unless autoregressive rollout is intentionally being studied.

Do not retain a general user-facing `absolute/delta` flag. If a direct-absolute ablation is required for publication, keep it as a narrowly scoped experimental implementation or branch. The production checkpoint should record an architecture/version identifier, not expose two competing data semantics.

Keep the wrench branch absolute. Unlike object/image state, wrench does not generally persist through contact transitions.

For pose-state experiments, use group-aware rotation composition rather than component-wise addition:

```text
q_prediction[k] = delta_q[k] * q_latest
```

## Minimum Decisive Experiments

The literature cannot replace the following controlled tests.

1. **Persistence baseline**

   Compare against `prediction[k] = latest_observation`. This quantifies how much of direct-absolute accuracy comes from copying static content.

2. **Direct absolute versus internal residual**

   Hold inputs, width, training budget, normalization, loss on reconstructed absolute predictions, and data split fixed. Change only the skip parameterization.

3. **Fixed VAE posterior mean**

   Repeat the comparison after replacing state inputs with `mu_encoder(image)`, with the encoder frozen.

4. **Physics-latent dependence**

   Evaluate correct, shuffled, wrong-object, zeroed, and adapted `z_phys`. A lower image/state error is insufficient if the model ignores `z_phys`.

5. **Same appearance, different physics**

   Keep geometry and visual appearance matched while changing mass, center of mass, inertia, friction, or compliance. This is the cleanest distinction from AdaWorldPolicy's broader online adaptation setting.

6. **Horizon and contact breakdown**

   Report errors by prediction horizon and by free-space/contact/contact-transition segments. Residual state prediction and absolute wrench prediction may have different failure patterns.

## Decision

The raw corpus strengthens, rather than weakens, the current design:

```text
absolute external target and return value
+ internal residual state/image head
+ absolute wrench head
+ fixed deterministic image encoder initially
+ explicit persistent z_phys
```

AdaWorldPolicy should be treated as a central final-system reference. The pedestrian-trajectory paper is the most direct independent comparison of absolute and latest-observation-relative coordinate choices. DPMPB, HADYNET, and the active-perception MTRNN are closer references for compact dynamics-variable adaptation with multimodal prediction. EDO-Net and EXI-Net are the clearest evidence that Delta dynamics and hidden-physics representations are compatible. DSSP is the most useful local precedent for stabilizing learned next-state targets when representation learning is eventually made trainable.

None of them overturns the project's own empirical result in favor of direct absolute regression.
