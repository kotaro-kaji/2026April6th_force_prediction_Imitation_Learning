# Preventing Hidden-Context Latents From Being Ignored

## Question

When a force estimator, force predictor, or dynamics model receives a PB-like variable, environment latent, or context embedding, why might changing that quantity have almost no effect on inference, and what do prior methods do to avoid that failure mode?

## Short Answer

This can happen when the model has an easier path to minimize the loss without using the hidden-context input.
For example, if recent state, action, measured force history, or visual features already explain the target well enough on average, a network can treat the PB/context channel as redundant.

The relevant prior-work pattern is not just "add a context vector as another input."
The stronger pattern is:

- make the prediction target depend on hidden dynamics across controlled environment variation,
- infer a compact latent from context transitions or history,
- inject the latent in a way that changes intermediate computation, often through conditioning or FiLM,
- train with losses or ablations that reveal whether the latent is actually used.

## Local Prior References

### EXI-Net

EXI-Net conditions a dynamics predictor on explicit dynamics parameters `d_e` and implicit parametric-bias variables `d_i`.
In the standard pushing setup, `d_e` is 4-dimensional and `d_i` is 5-dimensional.
The important lesson is the explicit separation:

`s_{t+1} = f(s_t, a_t, d_e, d_i)`

For this project, the analogous design is to keep a compact hidden-material latent rather than letting all physical variation be absorbed by the main observation/action pathway.

### IIDA

IIDA infers a compact environment latent `z_e in R^8` from context transition tuples `(s, a, s')`, then conditions the dynamics predictor on that latent.
The context size is the number of transition examples, while the final latent remains small.

The useful lesson is that the latent is supervised indirectly by the dynamics-prediction task, but the data organization makes the same predictor face multiple environments whose differences must be explained by context.

### DyWA

DyWA uses a history-conditioned adaptation embedding and FiLM conditioning.
Its implementation uses a 128-dimensional history adaptation representation and injects it through FiLM blocks rather than only concatenating it to the input.

The useful lesson is that hidden dynamics can be made more influential by using the inferred embedding to modulate internal features.
Its ablation also suggests that world modeling, dynamics adaptation, and FiLM conditioning work best together.

## Practical Failure Modes

- The dataset does not contain controlled pairs where the same visible state/action requires different predictions because of hidden physics.
- The estimation/prediction target can be fit from local kinematics or recent force without needing a persistent latent.
- The PB/context vector is concatenated only at the input, and later layers learn to route around it.
- The latent dimension is too large and becomes noisy, or too weakly regularized to become an environment-level representation.
- Direct measured force history gives the model or policy an easier shortcut than using a material/context latent.
- The training objective never penalizes latent-invariant behavior.

## Useful Checks

- Compare force estimates or predictions with the latent fixed, randomized, zeroed, and inferred.
- Measure prediction sensitivity: `||f(x, z_1) - f(x, z_2)||` for the same state/action and different known environments.
- Train or evaluate on same-geometry, same-appearance, different-mass/friction/CoM cases.
- Check performance as a function of context length.
- Check whether inferred latents cluster by physical property.
- Add an ablation where recent measured force history is removed or shortened, so it cannot fully replace the latent.

## Recommended Design Adjustments

- Prefer a compact latent such as 5 to 32 dimensions for the explicit hidden-physics variable.
- Use multi-step rollout loss in addition to one-step prediction, so the latent must carry persistent dynamics information.
- Inject the latent through FiLM, gating, or layer-wise conditioning, not only by appending it once to the raw input.
- Use controlled data splits where hidden physics is the only explanation for different outcomes.
- For the current force-estimation direction, use measured-vs-estimated force error to update only the compact latent while keeping model weights fixed.

## Sources

- [EXI-Net Explicit and Implicit Dynamics Parameters](exi_net_explicit_implicit_dynamics_parameters.md)
- [IIDA: Context Is Everything for Implicit Identification](iida_context_is_everything_implicit_identification.md)
- [DyWA: Dynamics-adaptive World Action Model](dywa_dynamics_adaptive_world_action_model_for_generalizable_non_prehensile_manipulation.md)
- [DyWA vs Explicit Hidden-Physics Latent](dywa_vs_hidden_physics_latent_controlled_dynamics_gap.md)
- [Force Estimation and Latent Adaptation Prior Art](force_estimation_latent_adaptation_prior_art.md)
