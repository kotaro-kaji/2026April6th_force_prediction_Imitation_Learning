# Wiki Index

## Research Themes

- [Research Theme 1: Force Prediction and Material Latent for Imitation Learning](research_theme_1_force_prediction_material_imitation.md): Force prediction with an object-specific latent for adapting imitation policies to unseen physical properties.
- [Overlap With ACP, UMI-FT, CompACT, and ForceMimic](overlap_with_adaptive_compliance_policy_umift_compact_forcemimic.md): Comparison note on whether recent compliant and force-aware imitation papers materially overlap with the object-specific latent direction.

## Task Design

- [Task Candidates for Hidden-Physics Imitation Learning](task_candidates_for_hidden_com_material_latent.md): Ranked alternatives to box standing, emphasizing tasks where hidden physics matters beyond trajectory replay.

## Paper Notes

- [Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers](improving_robotic_imitation_learning_with_predicted_facial_motion_using_transformers.md): Transformer-based facial landmark prediction is fed into ACT for robotic feeding, improving success from 42% to 74% with short-horizon prediction.
- [Dynamical-Metalearning vs Face Prediction for Force/Material Latent](dynamical_metalearning_vs_face_prediction_for_force_latent.md): Why the system-identification repo is closer than Li-san's facial predictor to hidden-physics force learning, but still not the same as an explicit latent method.
- [DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation](dywa_dynamics_adaptive_world_action_model_for_generalizable_non_prehensile_manipulation.md): A recent single-view non-prehensile manipulation paper that combines world modeling, history-based dynamics adaptation, and FiLM conditioning.
- [DyWA vs Explicit Hidden-Physics Latent: The Controlled-Dynamics Gap](dywa_vs_hidden_physics_latent_controlled_dynamics_gap.md): Why DyWA is strong under mixed randomized dynamics, yet still leaves open the clean same-geometry hidden-physics evaluation that would better fit this project.
- [Force/Wrench Preprocessing in Local Paper Collection](force_wrench_preprocessing_in_local_papers.md): Local scan for papers that explicitly filter or frequency-transform force/wrench signals rather than using raw wrench directly.
- [AdaWorldPolicy Force Preprocessing](adaworldpolicy_force_preprocessing.md): AdaWorldPolicy does not appear to use explicit force filtering, but it does describe quantile-based normalization for force data to reduce outlier effects.
- [AdaWorldPolicy vs Same-Visual Different-Physics Gap](adaworldpolicy_vs_same_visual_different_physics_gap.md): Why AdaWorldPolicy still leaves open a cleaner evaluation where appearance is matched and adaptation must come from hidden-physics cues such as force prediction error.
