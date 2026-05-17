# Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers

## Summary

This paper studies a face-related robotic feeding task where the robot benefits from short-horizon prediction of human facial motion.
The method predicts future facial landmarks with a cross-attention Transformer and injects those predicted landmarks into an ACT-style imitation learning policy.
In simulation, adding facial motion information improves feeding success from 42% to 60%, and adding predicted future landmarks further improves it to 74%, with the best result at a 2-frame prediction horizon.

## Main Idea

The core claim is that static facial observations are insufficient for dynamic face-related manipulation such as feeding.
Instead of conditioning the policy only on current observations, the system first predicts near-future facial motion and then supplies those predictions to the control policy.
This is intended to reduce planning error and improve safety when the human face moves during interaction.

## Method

The system has two modules:

1. A facial motion prediction model based on a 4-layer Transformer with cross-attention between image features and facial landmark features.
2. An imitation learning controller based on ACT that uses predicted future facial landmarks as part of the observation.

The facial prediction model uses:

- RGB facial video at 30 FPS
- 468 MediaPipe landmarks per frame
- image resizing to `256 x 256`
- joint prediction of future facial images and landmarks during training

Three prediction approaches were compared:

1. Predict future images first, then extract landmarks.
2. Simple fusion of image and landmark embeddings.
3. Cross-attention fusion of image and landmark features.

The proposed cross-attention method performed best.

## Experimental Setup

The control experiments were run in `RoboManipBaselines` on a simulated feeding task.
The robot observes three RGB cameras plus robot joint states, while the facial prediction module uses a front-facing camera to predict future mouth motion.
A scripted policy generates demonstrations for grasping a cube, moving it toward the moving mouth, and releasing it.

Three observation settings were compared:

1. `Baseline`: no mouth landmark input
2. `Current Only`: current mouth position only
3. `Current + Prediction`: current mouth position plus future predicted mouth positions up to 5 frames ahead

The paper also evaluates both a clean setting and a perturbed setting that injects prediction-derived spatial error based on real facial data.

## Key Results

### Facial motion prediction

The cross-attention model achieved the best landmark consistency and image quality.
Reported aggregate metrics include:

- `Mean ΔU = 0.015`, `Var ΔU = 0.013`
- `Mean ΔV = 0.017`, `Var ΔV = 0.025`
- `Average SSIM = 0.928`

These outperform the weaker baselines described in the paper.

### Imitation learning

Success rate improved as follows:

- `Baseline`: `42%`
- `Current Only`: `60%`
- `Current + Prediction`: best at `74%` with a `2-frame` prediction horizon

The paper also reports that short horizons, especially `2-3` frames, gave faster and more stable execution than longer horizons.

## Relevance To This Project

This paper is relevant as a nearby example of improving imitation learning by augmenting observations with predicted future task-relevant state.
Its latent or predictive signal is not hidden physical property, but short-term human facial motion.
So it is not direct evidence for your hidden-physics thesis, but it is useful as a structural precedent for:

- predicting a task-relevant future variable before control
- injecting that prediction into an ACT-like imitation learning policy
- showing that dynamic partially changing targets can benefit from short-horizon prediction

Under the older project framing, the analogous move was to predict future force/contact consequences before control.
Under the current force-estimation framing, this paper is mainly a precedent for supplying an auxiliary estimated task-relevant signal to an imitation policy, not a direct template for the force estimator.

## Limits

- The experiments are in simulation for the control phase.
- The task is face-related feeding, not contact-rich manipulation driven by hidden material property.
- The gain comes from predicting human motion, not from estimating hidden mechanics of objects or tools.
- The paper uses landmark prediction rather than force, tactile, or material latent estimation.

## Source

- [raw/Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers - student report.pdf](../raw/Improving%20Robotic%20Imitation%20Learning%20with%20Predicted%20Facial%20Motion%20Using%20Transformers%20-%20student%20report.pdf)
