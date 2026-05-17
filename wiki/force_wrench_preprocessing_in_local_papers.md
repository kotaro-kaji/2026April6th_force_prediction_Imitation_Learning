# Force/Wrench Preprocessing in Local Paper Collection

## Question

Which papers in the current local collection explicitly preprocess force or wrench signals with filters or frequency-domain transforms, instead of using raw wrench directly?

## Definite Positives

### Adaptive Wiping: Butterworth low-pass filter

- The paper explicitly states that the datasets are pre-processed with a **Butterworth low-pass filter**.
- The filtering is applied **offline to unlabeled data** and **online to FT demonstration data** before normalization.

Evidence:
- `raw/Adaptive Wiping Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations.md`, line 141.

### Adaptive Compliance Policy (ACP): moving average filter + FFT encoding

- The paper explicitly states that, during training, the whole episode of **wrench data is passed through a moving average filter with one-second window size**.
- It also explicitly uses **FFT encoding** of 6D force/torque data by converting each dimension into a **2D spectrogram**.
- Therefore ACP is the clearest example in this collection of both:
  1. filtering wrench before downstream target construction, and
  2. transforming force/torque into a frequency-domain representation.

Evidence:
- `raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md`, lines 166-180.
- Same file, lines 191-195, 231, 266 for the FFT ablation and discussion.

## Mentioned But Not a Force/Wrench Filter

### ForceMimic

- The paper mentions **gravity compensation** for captured wrench and filtering of **point-cloud observations**.
- I did **not** find an explicit statement that the force/wrench signal itself is low-pass filtered, smoothed, or Fourier-transformed.

Evidence:
- `raw/ForceMimic Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation.md`, lines 46 and 72.

## No Explicit Evidence Found in the Current Local Sources

- `In-the-Wild Compliant Manipulation with UMI-FT`
- `Bi-HIL Bilateral Control-Based Multimodal Hierarchical Imitation Learning`
- `Learning Variable Compliance Control From a Few Demonstrations for Bimanual Robot with Haptic Feedback Teleoperation System` (`raw/Learning Variable Compliance Control From a Few Demonstrations for Bimanual Robot with Haptic Feedback Teleoperation System.pdf`)
- `Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum`
- `DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation`
- `SCCRUB: Surface Cleaning Compliant Robot Utilizing Bristles`
- `Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers`
- `EXI-Net`
- `Imitation Learning System Design with Small Training Data for Flexible Tool Manipulation`

For these local sources, I did not find explicit wording about low-pass filtering, Butterworth filtering, moving-average smoothing, FFT/Fourier transforms, or other direct force/wrench preprocessing.

## Bottom Line

In the current local paper collection, the clear positives are:

1. `Adaptive Wiping` for **Butterworth low-pass filtering** of FT data.
2. `Adaptive Compliance Policy` for **moving-average filtering** of wrench and **FFT-based force/torque encoding**.

Everything else I checked either uses force/wrench without explicit preprocessing details, or mentions filtering only for non-force modalities.
