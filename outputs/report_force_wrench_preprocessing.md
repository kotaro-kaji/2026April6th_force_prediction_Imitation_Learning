# 力覚・wrench 前処理の有無に関するローカル論文調査

timestamp: 2026-04-27 14:35 JST
operation: query
changed files: `outputs/report_force_wrench_preprocessing.md`
result summary: ローカルの論文群を横断し、raw の force/wrench ではなく LPF・moving average・FFT などの前処理を明示的に使っている論文を抽出した。

## 結論

明確に見つかったのは 2 本。

1. `Adaptive Wiping Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations`
2. `Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control`

## 1. Adaptive Wiping

この論文は、力覚データに **Butterworth low-pass filter** をかけている。

- `unlabeled data` には offline で適用
- `FT demonstrations data` には online で適用
- その後に正規化

根拠:

> The datasets are pre-processed before training; we apply a Butterworth low-pass filter offline to unlabeled data and online to FT demonstrations data.

参照:
- `raw/Adaptive Wiping Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations.md:141`

## 2. Adaptive Compliance Policy (ACP)

この論文は、force/wrench に対して 2 種類の前処理・表現変換を明示的に行っている。

### 2-1. moving average filter

学習時に、エピソード全体の wrench data に **1 秒窓の moving average filter** をかけている。

目的は論文中で次の 2 つと説明されている。

- virtual target を smooth にする
- これから起こる接触に関する hindsight 情報を action label に持たせる

参照:
- `raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md:180`

### 2-2. FFT encoding

6 次元の force/torque 各チャネルを **FFT ベースの 2D spectrogram** に変換している。
つまり raw wrench をそのまま入れるのではなく、周波数表現に変えてからエンコードしている。

参照:
- `raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md:166`
- `raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md:191`
- `raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md:231`
- `raw/Adaptive Compliance Policy Learning Approximate Compliance for Diffusion Guided Control.md:266`

## 3. 近いが該当しなかったもの

### ForceMimic

`ForceMimic` は以下はある。

- wrench の gravity compensation
- point cloud の filtering

ただし、**force/wrench 自体への LPF, moving average, FFT, Fourier 変換**は、少なくとも今回確認したローカル原稿では明示されていなかった。

参照:
- `raw/ForceMimic Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation.md:46`
- `raw/ForceMimic Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation.md:72`

## 4. 明示的な記述を見つけられなかった論文

今回のローカル source では、少なくとも次については **force/wrench に対する明示的な LPF / Butterworth / moving average / FFT / Fourier** の記述を見つけられなかった。

- `In-the-Wild Compliant Manipulation with UMI-FT`
- `Bi-HIL Bilateral Control-Based Multimodal Hierarchical Imitation Learning`
- `Learning Variable Compliance Control From a Few Demonstrations for Bimanual Robot with Haptic Feedback Teleoperation System`
- `Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum`
- `DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation`
- `SCCRUB: Surface Cleaning Compliant Robot Utilizing Bristles`
- `Improving Robotic Imitation Learning with Predicted Facial Motion Using Transformers`
- `EXI-Net`
- `Imitation Learning System Design with Small Training Data for Flexible Tool Manipulation`

## 要約

「raw の wrench ではなく、明示的にフィルタや周波数変換をかけている論文」という意味では、ローカル collection 内の確定例は次の 2 本。

- `Adaptive Wiping`: Butterworth low-pass filter
- `Adaptive Compliance Policy`: moving average filter + FFT encoding
