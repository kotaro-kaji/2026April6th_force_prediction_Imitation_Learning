Thanpimon Buamanee <sup>†1</sup>, Masato Kobayashi <sup>†1,2∗</sup>, Yuki Uranishi <sup>1</sup> <sup>†</sup> Equal Contribution, <sup>1</sup> The University of Osaka, <sup>2</sup> Kobe University,  
<sup>∗</sup> corresponding author: kobayashi.masato.cmc@osaka-u.ac.jp

###### Abstract

Long-horizon contact-rich robotic manipulation remains challenging due to partial observability and unstable subtask transitions under contact uncertainty. While hierarchical architectures improve temporal reasoning and bilateral imitation learning enables force-aware control, existing approaches often rely on flat policies that struggle with long-horizon coordination. We propose Bi-HIL, a bilateral control-based multimodal hierarchical imitation learning framework for long-horizon manipulation. Bi-HIL stabilizes hierarchical coordination by integrating keyframe memory with subtask-level progress rate that models phase progression within the active subtask and conditions both high- and low-level policies. We evaluate Bi-HIL on unimanual and bimanual real-robot tasks, demonstrating consistent improvements over flat and ablated variants. The results highlight the importance of explicitly modeling subtask progression together with force-aware control for robust long-horizon manipulation. For additional material, please check: https://mertcookimg.github.io/bi-hil

## I Introduction

Driven by population aging and labor shortages, robots are increasingly required to perform long-horizon, contact-rich manipulation in real-world environments. Such tasks unfold as sequences of subtasks, where small execution errors accumulate over time. Robust execution therefore requires both reliable task-level reasoning over extended horizons and stable low-level control under contact uncertainty [^1].

Two fundamental challenges arise in this setting. First, long-horizon manipulation suffers from *partial observability*: from limited observations, it is often unclear whether a subtask has completed or how far it has progressed. As a result, high-level policies may repeat or prematurely switch subtasks, leading to cascading failures [^2] [^3] [^4]. Second, contact-rich execution requires force-aware control. Small perturbations such as slips or misalignments can corrupt the perceived task state, further increasing ambiguity at subtask boundaries and destabilizing hierarchical coordination [^5].

For long-horizon manipulation, hierarchical frameworks primarily address the temporal reasoning challenge by decomposing behavior into high-level planning and low-level control [^6] [^7] [^8] [^9]. While keyframe-based memory (e.g., MemER [^6]) and task-progress estimation improve temporal abstraction, high-level decisions are still often inferred mainly from current observations without explicitly modeling subtask-level phase progression. Consequently, transition timing remains ambiguous and hierarchical coordination unstable in practice.

To achieve robust contact-rich manipulation in real-world settings, policies must be learned from demonstrations that capture both motion and interaction forces. Imitation learning (IL) provides a practical paradigm for acquiring such policies from human demonstrations. Unilateral teleoperation records kinematic behavior but omits force information, limiting robustness under contact uncertainty [^10] [^11] [^12] [^13]. In contrast, bilateral control records both position and force information, enabling force-aware policy learning [^14] [^15].

However, even with force-rich demonstrations, many existing bilateral control-based imitation learning (Bi-IL) approaches adopt a single flat policy that implicitly infers long-term task progression from current observations [^16] [^17] [^18]. Without explicit hierarchical coordination, subtask transitions remain unstable in long-horizon settings. As a result, the temporal reasoning challenge and the force-aware control challenge are often addressed separately rather than jointly.

To address both challenges in a unified manner, we propose Bi-HIL (Fig. 1), a bilateral control-based multimodal hierarchical imitation learning framework for long-horizon, contact-rich manipulation. Bi-HIL integrates (i) keyframe memory [^6] to anchor completed subtasks and (ii) subtask-level progress rate that models phase progression within the active subtask. The subtask-level progress rate is reset at each subtask transition, providing a consistent local coordination signal across hierarchical levels. Importantly, it conditions both the high-level policy and the low-level force-aware policy, enabling phase-aware short-horizon control.

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/bi-hil-t.png)

Figure 1: Concept of Bi-HIL

TABLE I: Comparison of Long-Horizon Robot Manipulation.

| Method | Hierarchy | Bilateral | Vision | Memory | Task Progress |
| --- | --- | --- | --- | --- | --- |
| HAMSTER [^9] | ✓ | ✗ | ✓ | ✗ | ✗ |
| SARM [^2] | ✓ | ✗ | ✓ | ✗ | ✓ |
| YaY Robot [^7] | ✓ | ✗ | ✓ | ✗ | ✗ |
| MemER [^6] | ✓ | ✗ | ✓ | ✓ | ✗ |
| Bi-ACT [^18] [^19] [^15] | ✗ | ✓ | ✓ | ✗ | ✗ |
| Bi-LAT [^20], Bi-VLA [^21] | ✗ | ✓ | ✓ | ✗ | ✗ |
| Hierarchical Bi-IL [^22] | ✓ | ✓ | ✗ | ✗ | ✗ |
| Bi-HIL (Ours) | ✓ | ✓ | ✓ | ✓ | ✓ |

Our contributions are threefold:

- We propose Bi-HIL, a bilateral multimodal hierarchical imitation learning framework for long-horizon contact-rich manipulation.
- We introduce a hierarchical coordination that combines keyframe memory with a resettable subtask-level progress rate, providing a subtask-local phase signal to stabilize transitions and condition low-level policy.
- We validate Bi-HIL on real-robot unimanual and bimanual tasks, with ablations confirming that both keyframe memory and subtask progress-rate conditioning are necessary for robust contact-rich execution.

## II Related Work

### II-A Framework for Long-Horizon Robotic Manipulation

Long-horizon manipulation requires executing sequential subtasks over extended time horizons, where errors in task progression or execution can accumulate and cause failure. Hierarchical frameworks address this by decomposing behavior into high-level task planning and low-level motor control. High-level policies provide subtask commands [^7] [^6], progress estimation [^8], or motion guidance [^9] to guide low-level controller, which executes continuous control.

In imitation learning, YaY Robot predicts high-level commands from visual observations [^7], and MemER introduces keyframe-based memory to improve temporal reasoning [^6]. BPP further enhances long-context imitation by selecting semantically important history frames to reduce distribution shift [^23]. Beyond architectural design, some methods explicitly encode task structure during learning: SARM employs stage-aware reward modeling [^2], and TOPReward extracts task progress directly from internal token probabilities of video VLMs for zero-shot estimation [^3].

However, as summarized in Table I, these approaches primarily assume unilateral control without force feedback, limiting applicability to contact-rich manipulation. Moreover, although keyframe memory provides boundary cues, it does not explicitly model subtask-level progression for low-level execution. Without progress-aware representation, transition timing remains ambiguous, leading to error accumulation.

In contrast, we propose Bi-HIL, where the high-level policy predicts subtask commands and subtask-level progress rate, while the low-level policy generates force-aware actions for robust long-horizon execution. Unlike prior work that estimates global task completion or trajectory-level value, our method models phase progression within each individual subtask to stabilize hierarchical coordination.

### II-B Bilateral Control-based Imitation Learning (Bi-IL)

Bi-IL enables force-aware manipulation from demonstrations. Early work used RNN and LSTM models [^14] [^16] [^17], while recent approaches adopt transformers [^24] for improved temporal modeling and data augumentation [^25] [^18] [^26] [^19]. Bi-ACT employs a transformer-based policy inspired by Action Chunking with Transformers (ACT) [^11] to perform force-aware tasks [^15]. Building on this foundation, language has been further incorporated: Bi-LAT uses language instructions to modulate force magnitude during execution [^20], while Bi-VLA uses language conditioning to disambiguate task intent under ambiguous observations [^21]. As summarized in Table I, most Bi-IL frameworks rely on a single flat policy that implicitly infers long-horizon task progression, without explicit hierarchical coordination. This becomes problematic in long-horizon settings, where partial observability and contact-induced perturbations destabilize subtask transitions. Although Hierarchical Bi-IL [^22] introduces hierarchical prediction of joint states, it does not leverage vision or language for global task reasoning.

In contrast, our Bi-HIL integrates hierarchical decision-making directly into Bi-IL. By introducing a high-level policy that predicts both subtask commands and a resettable subtask-level progress rate, and conditioning a force-aware low-level policy on these signals, we explicitly couple long-horizon reasoning with contact-aware control within a unified framework.

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/OverviewOfBiHIL.png)

Figure 2: Overview of Bi-HIL: Bilateral Control-Based Multimodal Hierarchical Imitation Learning via Subtask-Level Progress Rate and Keyframe Memory

## III Bi-HIL: Bilateral Control-Based Multimodal Hierarchical Imitation Learning via Subtask-Level Progress and Keyframe Memory

### III-A Overview

Bi-HIL adopts a hierarchical framework consisting of a high-level policy and a low-level policy as shown in Fig. 2. High-level policy performs subtask-level reasoning by predicting subtask commands and subtask-level progress rate, while maintaining task memory using representative keyframes. Low-level policy predicts motor actions conditioned on visual observations, robot joint states, and high-level guidance.

During inference, the high-level policy provides temporally grounded task context, and the low-level policy executes continuous control actions based on this structured information. This hierarchical design enables robust execution of long-horizon manipulation tasks.

### III-B Data Collection

As shown in Fig. 2, Bi-HIL employs a four-channel bilateral control method for data collection, in which leader robot controlled by operator and follower robot interacts with the environment and provides force feedback to the leader. The control objective is defined as:

$$
\theta_{l}-\theta_{f}=0
$$
 
$$
\tau_{l}+\tau_{f}=0
$$

where $\theta$ and $\tau$ denote the joint angle and torque, respectively, and the subscripts $l$ and $f$ indicate the leader and follower systems. Condition (1) enforces synchronized motion between the leader and follower, while condition (2) ensures force consistency via an action–reaction relationship. Joint angles are measured using encoders, and angular velocities are computed by numerical differentiation. Disturbance torques are estimated using a disturbance observer (DOB) [^27], and reaction torques are inferred via a reaction force observer (RFOB) [^28].

After data collection, subtask boundaries are manually annotated based on visual observations using natural language instructions. These annotations supervise high-level policy training.

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/highPolicyArchitecture.png)

Figure 3: High-Policy Architecture

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/KeyframeProgressRateExplaination.png)

Figure 4: Definition of Representative Keyframes and Subtask-Level Progress Rate

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/low-v1.png)

Figure 5: Low-Policy Architecture

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/uni-env3.png)

Figure 6: Data Collection of Put-Three-Balls-In-Drawer Task

### III-C High-Level Policy

As shown in Fig. 3, the high-level policy is a transformer encoder that performs subtask-level reasoning. It extends a YaY-style architecture [^7] with (i) keyframe-based memory inspired by MemER [^6] and (ii) subtask-level progress prediction. These components enable temporally consistent decision-making under partial observability.

##### Inputs.

At each high-level timestep $t$, the policy receives: (i) a window of the most recent $N$ observations from each camera, $R_{t}=I_{t-N:t}$ and, (ii) a set of previously selected keyframes $K_{t}\subseteq I_{0:t-1}$, where $|K_{t}|\leq K_{\max}$. All images are encoded by a frozen CLIP encoder and processed by the transformer.

##### Outputs.

The policy predicts three elements: (i) a subtask command $C_{t}$, (ii) a subtask-level progress rate $p_{t}\in[0,1]$, and (iii) keyframe scores for each frame in the current window $R_{t}$. Definition of representative keyframes and subtask-level progress rate are as shown in Fig. 4. For a subtask $s$ with start timestep $t_{s}^{\mathrm{Start}}$ and end timestep $t_{s}^{\mathrm{End}}$, the subtask-level progress rate is defined as

$$
p_{s}=\frac{t-t_{s}^{\mathrm{Start}}}{t_{s}^{\mathrm{End}}-t_{s}^{\mathrm{Start}}},
$$

which represents the normalized completion ratio of the active subtask. The subtask-level progress rate is reset to zero when a new subtask begins.

##### Keyframe memory update.

Positions in $R_{t}$ whose predicted keyframe probability exceeds a threshold $O_{th}$ are treated as candidate keyframes. Candidate indices are accumulated over time and clustered using 1D single-linkage with a fixed temporal distance. For each cluster, the median index is selected as the representative keyframe and stored in memory $K_{t}$, subject to the limit $K_{\max}$. These keyframes serve as visual anchors of completed subtasks.

##### Training objective.

The command is optimized using cross-entropy loss, the subtask-level progress rate using mean absolute error (MAE), and keyframe prediction using weighted binary cross-entropy (BCE) to address class imbalance. The total loss is

$$
\mathcal{L}_{\mathrm{high}}=\lambda_{\mathrm{cmd}}\mathcal{L}_{\mathrm{cmd}}+\lambda_{\mathrm{prog}}\mathcal{L}_{\mathrm{prog}}+\lambda_{\mathrm{keyframe}}\mathcal{L}_{\mathrm{keyframe}},
$$

where $\lambda_{\mathrm{cmd}}$, $\lambda_{\mathrm{prog}}$, and $\lambda_{\mathrm{keyframe}}$ balances auxiliary losses.

### III-D Low-Level Policy

As shown in Fig. 5, the low-level policy is implemented as a transformer-based conditional variational autoencoder (CVAE [^29]). The model receives the subtask-level progress rate and the SigLIP-embedded subtask instruction predicted by the high-level policy, along with the current RGB images and the follower robot’s joint states (angle, angular velocity, and torque). Based on these inputs, the low-level policy predicts the next joint states of the leader robot (angle, angular velocity, and torque). Specifically, the SigLIP-encoded subtask instruction and RGB images are first processed by a FiLM-conditioned EfficientNet, where the language embedding modulates the visual features via FiLM. Subtask-level progress rate is discretized into ten uniform levels to improve robustness. The extracted visual features and subtask-level progress rate merged with language embedding are then passed to a transformer encoder to predict the leader robot motion. The low-level policy is trained to minimize

$$
\mathcal{L}_{\mathrm{low}}=\mathcal{L}_{\mathrm{action}}+\beta\,\mathcal{L}_{\mathrm{KL}},
$$

where $\mathcal{L}_{\mathrm{action}}$ is the mean absolute error (L1) between the predicted and target action chunks, and $\mathcal{L}_{\mathrm{KL}}$ is the KL divergence. The weight $\beta$ balances reconstruction and regularization.

TABLE II: Experimental Results: Put-Three-Balls-in-Drawer. KF: Keyframe, SPR: Subtask-Level Progress Rate.

<table><tbody><tr><td rowspan="2">Model Name</td><td rowspan="2">Type</td><td colspan="13">Completed Subtask(%)</td></tr><tr><td>#1</td><td>#2</td><td>#3</td><td>#4</td><td>#5</td><td>#6</td><td>#7</td><td>#8</td><td>#9</td><td>#10</td><td>#11</td><td>#12</td><td>Success</td></tr><tr><td rowspan="2">Bi-ACT (Baseline)</td><td>Left</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td>60</td><td rowspan="2">80</td></tr><tr><td>Right</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td rowspan="2">Bi-HIL (w/o KF&SPR)</td><td>Left</td><td>80</td><td>60</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td rowspan="2">30</td></tr><tr><td>Right</td><td>60</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td rowspan="2">Bi-HIL (w/o SPR)</td><td>Left</td><td>100</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td rowspan="2">50</td></tr><tr><td>Right</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td></tr><tr><td rowspan="2">Bi-HIL (w/o KF)</td><td>Left</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>60</td><td>60</td><td>60</td><td rowspan="2">70</td></tr><tr><td>Right</td><td>100</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td></tr><tr><td rowspan="2">Bi-HIL (Proposed)</td><td>Left</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td rowspan="2">100</td></tr><tr><td>Right</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></tbody></table>

## IV UnimanualExperiments

### IV-A Hardware

TABLE III: Model Comparison on the Put-Three-Balls-in-Drawer Task. KF: Keyframe, SPR: subtask-level Progress Rate.

<table><tbody><tr><td rowspan="2">Model Name</td><td colspan="3">High-level Output</td><td colspan="2">Low-level Input</td></tr><tr><td>KF</td><td>SPR</td><td>Command</td><td>SPR</td><td>Command</td></tr><tr><td>Bi-ACT (Baseline)</td><td colspan="3">N/A</td><td>✗</td><td>✗</td></tr><tr><td>Bi-HIL (w/o KF&SPR)</td><td>✗</td><td>✗</td><td>✓</td><td>✗</td><td>✓</td></tr><tr><td>Bi-HIL (w/o SPR)</td><td>✓</td><td>✗</td><td>✓</td><td>✗</td><td>✓</td></tr><tr><td>Bi-HIL (w/o KF)</td><td>✗</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>Bi-HIL (Proposed)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>

As shown in Fig. 6, unimanual experiments were conducted using OpenMANIPULATOR-X robotic arms developed by ROBOTIS. The setup consists of two robots: a leader robot operated by a human demonstrator and a follower robot interacting with the environment. Each robot is equipped with 4 degrees of freedom (DOF) for arm motion and an additional DOF for the gripper, resulting in a total of 5 actuated joints. The control cycle was set to 1000 Hz for precise movement. Furthermore, two RGB cameras were positioned top and in the gripper area of the follower robot to record observations at 100 Hz.

### IV-B Task Setting

We evaluate the proposed method on the long-horizon manipulation task *Put-Three-Balls-in-Drawer*, in which the robot sequentially places green, red, and white balls into their corresponding drawers from top to bottom. The positions of the green and white balls are interchangeable, requiring the robot to rely on visual observations to determine the correct execution order.

This task presents three challenges: long temporal duration (59.2,s on average), visually similar subtasks, and ambiguous initial configurations. The task consists of 12 subtasks as shown in Fig. 6, and the execution order depends on the initial ball placement. For example, when the green ball is placed on the left (Left configuration), the robot must pick the left ball first; when it is placed on the right (Right configuration), the picking order changes accordingly. This setup requires effective temporal reasoning and contextual understanding for successful execution.

### IV-C Training Setting

Six demonstration episodes were collected using bilateral control (three Left, three Right). Each episode was 57.3–61.2 seconds. Demonstrations were manually annotated with subtask boundaries and augmented using DABI [^26], increasing the dataset to 60 demonstrations. These augmented data were used to train Bi-ACT, Bi-HIL, and ablation variants of Bi-HIL, ass shown in Table III.

We fix the following constants for reproducibility. High-level policy: image window length $N=5$ (history length $T=4$ plus current frame), maximum keyframes $K_{\max}=8$, and loss weights $\lambda_{\mathrm{cmd}}=\lambda_{\mathrm{prog}}=\lambda_{\mathrm{keyframe}}=1$ in Eq. (4). The high-level policy is trained with Adam, learning rate $1\times 10^{-4}$. Low-level policy: subtask-level progress rate is discretized into 10 levels; the transformer-based CVAE has 4 encoder and 7 decoder layers, 8 attention heads, hidden dimension 512, feedforward dimension 3200. At inference, the high-level policy runs at $f_{h}=1$  Hz and the low-level at $f_{l}=100$  Hz. The low-level policy is trained with Adam, learning rate $1\times 10^{-5}$, KL weight $\beta=10$.

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/uni-key-v0.png)

Figure 7: Put-Three-Balls-in-Drawer: Subtask-Level Progress Rate and Keyframe Memory

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/bi-env-v2.png)

Figure 8: Data Collection of 6-Cup Downstack and 4-Peg-in-Hole

### IV-D Experiment Result

Table II reports the performance of each model over ten trials: five on Put-Three-Balls-in-Drawer (Left) and five on Put-Three-Balls-in-Drawer (Right). For failed trials, the completed subtask columns indicate the furthest progress achieved by the robot before failure, ranging from subtask (#1) Open top drawer to subtask (#12) Close bottom drawer. The Success column reports the overall task success rate.

For the baseline method Bi-ACT, a 100% success rate (5/5) is achieved in the Right configuration, whereas performance drops to 60% (3/5) in the Left configuration, with the model mistakenly opening the drawer in an incorrect order. This reveals difficulty in handling ambiguous task progression and highlights the need for hierarchical reasoning.

Despite lacking a high-level policy, Bi-ACT performs second-best among the models, outperforming the Bi-HIL ablation variants that incorporate a high-level policy. This indicates that simply adding a high-level policy does not guarantee improved performance without appropriate design.

The proposed Bi-HIL model is the only method to achieve a 100% success rate on both configurations(5/5 for Left, 5/5 for Right), demonstrating superior performance on this task. Fig. 7 shows the evolution of predicted subtasks and subtask-level progress rate over time. The visualization shows a clear progression from lower left to upper right, indicating consistent subtask prediction and progress estimation advancement. Keyframes (red) are predicted near subtask boundaries, demonstrating effective memory selection.

Ablation results demonstrate that both keyframe memory and subtask-level progress rate are essential: commands alone lead to unstable predictions, keyframes without subtask-level progress rate lack temporal guidance, and subtask-level progress rate without keyframes lacks reliable task memory. In contrast, the full Bi-HIL model integrates both components to achieve stable reasoning and reliable execution.

## V Bimanual Experiments

### V-A Hardware

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/bi-e.png)

Figure 9: Experimental Setup

As shown in Fig. 9, ALPHA- $\alpha$ were used for experiments of bimanual robotic manipulation. A total of four robots were utilized in the experiments, including two leader robots operated by the human operator and two follower robots. Each robot has six degrees of freedom (DOF) for versatile movement, as well as an additional DOF for the gripper, utilizing a total of seven motors for its operation. The bilateral control cycle was set to 1000 Hz for precise movement and data collection of joint angle, velocity, and torque. Furthermore, four RGB cameras were placed top, on the sides, and at both the right and left gripper areas of the follower robots to record observations.

### V-B Task Setting

To examine the applicability of Bi-HIL, experiments were conducted on ”6-Cup Downstack” and ”4-Peg-in-Hole” task, as shown in Fig. 8.

#### V-B1 6-Cup Downstack

The 6-cup downstack is performed through a sequence of structured pick-and-insert actions, progressively nesting the upper tiers into the base layer and consolidating the cups into a single centralized stack.

#### V-B2 4-Peg-in-Hole

The 4-Peg-in-Hole task consists of a sequence of precise pick-and-insert actions, where the robot grasps geometric pegs and inserts them into their corresponding holes. The 4-Peg-in-Hole task emphasizes accurate alignment and force-sensitive insertion under contact-rich conditions. The peg and hole geometries are adopted from FMB [^30]. Both the pegs and holes are scaled to 0.8 times their original size and fabricated using a 3D printer.

### V-C Training Setting

For each task, we collected five demonstrations as training data. Each 6-Cup Downstack episode was 45.1–47.3 seconds. Each 4-Peg-in-Hole episode was 49.5–54.6 seconds. Demonstrations were manually annotated with subtask boundaries and augmented using DABI [^26], increasing the dataset to 50 demonstrations. These were used to train Bi-ACT without force feedback, Bi-ACT, and Bi-HIL. The high-level and low-level policy of parameters are the same as in the unimanual experiments.

### V-D Experimental Results

#### V-D1 6-Cup Downstack Evaluation

TABLE IV: Experimental Results: 6-Cup Downstack

<table><tbody><tr><td rowspan="2">Model Name</td><td colspan="9">Completed Subtask(%)</td></tr><tr><td>#1</td><td>#2</td><td>#3</td><td>#4</td><td>#5</td><td>#6</td><td>#7</td><td>#8</td><td>Success</td></tr><tr><td>Bi-ACT (w/o Force)</td><td>60</td><td>60</td><td>60</td><td>60</td><td>40</td><td>40</td><td>40</td><td>20</td><td>20</td></tr><tr><td>Bi-ACT</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>80</td><td>80</td><td>60</td><td>60</td></tr><tr><td>Bi-HIL</td><td>100</td><td>100</td><td>100</td><td>100</td><td>80</td><td>80</td><td>80</td><td>80</td><td>80</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/bi-key1-v0.png)

Figure 10: 6-Cup Downstack: Subtask-Level Progress Rate and Keyframe

Table IV reports the experimental results on the 6-Cup Downstack task over five trials. For failed trials, the completed subtask indicates the furthest progress achieved before execution failure. The Success column reports the overall task success rate.

Bi-ACT without force feedback achieves only 20% success (1/5), with performance degradation occurring after subtask 5. Failures were mainly caused by unstable insertion during contact-rich stacking motions, highlighting the importance of force information in bimanual manipulation.

Bi-ACT with force control improves the success rate to 60% (3/5), successfully completing the initial stacking phases. However, performance decreases in later subtasks (7–8), where coordinated bimanual reasoning is required to merge the final cup structures. These results indicate that while force feedback improves execution stability, a flat policy without hierarchical reasoning struggles with long-horizon coordination.

The proposed Bi-HIL achieves the highest performance with an 80% success rate (4/5). All trials successfully completed subtasks 1–4, and most trials progressed consistently through subtasks 5–8. Compared to Bi-ACT, Bi-HIL shows improved stability in the later merging stages, demonstrating that hierarchical task reasoning combined with force-aware low-level control enhances long-horizon bimanual manipulation. Fig. 10 shows the evolution of the predicted subtasks and progress rate over time for the 6-Cup Downstack task. The visualization illustrates an overall progression from the lower left to the upper right, indicating consistent subtask prediction and progress estimation. Although slight fluctuations are observed in the subtask-level progress rate, the prediction stabilizes in subsequent steps. Keyframe candidates, shown in red, are predicted seven times, capturing the end of each subtask.

These results confirm that integrating subtask-level reasoning with bilateral control improves robustness in structured, contact-rich bimanual tasks such as the 6-Cup Downstack.

#### V-D2 4-Peg-in-Hole Evaluation

TABLE V: Experimental Results: 4-Peg-in-Hole

<table><tbody><tr><td rowspan="2">Model Name</td><td colspan="9">Completed Subtask(%)</td></tr><tr><td>#1</td><td>#2</td><td>#3</td><td>#4</td><td>#5</td><td>#6</td><td>#7</td><td>#8</td><td>Success</td></tr><tr><td>Bi-ACT (w/o Force)</td><td>100</td><td>80</td><td>40</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Bi-ACT</td><td>100</td><td>100</td><td>100</td><td>60</td><td>60</td><td>40</td><td>40</td><td>20</td><td>20</td></tr><tr><td>Bi-HIL</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>80</td><td>80</td><td>80</td><td>80</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2603.13315v2/fig/bi-key2-v0.png)

Figure 11: 4-Peg-in-Hole: Subtask-Level Progress Rate and Keyframe

Table V summarizes the experimental results on the 4-Peg-in-Hole task over five trials. For failed trials, the completed subtask indicates the furthest progress achieved prior to execution failure, and the Success column reports the overall task success rate.

Bi-ACT without force feedback fails to complete the task in all trials (0% success). While the model consistently accomplishes the initial grasping subtasks (#1–#2), performance degrades during the first insertion phase, and no trial progresses beyond subtask #3. This failure pattern indicates that vision-only control is insufficient for maintaining stable alignment and contact during peg insertion, where precise force modulation is required.

Introducing force feedback (Bi-ACT) improves early-stage execution, with all trials successfully completing subtasks #1–#3. However, the overall success rate remains limited to 20% (1/5). Most failures occur in later insertion stages (#6–#8), where sequential coordination across multiple pegs is necessary. These results suggest that although force feedback enhances local contact stability, a flat policy lacks the temporal abstraction required for consistent long-horizon execution.

The proposed Bi-HIL achieves the highest performance, attaining an 80% success rate (4/5). All trials successfully complete subtasks #1–#4, and the majority progress reliably through the remaining insertion phases. Compared to both Bi-ACT variants, Bi-HIL exhibits markedly improved robustness during repeated alignment and insertion operations. Fig. 11 shows that predicted subtasks and progress rate evolve smoothly over time, and keyframe candidates concentrate near subtask boundaries, indicating stable phase estimation and transition detection.

Overall, these results indicate that hierarchical task decomposition combined with force-aware low-level control substantially improves reliability in contact-rich assembly tasks. Explicit modeling of subtask-level progress provides structured temporal guidance that enables consistent coordination across sequential insertion phases.

## VI Conclusion

We presented Bi-HIL, a bilateral control-based multimodal hierarchical imitation learning framework for long-horizon contact-rich manipulation. Bi-HIL couples a high-level policy that predicts subtask commands and a resettable subtask-level progress rate with a force-aware low-level policy learned from bilateral demonstrations. Experiments on real robots show that Bi-HIL improves robustness over baselines and ablated variants on both unimanual and bimanual settings. On the unimanual task, Bi-HIL achieves reliable long-horizon execution with stable subtask transitions. On bimanual contact-rich tasks, Bi-HIL consistently outperforms force-aware baseline policies, particularly in later subtasks. These results indicate that explicit subtask-level phase modeling, together with keyframe memory and force-aware control, is critical for robust long-horizon manipulation.

[^1]: O. Kroemer, S. Niekum, and G. Konidaris, “A review of robot learning for manipulation: Challenges, representations, and algorithms,” *Journal of Machine Learning Research*, vol. 22, no. 30, pp. 1–82, 2021. \[Online\]. Available: http://jmlr.org/papers/v22/19-804.html

[^2]: Q. Chen, J. Yu, M. Schwager, P. Abbeel, Y. Shentu, and P. Wu, “Sarm: Stage-aware reward modeling for long horizon robot manipulation,” *arXiv preprint arXiv:2509.25358*, 2025.

[^3]: S. Chen, C. Harrison, Y.-C. Lee, A. J. Yang, Z. Ren, L. J. Ratliff, J. Duan, D. Fox, and R. Krishna, “Topreward: Token probabilities as hidden zero-shot rewards for robotics,” *arXiv preprint arXiv:2602.19313*, 2026.

[^4]: Y. J. Ma, J. Hejna, A. Wahid, C. Fu, D. Shah, J. Liang, Z. Xu, S. Kirmani, P. Xu, D. Driess, T. Xiao, J. Tompson, O. Bastani, D. Jayaraman, W. Yu, T. Zhang, D. Sadigh, and F. Xia, “Vision language models are in-context value learners,” *arXiv preprint arXiv:2411.04549*, 2024.

[^5]: T. Tsuji, Y. Kato, G. Solak, H. Zhang, T. Petrič, F. Nori, and A. Ajoudani, “A survey on imitation learning for contact-rich tasks in robotics,” *arXiv preprint arXiv:2506.13498*, 2025.

[^6]: A. Sridhar, J. Pan, S. Sharma, and C. Finn, “Memer: Scaling up memory for robot control via experience retrieval,” *arXiv preprint arXiv:2510.20328*, 2025.

[^7]: L. X. Shi, Z. Hu, T. Z. Zhao, A. Sharma, K. Pertsch, J. Luo, S. Levine, and C. Finn, “Yell at your robot: Improving on-the-fly from language corrections,” in *Proceedings of Robotics: Science and Systems(RSS) 2024*, ser. Proceedings of Robotics: Science and Systems(RSS). Delft, Netherlands: RSS, 15–19 July 2024.

[^8]: T. W. Ayalew, X. Zhang, K. Y. Wu, T. Jiang, M. Maire, and M. Walter, “Progressor: A perceptually guided reward estimator with self-supervised online refinement,” in *2025 International Conference on Computer Vision, (ICCV)*, 2025. \[Online\]. Available: https://arxiv.org/abs/2411.17764

[^9]: Y. Li, Y. Deng, J. Zhang, J. Jang, M. Memmel, R. Yu, C. R. Garrett, F. Ramos, D. Fox, A. Li, A. Gupta, and A. Goyal, “Hamster: Hierarchical action models for open-world robot manipulation,” 2025. \[Online\]. Available: https://arxiv.org/abs/2502.05485

[^10]: P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel, “Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators,” in *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2024, pp. 12 156–12 163.

[^11]: T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” *arXiv preprint arXiv:2304.13705*, 2023.

[^12]: Z. Fu, T. Z. Zhao, and C. Finn, “Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation,” 2024. \[Online\]. Available: https://arxiv.org/abs/2401.02117

[^13]: A.. Team, J. Aldaco, T. Armstrong, R. Baruch, J. Bingham, S. Chan, K. Draper, D. Dwibedi, C. Finn, P. Florence, S. Goodrich, W. Gramlich, T. Hage, A. Herzog, J. Hoech, T. Nguyen, I. Storz, B. Tabanpour, L. Takayama, J. Tompson, A. Wahid, T. Wahrburg, S. Xu, S. Yaroshenko, K. Zakka, and T. Z. Zhao, “Aloha 2: An enhanced low-cost hardware for bimanual teleoperation,” 2024. \[Online\]. Available: https://arxiv.org/abs/2405.02292

[^14]: T. Adachi, K. Fujimoto, S. Sakaino, and T. Tsuji, “Imitation learning for object manipulation based on position/force information using bilateral control,” in *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2018, pp. 3648–3653.

[^15]: M. Kobayashi, T. Buamanee, and T. Kobayashi, “Alpha- $\alpha$ and bi-act are all you need: Importance of position and force information/ control for imitation learning of unimanual and bimanual robotic manipulation with low-cost system,” *IEEE Access*, vol. 13, pp. 29 886–29 899, 2025.

[^16]: S. Sakaino, K. Fujimoto, Y. Saigusa, and T. Tsuji, “Imitation learning for variable speed contact motion for operation up to control bandwidth,” in *IEEE Open Journal of the Industrial Electronics Society*, vol. 3, 2022, pp. 116–127.

[^17]: K. Fujimoto, S. Sakaino, and T. Tsuji, “Time series motion generation considering long short-term motion,” in *2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2019, pp. 6842–6848.

[^18]: T. Buamanee, M. Kobayashi, Y. Uranishi, and H. Takemura, “Bi-act: Bilateral control-based imitation learning via action chunking with transformer,” *arXiv preprint arXiv:2401.17698*, 2024.

[^19]: K. Yamane, Y. Li, M. Konosu, K. Inami, J. Oaki, S. Sakaino, and T. Tsuji, “Fast bilateral teleoperation and imitation learning using sensorless force control via accurate dynamics model,” *arXiv preprint arXiv:2507.06174*, 2025.

[^20]: T. Kobayashi, M. Kobayashi, T. Buamanee, and Y. Uranishi, “Bi-lat: Bilateral control-based imitation learning via natural language and action chunking with transformers,” *arXiv preprint arXiv:2504.01301*, 2025.

[^21]: M. Kobayashi and T. Buamanee, “Bi-vla: Bilateral control-based imitation learning via vision-language fusion for action generation,” *arXiv preprint arXiv:2509.18865*, 2025.

[^22]: K. Hayashi, S. Sakaino, and T. Tsuji, “An independently learnable hierarchical model for bilateral control-based imitation learning applications,” in *IEEE Access*, vol. 10, 2022, pp. 32 766–32 781.

[^23]: M. S. Mark, J. Liang, M. Attarian, C. Fu, D. Dwibedi, D. Shah, and A. Kumar, “Bpp: Long-context robot imitation learning by focusing on key history frames,” 2026. \[Online\]. Available: https://arxiv.org/abs/2602.15010

[^24]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Proceedings of the 31st International Conference on Neural Information Processing Systems*, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 6000–6010.

[^25]: M. Kobayashi, T. Buamanee, Y. Uranishi, and H. Takemura, “Ilbit: Imitation learning for robot using position and torque information based on bilateral control with transformer,” *IEEJ Journal of Industry Applications*, vol. 14, no. 2, pp. 161–168, 2025.

[^26]: M. Kobayashi, T. Buamanee, and Y. Uranishi, “Dabi: Evaluation of data augmentation methods using downsampling in bilateral control-based imitation learning with images,” in *2025 IEEE International Conference on Robotics and Automation (ICRA)*, 2025, pp. 16 892–16 898.

[^27]: K. Ohnishi, M. Shibata, and T. Murakami, “Motion control for advanced mechatronics,” *IEEE/ASME Transactions on Mechatronics*, vol. 1, no. 1, pp. 56–67, 1996.

[^28]: T. Murakami, F. Yu, and K. Ohnishi, “Torque sensorless control in multidegree-of-freedom manipulator,” *IEEE Transactions on Industrial Electronics*, vol. 40, no. 2, pp. 259–265, 1993.

[^29]: K. Sohn, H. Lee, and X. Yan, “Learning structured output representation using deep conditional generative models,” in *Advances in Neural Information Processing Systems*, C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, Eds., vol. 28. Curran Associates, Inc., 2015. \[Online\]. Available: https://proceedings.neurips.cc/paper\_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf

[^30]: J. Luo, C. Xu, F. Liu, L. Tan, Z. Lin, J. Wu, P. Abbeel, and S. Levine, “Fmb: A functional manipulation benchmark for generalizable robotic learning,” *The International Journal of Robotics Research*, vol. 44, no. 4, pp. 592–606, 2025.