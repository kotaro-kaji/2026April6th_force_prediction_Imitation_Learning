Yulin Zhou, Yuankai Lin, Fanzhe Peng, Jiahui Chen, Kaiji Huang, Hua Yang, and Zhouping Yin Manuscript received: May 18, 2025; Revised: August 9, 2025; Accepted: September 20, 2025.This paper was recommended for publication by Editor Aleksandra Faust upon evaluation of the Associate Editor and Reviewers’ comments. This work was supported by the Joint Funds of the National Natural Science Foundation of China (Grant No. U22A20208), the Natural Science Foundation Innovation Group Project of Hubei Province (Grant No. 2022CFA018), and the Key Research and Development Program of Guangdong Province (Grant No. 2022B0202010001-2).(Corresponding author: Hua Yang.)Y. Zhou, Y. Lin, F. Peng, J. Chen, K. Huang, H. Yang, and Z. Yin are with the School of Mechanical Science and Engineering, Huazhong University of Science and Technology, Wuhan 430074, China (e-mail: yulinzhou@hust.edu.cn; huayang@hust.edu.cn).Digital Object Identifier (DOI): 10.1109/LRA.2025.3615520.

###### Abstract

Standard imitation learning (IL) methods have achieved considerable success in robotics, yet often rely on the Markov assumption, which falters in long-horizon tasks where history is crucial for resolving perceptual ambiguity. This limitation stems not only from a conceptual gap but also from a fundamental computational barrier: prevailing architectures like Transformers are often constrained by quadratic complexity, rendering the processing of long, high-dimensional observation sequences infeasible. To overcome this dual challenge, we introduce Mamba Temporal Imitation Learning (MTIL). Our approach represents a new paradigm for robotic learning, which we frame as a practical synthesis of World Model and Dynamical System concepts. By leveraging the linear-time recurrent dynamics of State Space Models (SSMs), MTIL learns an implicit, action-oriented world model that efficiently encodes the entire trajectory history into a compressed, evolving state. This allows the policy to be conditioned on a comprehensive temporal context, transcending the confines of Markovian approaches. Through extensive experiments on simulated benchmarks (ACT, Robomimic, LIBERO) and on challenging real-world tasks, MTIL demonstrates superior performance against SOTA methods like ACT and Diffusion Policy, particularly in resolving long-term temporal ambiguities. Our findings not only affirm the necessity of full temporal context but also validate MTIL as a powerful and a computationally feasible approach for learning long-horizon, non-Markovian behaviors from high-dimensional observations. Project code are available at [https://github.com/yulinzhouZYL/MTIL](https://github.com/yulinzhouZYL/MTIL)

## I Introduction

Imitation Learning (IL) has emerged as a powerful paradigm for teaching robots complex skills directly from expert demonstrations, bypassing the need for intricate reward engineering often required in reinforcement learning [^1] [^2] [^3]. Behavioral Cloning (BC), the simplest form of IL, learns a direct mapping from observations to actions via supervised learning and has enabled robots to perform a variety of tasks [^2] [^3] [^4] [^1]. Recent advancements, particularly leveraging powerful sequence models and generative approaches, have led to state-of-the-art (SOTA) methods such as the Action Chunking Transformer (ACT) [^5] [^6] and Diffusion Policy [^7] [^8] [^9] [^10], which excel at learning visuomotor control policies for complex manipulation. Despite these successes, a fundamental limitation persists in many current IL approaches: the reliance on the Markov assumption. These methods typically predict the action $a_{t}$ based solely on the current observation $o_{t}$ or a short, fixed-length history window $o_{t-k:t}$ [^1] [^11] [^12] [^13]. This assumption breaks down in tasks where the history beyond this limited window is necessary to resolve ambiguity in the current observation. Consider a sequential task requiring a robot to first place an object at location A, and subsequently move it to location B. At an intermediate configuration, the robot’s visual observation and proprioceptive state ($o_{t}$) might be identical regardless of whether it has successfully completed the sub-task at location A. A Markovian policy, lacking the memory of visiting A, cannot distinguish these fundamentally different historical contexts and may erroneously proceed directly to B, failing the task [^1] [^14] [^15]. This temporal ambiguity signifies an underlying Partially Observable Markov Decision Process (POMDP), posing a critical challenge for standard IL methods in state-dependent tasks.

Since ambiguous tasks manifest as POMDPs and human demonstrations are inherently non-Markovian, effective imitation necessitates history-aware policies. While the critical role of historical context has been increasingly recognized [^12], a fundamental barrier has remained: the computational infeasibility of processing long, high-dimensional observation histories with prevailing architectures like Transformers. Addressing this, we introduce Mamba Temporal Imitation Learning (MTIL). Our approach is specifically designed to incorporate the complete observational history into decision-making by leveraging the unique properties of State Space Models (SSMs), particularly the recently developed Mamba architecture [^16] [^17]. Mamba’s recurrent structure allows it to maintain a compressed hidden state $h_{t}$ that theoretically encapsulates information from the entire preceding observation sequence $H_{t}=(o_{1},...,o_{t})$. Instead of relying solely on $o_{t}$, MTIL learns a policy $\pi(a_{t}|h_{t},o_{t})$ that explicitly conditions the action prediction on this history-infused hidden state $h_{t}$ in conjunction with the current observation $o_{t}$, enabling differentiation between observationally similar states and the correct execution of complex sequential tasks. Our contributions are threefold:

1. We propose MTIL, a novel architecture that is the first to leverage the linear-time recurrence of State Space Models to make full-trajectory imitation learning from high-dimensional visual data computationally feasible on commodity hardware, breaking the quadratic bottleneck of attention-based models.
2. We provide a new theoretical framing for this approach, positing MTIL as learning an implicit dynamical system. This system’s evolving state acts as a continuous belief-state representation, offering a robust solution to the core problem of temporal ambiguity in partially observable environments.
3. We provide extensive empirical validation demonstrating that MTIL significantly outperforms state-of-the-art methods, including ACT and Diffusion Policy. Furthermore, on tasks requiring long-term memory, MTIL also surpasses other full-history-capable baselines like Transformer-XL, validating the unique advantages of its underlying architecture.

## II Related Works

### II-A Markovian and Short-History Imitation Learning

A cornerstone of imitation learning, Behavioral Cloning (BC), typically learns a Markovian policy $\pi(a_{t}|o_{t})$ via supervised learning [^2] [^3] [^4] [^1]. Although fundamental, this approach inherently struggles with covariate shift and tasks that require memory beyond current observation [^3] [^11]. Many contemporary methods, despite advances, effectively operate within similar constraints or rely on limited observation histories. For instance, the Action Chunking Transformer (ACT) [^5] [^6], leveraging the Transformer architecture [^18], predicts chunks of actions $a_{t:t+K-1}$ conditioned on present observation and potentially a latent variable from a CVAE. Although action chunking improves temporal smoothness and reduces the effective horizon [^19] [^20], its temporal modeling is largely confined to short-term dependencies implicitly captured through the time aggregation of chunks while reasoning, potentially failing when resolving ambiguities requires longer context [^5]. Similarly, Diffusion Policy [^7] [^8] [^9] [^10], while adept at capturing complex, multimodal action distributions [^7] [^21], commonly conditions the diffusion process on the present observation or a short history, limiting its capacity for tasks requiring long-term memory [^7] [^21]. While extensions like Diff-Control [^21] introduce forms of statefulness, they differ fundamentally from MTIL’s direct use of a recurrent SSM state to encode the full task history. Other techniques, including Implicit BC [^22] and Energy-Based Models [^23], also often operate primarily on the current state.

### II-B Temporal and Sequential Imitation Learning

The inadequacy of the strict Markov assumption has long motivated efforts to incorporate temporal context. Early explorations employed Recurrent Neural Networks (RNNs) like LSTMs [^1] [^4] [^12] [^24]. However, these architectures face challenges with long-term dependencies due to vanishing gradients [^24]. Furthermore, practical implementations often resorted to fixed history windows and periodic state resets (e.g., sequence lengths of 10-50 in Robomimic [^12] [^25]), precluding the capture of full trajectory history. More recently, Transformer-based models (e.g., BeT [^26], RT-1 [^27] [^28], OPTIMUS [^29], ICRT [^30], Baku [^27] [^28], MDT [^31]) have become prominent, utilizing self-attention to model sequence correlations. However, the $O(L^{2})$ computational complexity of attention imposes practical limits on the size of the context window [^1] [^24], hindering their ability to efficiently process entire long trajectories. Even recurrent variants like Transformer-XL [^32], while theoretically capable of processing long sequences, still rely on the computationally intensive attention mechanism. Distinct strategies for managing long horizons involve temporal abstraction. Hierarchical Imitation Learning [^14] [^15] [^33] [^34] [^35] and Skill Chaining [^14] [^36] [^37] decompose tasks, learning policies over skills or sub-goals. Waypoint-based methods like AWE [^19] or primitive-based approaches like PRIME [^15] operate at higher levels of abstraction. While effective, these approaches fundamentally differ from MTIL, which aims to directly model the complete low-level observation-action sequence history, potentially offering robustness against issues like error propagation in skill chaining [^14] [^37].

### II-C State Space Models (SSMs) and Mamba in Robotics

State Space Models (SSMs) represent a compelling paradigm for sequence modeling, defined by their recurrent hidden state dynamics [^16] [^17] [^38] [^39] [^40] [^41] [^42]. Mamba [^16] marked a significant advancement, introducing input-dependent parameters ($\mathbf{A},\mathbf{B},\mathbf{C},\Delta$) via a selective scan mechanism. This allows Mamba models to dynamically focus on relevant sequence information while maintaining the linear time complexity characteristic of SSMs, synergizes the capacity for long-range dependency modeling, akin to Transformers, with the efficient recurrent updates reminiscent of RNNs, yet sidesteps the quadratic scaling bottlenecks of the former [^1] [^24] and the gradient propagation issues of the latter [^24]. achieving strong empirical results [^16].The robotics community has begun investigating Mamba’s potential [^16] [^17] [^43] [^44]. For instance, MaIL [^17] employed Mamba as an imitation learning backbone, showing promise particularly in low-data regimes [^16]. Mamba Policy [^45] [^46] integrated Mamba structures within diffusion models to enhance efficiency, while X-IL [^44] explored Mamba within a modular IL framework. While these works adeptly leverage Mamba’s sequence processing power, MTIL distinguishes itself through its core premise: harnessing the step-updated recurrent state $h_{t}$ as an explicit, dynamically built representation of the entire observation history.This approach, tightly coupled with its sequential training methodology, directly overcomes the temporal ambiguity challenges inherent in Markovian assumptions common in imitation learning.

## III Mamba Temporal Imitation Learning (MTIL)

We introduce Mamba Temporal Imitation Learning (MTIL), a novel imitation learning framework designed to overcome the limitations of the Markov assumption by leveraging the full history of observations encoded within the recurrent state of an advanced State Space Model (SSM) architecture.

![Refer to caption](https://arxiv.org/html/2505.12410v3/x1.png)

Figure 1: Overview of the Mamba Temporal Imitation Learning (MTIL) architecture. Multi-modal inputs (images via DINOv2, state) are fused and processed by sequential Mamba-2 blocks, updating the recurrent state h t h\_{t} which encodes history. At each step across the entire trajectory, MTIL predicts an action chunk a ^ + K − 1 \\hat{a}\_{t:t+K-1} (current plus K-1 future steps). This is supervised via L2 loss against ground truth actions a\_{t:t+K-1} from the demonstration (using last action for padding when near trajectory end). The historical context embedded in enables temporally coherent, long-horizon action generation.

### III-A Background and Motivation

Standard imitation learning often assumes a Markov Decision Process (MDP), learning reactive policies $\pi(a_{t}|o_{t})$ via Behavioral Cloning. However, observational ambiguity fundamentally renders many sequential tasks as Partially Observable MDPs (POMDPs) [^1], where the optimal policy necessitates conditioning on the full history $H_{t}=(o_{1},a_{1},...,o_{t})$. Theoretically, this history is captured by the belief state $b_{t}=P(s_{t}|H_{t})$, dictating the optimal policy $\pi^{*}(a_{t}|b_{t})$ [^47].

Directly computing or representing the belief state $b_{t}$ is generally intractable. This motivates learning a compressed history representation $h_{t}\approx b_{t}$ using recurrent models. This aligns with the core ideas of both World Models, which learn a predictive latent state of the world, and Dynamical Systems (DS) approaches to control, which rely on an evolving internal state. State Space Models (SSMs) like Mamba [^16] offer a particularly compelling synthesis of these ideas. They provide a structured recurrent update $h_{t}=f(h_{t-1},x_{t})$ (where $x_{t}$ encodes $o_{t}$) with linear time complexity $O(L)$. This efficiency is the critical enabler for tractably encoding the long sequences required for full history representation, overcoming the computational barriers of prior architectures [^48].

Our motivation for MTIL stems from leveraging Mamba’s state $h_{t}$ as this potent, efficiently computed representation of the full history. We view $h_{t}$ as the state of a learned, implicit dynamical system that acts as an action-oriented world model. By conditioning actions on both the current observation $o_{t}$ and this history-infused state $h_{t}$, MTIL learns a non-Markovian policy:

$$
\hat{a}_{t:t+K-1}\approx\pi(o_{t},h_{t})
$$

thereby directly addressing the core challenge of decision-making under ambiguity in POMDP-structured imitation learning by effectively utilizing the entire history.

### III-B Leveraging Full Trajectory History with Mamba-2

MTIL employs Mamba-2 [^49], an advanced structured State Space Model (SSM) notable for its refined selective mechanism and efficiency [^16]. Improving upon Mamba, Mamba-2 enhances hardware utilization and clarifies theoretical links to attention while retaining dynamic context adaptation via input-dependent parameters [^49]. Its core lies in the discretized SSM recurrence governing the hidden state $h_{t}\in\mathbb{R}^{N}$ evolution based on input $x_{t}$ (derived from observation $o_{t}$):

$$
h_{t}=\bar{\mathbf{A}}_{t}h_{t-1}+\bar{\mathbf{B}}_{t}x_{t}
$$
 
$$
y_{t}=\mathbf{C}_{t}h_{t}
$$

Crucially, the input-dependent parameters $(\Delta_{t},\bar{\mathbf{B}}_{t},\mathbf{C}_{t})=f(x_{t})$ enable selective state dynamics. This allows the model to learn a highly non-linear dynamical system where the system matrices themselves adapt based on the current input. This selective mechanism, combined with inherent linear-time complexity $O(L)$, facilitates learning from complete trajectory histories—a significant advantage over quadratic-complexity $O(L^{2})$ attention mechanisms. The resulting state $h_{t}$ acts as a dynamic summary of the salient history $(x_{1},...,x_{t-1})$, furnishing the requisite context even when the instantaneous observation $x_{t}$ is ambiguous. The MTIL policy leverages this directly:

$$
\pi(\hat{a}_{t:t+K-1}|x_{t},h_{t})
$$

By conditioning predictions $\hat{a}_{t:t+K-1}$ on both the current input $x_{t}$ and the comprehensive historical summary encoded in $h_{t}$, MTIL effectively transcends the limitations inherent in Markovian approaches, enabling sequential decision-making grounded in the full trajectory context.

Algorithm 1 MTIL Training (Sequential Step-based)

  Expert trajectories $\mathcal{D}=\{\tau_{i}\}$, $\tau_{i}=(o_{1},a_{1},...,o_{T_{i}},a_{T_{i}})$, MTIL Policy $\pi_{\theta}$, initialized parameters $\theta$, Loss $\mathcal{L}$ (MSE), Chunk size $k$, Optimizer $Opt$

  Initialize policy parameters $\theta$

  for each training epoch do

   for each trajectory $\tau_{i}\in\mathcal{D}$ do

    Initialize hidden state $h_{0}$, trajectory loss $\mathcal{L}_{traj}\leftarrow 0$

    for $t=0$ to $T_{i}-1$ do

      $x_{t}=\text{Encoder}(o_{t})$

     Predict $(\hat{a}_{t:t+k-1},h_{t})=\pi_{\theta}.\text{step}(x_{t},h_{t-1})$

     Get $a_{t:t+k-1}$ from $\tau_{i}$

     Calculate step loss: $\mathcal{L}_{t}=\mathcal{L}(\hat{a}_{t:t+k-1},a_{t:t+k-1})$

     Accumulate loss: $\mathcal{L}_{traj}+=\mathcal{L}_{t}$

    end for

     $Opt.\text{step}()$ {Update $\theta$ }

   end for

  end for

  return Trained policy $\pi_{\theta}$.

Algorithm 2 MTIL Inference (with Action Chunking and Temporal Aggregation)

  Trained policy $\pi_{\theta}$, Initial observation $o_{0}$, Chunk size $k$, Max steps $T_{max}$, Exponential aggregation weights $W$

  Initialize hidden state $h_{0}$, prediction buffer $B$.

  for $t=0$ to $T_{max}-1$ do

    $x_{t}=\text{Encoder}(o_{t})$

   Predict $(\hat{a}_{t:t+k-1},h_{t})=\pi_{\theta}.\text{step}(x_{t},h_{t-1})$

   Store prediction $\hat{a}_{t:t+k-1}$ in buffer $B$

   Aggregate Action for step $t$:

      Get predictions for step $t$ from $B$: $P_{t}=\{\hat{a}_{j:j+k-1}[t-j]\mid j\leq t<j+k\text{ and }\hat{a}_{j:j+k-1}\in B\}$

      Compute final action: $a_{t}^{\text{final}}=\text{WeightedAverage}(P_{t},W)$

   Execute action $a_{t}^{\text{final}}$

  end for

### III-C MTIL Training and Inference

MTIL enables imitation learning across complete expert trajectories, utilizing the architecture outlined in Figure 1. Distinctively, MTIL employs a sequential training procedure (Algorithm 1). This step-wise paradigm, leveraging Mamba’s recurrent ‘step‘ function, is fundamental to efficiently encoding arbitrarily long trajectories from high-dimensional observations (e.g., images) within feasible memory constraints—a key departure from parallel window-based approaches.A naive implementation of this sequential process would be limited to a batch size of one, posing a challenge for training efficiency. To address this, we introduce a novel batch-parallel training scheme. Instead of processing a single trajectory, our method processes a batch of $B$ trajectories simultaneously. At each timestep $t$, the model takes a batch of observations, updates their respective hidden states and computes the loss concurrently. This approach preserves the crucial temporal integrity within each trajectory while fully leveraging the parallel processing power of modern GPUs, making MTIL’s training time competitive with highly-parallelizable Markovian methods.During training, at each timestep $t$, the policy receives the observation embedding $x_{t}$, updates its history-encoding state from $h_{t-1}$ to $h_{t}$, and predicts an action chunk $\hat{a}_{t:t+K-1}$. Learning proceeds by minimizing the Mean Squared Error (MSE) against the ground truth actions $a_{t:t+K-1}$. During inference (Algorithm 2), the trained policy operates autoregressively, using the same ‘step‘ function to update its state and predict actions. For enhanced stability, temporal aggregation strategies [^19] [^20] [^5] are applied, averaging over predictions from overlapping action chunks to produce a smoother final action.

## IV Experimental Results

We conducted extensive experiments to evaluate the performance of MTIL across various benchmarks and real-world scenarios.All results stem from a rigorous protocol over three random seeds (100, 200, 300), with 50 roll-outs and the checkpoint for each run selected based on the minimum validation loss or as the final success rate for Robomimic.

### IV-A ACT benchmark

TABLE I: Success Rates (%) on the ACT Benchmark. Results are averaged over 3 seeds, All experiments run on a single RTX 4090.

<table><tbody><tr><th>Method</th><td>History Length</td><td>Cube Transfer (%)</td><td>Bimanual Insertion (%)</td></tr><tr><th>ACT <sup><a href="#fn:5">5</a></sup></th><td>1 (Markovian)</td><td>90.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.0</td><td>50.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.5</td></tr><tr><th>Diffusion Policy <sup><a href="#fn:7">7</a></sup></th><td>1 (Markovian)</td><td>72.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.6</td><td>28.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.2</td></tr><tr><th>Diffusion Policy <sup><a href="#fn:7">7</a></sup></th><td>10</td><td>78.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.5</td><td>32.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 4.1</td></tr><tr><th>Diffusion Policy <sup><a href="#fn:7">7</a></sup></th><td>20</td><td>80.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.2</td><td>34.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.8</td></tr><tr><th>Diffusion Policy <sup><a href="#fn:7">7</a></sup></th><td>30</td><td>82.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.1</td><td>36.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.5</td></tr><tr><th>Diffusion Policy <sup><a href="#fn:7">7</a></sup></th><td>40</td><td colspan="2">OOM</td></tr><tr><th>Transformer-XL <sup><a href="#fn:32">32</a></sup></th><td>Full (400)</td><td>86.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.5</td><td>42.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 4.0</td></tr><tr><th>MTIL (10-step)</th><td>10</td><td>92.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.5</td><td>56.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.5</td></tr><tr><th>MTIL (Full)</th><td>Full (400)</td><td>100.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.0</td><td>84.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.1</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2505.12410v3/x2.png)

(a) Learning Curves

We evaluated MTIL on the ACT benchmark to dissect its performance, efficiency, and learning dynamics against prominent architectural paradigms. The results, which juxtapose success rates with architectural choices and history lengths, are presented in Table I. The findings decisively establish MTIL’s superiority. On both tasks, MTIL (Full) achieves a perfect or near-perfect success rate, drastically outperforming all baselines. The learning curves in Figure 2(a) illuminate this outcome, showing that MTIL not only attains a higher performance ceiling but also converges significantly faster, indicating a more stable and sample-efficient learning process. Conversely, the performance of attention-based models reveals a critical insight: naively incorporating history is an inefficient, and ultimately, a computationally infeasible strategy. While Diffusion Policy’s success rate scales with history length, it remains notably inferior to the simple Markovian ACT baseline and incurs a prohibitive computational cost, culminating in an Out-of-Memory (OOM) error. Even Transformer-XL, theoretically capable of full-history processing, fails to match ACT, reinforcing the hypothesis that attention is a suboptimal inductive bias for modeling the continuous dynamics of physical interaction. Furthermore, the backbone ablation in Figure 2(b) confirms our advantage is architectural. MTIL, even with an identical ResNet18 backbone [^50], substantially outperforms ACT. The use of a stronger DINOv2 backbone [^51] further widens this gap.This proves MTIL’s success stems from a fundamentally superior paradigm: a computationally efficient recurrent architecture that is intrinsically better suited to capturing the temporal fabric of the physical world.

### IV-B LIBERO Benchmark

On LIBERO’s [^52] EWC [^53] lifelong learning benchmark (using standard ResNet/ViT backbones matching baselines for fair comparison), MTIL demonstrates strong lifelong learning when leveraging full history (-M (Full), Table II). It consistently achieves superior forward transfer (FWT $\uparrow$), reduced forgetting (NBT $\downarrow$), and higher overall performance (AUC $\uparrow$) compared to baselines and short-history (10-step) MTIL, which performs similarly to Transformers (-T). This advantage of full-history encoding, while notable across all categories, becomes particularly pronounced in LIBERO-LONG. Here, the performance margin over limited-context methods widens substantially, offering compelling evidence for the critical role of complete history as task horizons extend.

TABLE II: Lifelong Learning Performance on LIBERO (EWC Strategy).

<table><tbody><tr><th></th><td colspan="3">LIBERO-LONG</td><td colspan="3">LIBERO-SPATIAL</td></tr><tr><th>Policy Arch.</th><td>FWT(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td>NBT(<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math>)</td><td>AUC(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td>FWT(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td>NBT(<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math>)</td><td>AUC(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td></tr><tr><th>ResNet-RNN</th><td>0.02 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.04 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.00 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.14 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.03 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td></tr><tr><th>ResNet-T</th><td>0.13 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.03 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.33 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ResNet-M (10-step)</th><td>0.14 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.20 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.03 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.24 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.30 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ResNet-M (Full)</th><td>0.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.08 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.08 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.28 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.17 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ViT-T</th><td>0.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.09 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.01 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.32 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.48 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ViT-M (10-step)</th><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.01 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.33 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.45 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ViT-M (Full)</th><td>0.19 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.35 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.15 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th></th><td colspan="3">LIBERO-OBJECT</td><td colspan="3">LIBERO-GOAL</td></tr><tr><th>Policy Arch.</th><td>FWT(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td>NBT(<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math>)</td><td>AUC(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td>FWT(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td>NBT(<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math>)</td><td>AUC(<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td></tr><tr><th>ResNet-RNN</th><td>0.17 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.16 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ResNet-T</th><td>0.56 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.69 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.16 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.32 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.45 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.07 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ResNet-M (10-step)</th><td>0.50 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.39 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.15 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>0.31 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.42 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.07 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ResNet-M (Full)</th><td>0.55 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.36 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.17 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.30 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.11 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ViT-T</th><td>0.57 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.64 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>0.32 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.48 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.07 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ViT-M (10-step)</th><td>0.56 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.60 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.33 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.45 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.08 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr><tr><th>ViT-M (Full)</th><td>0.58 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.18 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.25 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>0.34 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>0.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>0.11 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td></tr></tbody></table>

FWT($\uparrow$): Forward Transfer, NBT($\downarrow$): Negative Backward Transfer (should be Backward Transfer, if it’s negative it’s good), AUC($\uparrow$): Area Under Curve. EWC strategy results averaged over 3 seeds (100, 200, 300) at 50 epochs. Baselines from [^52]. Short-history (10-step,similar performance for 20/50 steps) and full-history result shown.

### IV-C Robomimic (Vision-based Policy)

TABLE III: Behavior Cloning Benchmark (Visual Policy) on Robomimic. As per the original dataset, results are reported as final success rates.

<table><thead><tr><th></th><th colspan="2">Lift</th><th colspan="2">Can</th><th colspan="2">Square</th><th colspan="2">Transport</th><th>ToolHang</th></tr></thead><tbody><tr><td></td><td>ph</td><td>mh</td><td>ph</td><td>mh</td><td>ph</td><td>mh</td><td>ph</td><td>mh</td><td>ph</td></tr><tr><th>LSTM-GMM [29]</th><th>1.00</th><th>1.00</th><th>1.00</th><th>0.98</th><th>0.82</th><th>0.64</th><th>0.88</th><th>0.44</th><th>0.68</th></tr><tr><td>IBC [12]</td><td>0.94</td><td>0.39</td><td>0.08</td><td>0.00</td><td>0.03</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><th>DiffusionPolicy-C</th><th>1.00</th><th>1.00</th><th>1.00</th><th>1.00</th><th>0.98</th><th>0.98</th><th>1.00</th><th>0.89</th><th>0.95</th></tr><tr><td>DiffusionPolicy-T</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.94</td><td>0.98</td><td>0.73</td><td>0.76</td></tr><tr><th>MTIL (10-step)</th><th>1.00</th><th>1.00</th><th>1.00</th><th>0.99</th><th>0.87</th><th>0.65</th><th>0.92</th><th>0.52</th><th>0.72</th></tr><tr><td>MTIL (Full)</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.96</td><td>1.00</td><td>0.91</td><td>0.97</td></tr></tbody></table>

To assess MTIL’s ability to handle high-dimensional visual inputs, we evaluated it on the vision-based Robomimic tasks [^12]. As shown in Table III, MTIL (Full) significantly outperforms all baselines, including the strong DiffusionPolicy variants. Notably, MTIL (10-step) offers only a marginal improvement over the LSTM-GMM baseline, highlighting that a short-history SSM is insufficient. The substantial performance gain of MTIL (Full) underscores its superior capability in leveraging full spatio-temporal context from visual data. This can be attributed to its nature as a learned dynamical system; the recurrent state $h_{t}$ acts as an implicit world model, tracking not just object locations but also their latent physical states (e.g., momentum, contact stability) over time, which is crucial for complex manipulation.

### IV-D Real-World Dual-Arm Tasks

![Refer to caption](https://arxiv.org/html/2505.12410v3/x4.png)

Figure 3: Dual UR3 experimental setup with four cameras (Top: Kinect; Side: D435i; Wrists: D405) and custom grippers.

![Refer to caption](https://arxiv.org/html/2505.12410v3/x5.png)

TABLE IV: Sequential Insertion Success Rates (%), averaged over 50 roll-outs.

![Refer to caption](https://arxiv.org/html/2505.12410v3/x6.png)

TABLE V: Coordinated Pouring Success Rates (%), averaged over 50 roll-outs.

To validate MTIL in complex physical environments, we designed challenging tasks on a dual UR3 platform equipped with custom 2-finger grippers and four cameras providing multi-view observations (Figure 3). We compare MTIL (using DINOv2 backbone and full history) against ACT trained on identical demonstration data (100 demos per task). All real-world results are averaged over 50 evaluation roll-outs for the best checkpoint from each of the 3 seeds.

#### Sequential Insertion Task.

We designed this task (visualized in Figure 4) specifically to challenge Markovian policies by requiring long-term memory, a scenario where SOTA methods like ACT often fail. The four stages involve: (1) Left arm grasps Tube1, (2) Left passes Tube1 to Right arm, (3) Right arm inserts Tube1 into Tube2, (4) Right arm inserts Tube1 into Tube3. Critically, executing Stage 3 correctly necessitates recalling the completion of previous stages, as intermediate observations can be ambiguous. Table IV details the stage-wise success rates. MTIL, leveraging its full history state, successfully completes the entire sequence with high probability. In stark contrast, ACT, reliant on immediate context, is confounded by the temporal ambiguity, as the observations after completing Stage 2 can be identical with completing Stage 3, making it indistinguishable for policies relying solely on current or short-term history. AS a result, it frequently attempts Stage 4 directly after Stage 2, failing to execute the required sequence correctly and resulting in zero success for completing Stage 3, Stage 4, and the overall task. This outcome underscores the limitations of short-history approaches and validates the imperative of encoding complete history for reliably executing temporally complex manipulation sequences.

#### Coordinated Pouring Task.

This task (Figure 5) assesses precise bimanual coordination over a longer sequence: (1) Left arm grasps Tube1, (2) Left passes Tube1 to Right arm, (3) Left arm grasps Tube2, (4) Right arm pours water from Tube1 into Tube2. While less susceptible to the specific ambiguity of the insertion task, it still requires accurate, temporally coordinated actions. Table V (within Figure 5) shows that although both methods achieve non-zero success, MTIL consistently outperforms ACT across the stages, resulting in a higher overall success rate and exhibiting notably smoother execution trajectories.

## V Conclusion

The trajectory of intelligence is intrinsically linked to the capacity for memory – the ability to weave the tapestry of past experiences into the fabric of present action. This work confronts a central limitation in contemporary imitation learning: the prevalent reliance on the Markovian assumption, which often reduces complex sequential behaviors to mere reactions to the immediate sensory world. We introduced Mamba Temporal Imitation Learning (MTIL), a new paradigm that embraces the power of memory by leveraging the recurrent state dynamics inherent within the Mamba architecture. We posit that MTIL represents a practical and powerful synthesis of concepts from World Models and Dynamical Systems. By encoding the full history of observations into a compressed, evolving state representation, MTIL learns an implicit, action-oriented world model. This comprehensive temporal context allows MTIL to effectively disambiguate perception and unlock the execution of intricate, state-dependent sequential tasks previously challenging for established methods. Our findings not only showcase the significant performance and efficiency gains afforded by MTIL but, more profoundly, underscore the essential role of history in bridging the gap between perception and intelligent action. By demonstrating the efficacy of SSMs in capturing the long flow of time in a computationally feasible manner, this work illuminates a promising pathway towards building robotic agents capable of deeper understanding and more sophisticated interaction with the world.

## Acknowledgments

This work was supported by the Joint Funds of the National Natural Science Foundation of China (Grant No. U222A20208), the Natural Science Foundation Innovation Group Project of Hubei Province (Grant No. 2022CFA018), and the Key Research and Development Program of Guangdong Province (Grant No. 2022B0202010001-2).

[^1]: T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, and J. Peters, “An algorithmic perspective on imitation learning,” *Foundations and Trends® in Robotics*, vol. 7, no. 1–2, pp. 1–179, 2018.

[^2]: B. D. Argall, S. Chernova, M. Veloso, and B. Browning, “A survey of robot learning from demonstration,” *Robotics and autonomous systems*, vol. 57, no. 5, pp. 469–483, 2009.

[^3]: S. Schaal, “Learning from demonstration,” *Advances in neural information processing systems*, vol. 9, 1997.

[^4]: D. A. Pomerleau, “Alvinn: An autonomous land vehicle in a neural network,” in *Advances in neural information processing systems*, 1989.

[^5]: T. Z. Zhao, V. Kumar, L. Pinto, A. Gupta, and Z. Fu, “Learning fine-grained bimanual manipulation with low-cost hardware,” *Robotics: Science and Systems (RSS)*, 2023.

[^6]: T. Z. Zhao, J. Tompson, D. Driess, P. Florence, S. K. S. Ghasemipour, C. Finn, and A. Wahid, “Aloha unleashed: A simple recipe for robot dexterity,” in *8th Annual Conference on Robot Learning (CoRL)*, 2024.

[^7]: C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in *Robotics: Science and Systems (RSS)*, 2023.

[^8]: J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840–6851, 2020.

[^9]: Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy,” *CoRR*, 2024.

[^10]: T. Pearce, T. Rashid, A. Kanervisto, D. Bignell, M. Sun, R. Georgescu, S. V. Macua, S. Z. Tan, I. Momennejad, K. Hofmann *et al.*, “Imitating human behaviour with diffusion models,” in *The Eleventh International Conference on Learning Representations*, 2023.

[^11]: S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in *Proceedings of the fourteenth international conference on artificial intelligence and statistics*. JMLR Workshop and Conference Proceedings, 2011, pp. 627–635.

[^12]: A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín, “What matters in learning from offline human demonstrations for robot manipulation,” in *Conference on Robot Learning (CoRL)*. PMLR, 2021, pp. 950–961.

[^13]: L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, “Decision transformer: Reinforcement learning via sequence modeling,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 15 084–15 097.

[^14]: C. Lynch, M. Khansari, T. Xiao, V. Kumar, J. Tompson, S. Levine, and P. Sermanet, “Learning latent plans from play,” *Conference on Robot Learning (CoRL)*, pp. 1088–1103, 2020.

[^15]: T. Gao, S. Nasiriany, H. Liu, Q. Yang, and Y. Zhu, “Prime: Scaffolding manipulation tasks with behavior primitives for data-efficient imitation learning,” *IEEE Robotics and Automation Letters*, 2024.

[^16]: A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” *arXiv preprint arXiv:2312.00752*, 2023.

[^17]: X. Jia, Q. Wang, A. Donat, B. Xing, G. Li, H. Zhou, O. Celik, D. Blessing, R. Lioutikov, and G. Neumann, “Mail: Improving imitation learning with selective state space models,” in *8th Annual Conference on Robot Learning (CoRL)*, 2024.

[^18]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” *Advances in neural information processing systems*, vol. 30, 2017.

[^19]: L. X. Shi, A. Sharma, T. Z. Zhao, and C. Finn, “Waypoint-based imitation learning for robotic manipulation,” in *Conference on Robot Learning*. PMLR, 2023, pp. 2195–2209.

[^20]: X. Zhang, Y. Liu, H. Chang, L. Schramm, and A. Boularias, “Autoregressive action sequence learning for robotic manipulation,” *IEEE Robotics and Automation Letters*, vol. 10, no. 5, pp. 4898–4905, 2025.

[^21]: X. Liu, Y. Zhou, F. Weigend, S. Sonawani, S. Ikemoto, and H. B. Amor, “Diff-control: A stateful diffusion-based policy for imitation learning,” in *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2024, pp. 7453–7460.

[^22]: P. Florence, C. Lynch, A. Zeng, O. Lee, J. Tompson, V. Kumar, A. Herzog, J. Tan, and K. Bousmalis, “Implicit behavioral cloning,” in *Conference on Robot Learning*. PMLR, 2022, pp. 154–167.

[^23]: M. A. Bashiri, B. Ziebart, and X. Zhang, “Distributionally robust imitation learning,” *Advances in neural information processing systems*, vol. 34, pp. 24 404–24 417, 2021.

[^24]: M. Beck, K. Pöppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter, “xlstm: Extended long short-term memory,” *Advances in Neural Information Processing Systems*, vol. 37, pp. 107 547–107 603, 2025.

[^25]: A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox, “Mimicgen: A data generation system for scalable robot learning using human demonstrations,” in *7th Annual Conference on Robot Learning (CoRL)*, 2023.

[^26]: N. M. Shafiullah, Z. Cui, A. A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning $k$ modes with one stone,” *Advances in neural information processing systems (NeurIPS)*, vol. 35, pp. 22 955–22 968, 2022.

[^27]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Ho *et al.*, “Rt-1: Robotics transformer for real-world control at scale,” *arXiv preprint arXiv:2212.06817*, 2022.

[^28]: S. Haldar, Z. Peng, and L. Pinto, “Baku: An efficient transformer for multi-task policy learning,” in *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024.

[^29]: M. Dalal, A. Mandlekar, C. R. Garrett, A. Handa, R. Salakhutdinov, and D. Fox, “Imitating task and motion planning with visuomotor transformers,” in *Conference on Robot Learning (CoRL)*, 2023.

[^30]: L. Fu, H. Huang, G. Datta, L. Y. Chen, W. C.-H. Panitch, F. Liu, H. Li, and K. Goldberg, “In-context imitation learning via next-token prediction,” in *NeurIPS 2024 Workshop on Open-World Agents*, 2024.

[^31]: M. Reuss, Ö. E. Yagmurlu, F. Wenzel, and R. Lioutikov, “Multimodal diffusion transformer: Learning versatile behavior from multimodal goals,” *CoRR*, 2024.

[^32]: G. Tianci, “Transformer-xl for long sequence tasks in robotic learning from demonstrations,” *arXiv preprint arXiv:2405.15562*, 2024.

[^33]: Y. Zhu, P. Stone, and Y. Zhu, “Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation,” *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 4126–4133, 2022.

[^34]: A. Gupta, V. Kumar, C. Lynch, S. Levine, and K. Hausman, “Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning,” in *Conference on Robot Learning (CoRL)*. PMLR, 2019, pp. 1001–1013.

[^35]: W. Mao, W. Zhong, Z. Jiang, D. Fang, Z. Zhang, Z. Lan, H. Li, F. Jia, T. Wang, H. Fan *et al.*, “Robomatrix: A skill-centric hierarchical framework for scalable robot task planning and execution in open-world,” *arXiv preprint arXiv:2412.00171*, 2024.

[^36]: Y. Lee, J. J. Lim, A. Anandkumar, and Y. Zhu, “Adversarial skill chaining for long-horizon robot manipulation via terminal state regularization,” in *5th Annual Conference on Robot Learning (CoRL)*, 2021.

[^37]: Z. Chen, Z. Ji, J. Huo, and Y. Gao, “Scar: Refining skill chaining for long-horizon robotic manipulation via dual regularization,” *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37, pp. 111 679–111 714, 2024.

[^38]: P. Bevanda, M. Beier, A. Capone, S. G. Sosnowski, S. Hirche, and A. Lederer, “Koopman-equivariant gaussian processes,” in *The 28th International Conference on Artificial Intelligence and Statistics*, 2025.

[^39]: T. Fernando and M. Darouach, “Existence and design of target output controllers,” *IEEE Transactions on Automatic Control*, 2025.

[^40]: J. Luo, J. Cheng, X. Tang, Q. Zhang, B. Xue, and R. Fan, “Mambaflow: A novel and flow-guided state space model for scene flow estimation,” *arXiv preprint arXiv:2502.16907*, 2025.

[^41]: J. Du, Y. Sun, Z. Zhou, P. Chen, R. Zhang, and K. Mao, “Mambaflow: A mamba-centric architecture for end-to-end optical flow estimation,” *arXiv preprint arXiv:2503.07046*, 2025.

[^42]: K. Zeng, H. Shi, J. Lin, S. Li, J. Cheng, K. Wang, Z. Li, and K. Yang, “Mambamos: Lidar-based 3d moving object segmentation with motion-aware state space model,” in *Proceedings of the 32nd ACM International Conference on Multimedia*, 2024, pp. 1505–1513.

[^43]: T. Tsuji, “Mamba as a motion encoder for robotic imitation learning,” *IEEE Access*, 2025.

[^44]: X. Jia, A. Donat, X. Huang, X. Zhao, D. Blessing, H. Zhou, H. Zhang, H. A. Wang, Q. Wang, R. Lioutikov *et al.*, “X-il: Exploring the design space of imitation learning policies,” *arXiv preprint arXiv:2502.12330*, 2025.

[^45]: J. Cao, Q. Zhang, J. Sun, J. Wang, H. Cheng, Y. Li, J. Ma, Y. Shao, W. Zhao, G. Han *et al.*, “Mamba policy: Towards efficient 3d diffusion policy with hybrid selective state models,” *arXiv preprint arXiv:2409.07163*, 2024.

[^46]: M. Reuss, J. Pari, P. Agrawal, and R. Lioutikov, “Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning,” *arXiv preprint arXiv:2412.12953*, 2024.

[^47]: L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, “Planning and acting in partially observable stochastic domains,” *Artificial intelligence*, vol. 101, no. 1-2, pp. 99–134, 1998.

[^48]: A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,” in *International Conference on Learning Representations (ICLR)*, 2022.

[^49]: T. Dao and A. Gu, “Transformers are ssms: generalized models and efficient algorithms through structured state space duality,” in *Proceedings of the 41st International Conference on Machine Learning*, 2024, pp. 10 041–10 071.

[^50]: K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

[^51]: M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Pasqualini, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[^52]: B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” in *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2023.

[^53]: J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska *et al.*, “Overcoming catastrophic forgetting in neural networks,” in *Proceedings of the national academy of sciences (PNAS)*, vol. 114. National Acad Sciences, 2017, pp. 3521–3526.