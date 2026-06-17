Zhiyuan Guan <sup>1</sup>  Jianshu Hu <sup>1</sup>  Han Fang <sup>1</sup>  Yunpeng Jiang <sup>1</sup>  
Yize Huang <sup>1</sup>   Shujia Li <sup>1</sup>   Xiao Li <sup>1</sup>   Yutong Ban <sup>1</sup>  
<sup>1</sup> Shanghai Jiao Tong University Corresponding author.

###### Abstract

Diffusion-based imitation learning has shown strong promise for robot manipulation. However, most existing policies condition only on the current observation or a short window of recent observations, limiting their ability to resolve history-dependent ambiguities in long-horizon tasks. To address this, we introduce DSSP, a history-conditioned Diffusion State Space Policy that enables efficient, full-history conditioning for robot manipulation. Leveraging the continuous sequence modeling properties of State Space Models (SSMs), our history encoder effectively compresses the entire observation stream into a compact context representation. To ensure this context preserves critical information regarding future state evolution, the encoder is optimized with a dynamics-aware auxiliary training objective. This high-level context representation is then seamlessly fused with recent state observations to form a hierarchical conditioning mechanism for action generation. Furthermore, to maintain architectural consistency and minimize GPU memory overhead, we also instantiate the diffusion backbone itself using an SSM. Extensive experiments across simulation benchmarks and real-world manipulation tasks show that DSSP achieves state-of-the-art performance with a significantly smaller model size, demonstrating superior efficiency of the hierarchical conditioning in capturing crucial information as the history length increases.

## 1 Introduction

Deploying robots in complex, unstructured environments requires policies capable of reasoning over multi-step tasks and high-dimensional sensory inputs. Imitation learning [^43] has emerged as a central paradigm for acquiring such skills from expert demonstrations. Among recent imitation learning approaches, diffusion policies [^7] have shown strong potential for modeling complex and multi-modal action distributions. Recent works further improve diffusion-based policies through expressive denoising backbones [^38], multi-modal conditioning representations [^42] [^27], and improved action generation formulations [^33]. However, most policies still rely on short-context conditioning, predicting future actions from the current observation or a short observation window.

This short-context conditioning becomes a critical bottleneck in long-horizon manipulation. In multi-stage tasks, visually similar observations may correspond to different task progress, such as whether an object has already been moved, whether a container has been filled, or which object was previously placed in a buffer location. The desired actions therefore depend not only on local visual details, but also on sparse historical events that reveal task progress and previous interactions. Without access to such history, policies may fail to disambiguate temporally aliased states, leading them to ignore completed subgoals or undo previous actions.

![Refer to caption](https://arxiv.org/html/2605.14598v2/x1.png)

Figure 1: The proposed DSSP leverages full-history context to resolve visual aliasing when history-blind baselines lose track of task progress, enabling consistent execution in long-horizon tasks. DSSP achieves superior success rates across both simulation tasks and real-world experiments.

Existing methods have attempted to address the limitations of short-context policies by adding temporal context. A direct solution extends the observation horizon or feeds longer observation histories into the policy, but this increases memory and inference cost and may introduce redundant visual inputs or spurious correlations [^44]. Attention-based and multi-frame VLA methods provide more flexible access to past observations, yet remain costly because they explicitly process additional context as the horizon grows [^25] [^18]. To reduce this overhead, recent works therefore explore more compact memory forms, including keyframes [^36], visual traces [^44], point tracks [^5], or recurrent latent states \[zhou\_mtil\_2025\]. However, compactness does not ensure task relevance: compressed memories may discard progress-critical events or focus on nuisance visual changes. These limitations motivate a streaming history mechanism that efficiently compresses long-horizon context while preserving events predictive of future state.

To fulfill these requirements, State Space Models (SSMs) offer a well-suited architecture for history encoding. Their recurrent formulation maintains a compact hidden state that is updated online as new observations arrive, enabling linear-time aggregation of long histories without re-encoding the entire past. Moreover, modern selective SSMs such as Mamba [^13] make the state update input-dependent, allowing the model to adaptively decide what to preserve or discard. This content-dependent memory update is particularly suitable for long-horizon manipulation, where visually similar observations may correspond to different task stages and require different actions. Building on this insight, we propose DSSP, a full-history encoding diffusion state-space policy for long-horizon manipulation. Following the use of SSMs in diffusion-based robot policies [^3] [^19] [^40], DSSP instantiates the action denoiser with an SSM backbone, but conditions it on a compact online memory compressed from the full multi-modal observation stream together with immediate state information. A dynamics-aware auxiliary objective encourages this memory to preserve cues predictive of future states. We further decouple task conditioning from diffusion-step modulation by injecting observation-derived conditions as a prefix and timestep information through Adaptive Layer Normalization (AdaLN) [^30].

Our contributions are summarized as follows:

- Hierarchically conditioned state space policy. We design a policy framework instantiated with a state space model and a hierarchical conditioning mechanism. Specifically, learned context representations and immediate state representations are fused via prefix conditioning, while the diffusion timestep is decoupled and injected independently through AdaLN.
- Full-history context learning. We propose a causal state-space history encoder that maintains a compact context representation by recurrently integrating incoming multi-modal observations. Together with a dynamics-aware auxiliary objective, the encoder summarizes the full observation history while retaining task-relevant events predictive of future states.
- Comprehensive evaluation. We evaluate DSSP on extensive simulated and real-world long-horizon manipulation tasks, showing improved success rates, effective history summarization, perturbation robustness, and efficient inference with increasing history length.

## 2 Related Work

Imitation Learning for Robot Manipulation. Imitation learning enables policies to acquire skills from expert demonstrations. While early behavior cloning methods directly predict actions from observations, recent approaches improve closed-loop consistency by modeling temporally extended or structured actions, such as action chunks [^43] and multimodal action representations \[shafiullah\_behavior\_2022, lee\_behavior\_2024\]. Diffusion-based policies further generate continuous action trajectories via conditional denoising [^7], with 3D Diffusion Policy extending this framework to point-cloud observations [^42]. Recent variants improve diffusion policies in perception [^27], efficiency [^10] [^33], and generation quality [^38] [^28]. Our method follows this paradigm, using conditional denoising for action generation.

Long-Horizon and History-Aware Policy Learning. Most existing robot policies condition action prediction on the current observation or a short window of recent observations, which is often sufficient for short-horizon or near-Markovian tasks. In long-horizon manipulation, resolving temporal ambiguity requires observation history. However, naively stacking past observations often degrades performance due to redundancy and causal confusion \[haan\_causal\_2019, wen\_fighting\_2020, wen\_keyframe-focused\_2021, swamy\_sequence\_2023\]. Recent methods therefore explore more selective uses of history, such as regularizing long-context policies with past-token prediction \[torne\_learning\_2025\], selecting task-relevant keyframes from history \[mark\_bpp\_2026\], using demonstration trajectories as in-context prompts [^37], or compressing long observation histories into an evolving latent state \[gui\_seedpolicy\_2026, zhou\_mtil\_2025\]. In contrast, our method learns a compact, dynamics-aware history context that preserves future-relevant information for effective long-horizon policy learning.

State Space Models for Robot Policies. State space models have recently shown strong potential for sequence modeling by maintaining an evolving latent state over time. Mamba [^13] [^8] [^24] further improves this paradigm with selective state updates and hardware-aware parallel scan, enabling efficient long-sequence modeling. Inspired by these properties, recent robotic learning methods have adopted Mamba as a backbone for imitation learning and diffusion-based action generation [^29] [^19] [^3], or as a recurrent encoder for long observation histories in temporally ambiguous tasks \[tsuji\_mamba\_2025, zhou\_mtil\_2025\]. These findings demonstrate the effectiveness of Mamba for modeling robot trajectories and temporal dependencies. In this paper, we introduce a unified Mamba-based policy that integrates dynamics-aware history encoding with diffusion-based action generation.

## 3 Preliminaries

Problem Formulation. We formulate long-horizon 3D manipulation as a Partially Observable Markov Decision Process (POMDP), defined by the tuple $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{T},\Omega,\mathcal{O})$. At any time step $t$, the agent receives an observation $o_{t}\in\Omega$ generated from the unobserved true state $s_{t}\in\mathcal{S}$ via the observation function $\mathcal{O}(o_{t}\mid s_{t})=P(o_{t}\mid s_{t})$. The environment evolves according to transition dynamics $\mathcal{T}(s_{t+1}\mid s_{t},a_{t})=P(s_{t+1}\mid s_{t},a_{t})$ given action $a_{t}\in\mathcal{A}$. Since $o_{t}$ is insufficient to infer $s_{t}$, we define the interaction history as $h_{t}=(o_{0},a_{0},\dots,a_{t-1},o_{t})\in\mathcal{H}$, where $\mathcal{H}$ denotes the full history space. Our goal is to learn a history-dependent policy $\pi_{\theta}(a_{t}\mid h_{t})$ imitating an expert policy $\pi_{E}$ with expert trajectories $\mathcal{D}_{E}=\{\zeta_{1},\dots,\zeta_{N}\}$, where each trajectory is a sequence $\zeta=(o_{0},a_{0},\dots,o_{T})$. Due to the page limit, we include a detailed introduction to diffusion policy and SSM in Appendix A.

Observation Aliasing. A fundamental challenge in long-horizon manipulation is that the mapping $\mathcal{O}$ is often non-injective, resulting in observation aliasing.

###### Definition 1 (Observation Aliasing).

Observation aliasing occurs when two distinct histories $h_{t}^{1},h_{t}^{2}\in\mathcal{H}$ yield identical current observations $o_{t}^{1}=o_{t}^{2}$, but require different expert action distributions:

$$
P_{E}(a_{t}\mid h_{t}^{1})\neq P_{E}(a_{t}\mid h_{t}^{2}).
$$

Under these conditions, a purely reactive policy $\pi(a_{t}\mid o_{t})$ collapses distinct contexts into a suboptimal marginal distribution $P_{E}(a_{t}\mid o_{t})$. By processing the full history sequence using the aforementioned SSM backbone, our policy $\pi_{\theta}(a_{t}\mid h_{t})$ resolves this ambiguity (see section˜4.3 for analysis).

## 4 Method

![Refer to caption](https://arxiv.org/html/2605.14598v2/x2.png)

Figure 2: Overview of DSSP. DSSP summarizes past multi-modal observations into a compact context token using a state-space history encoder. A dynamics-aware auxiliary loss encourages this token to retain historical information predictive of future state evolution. The learned context token is then combined with recent state tokens as a hierarchical prefix condition for a state-space diffusion denoiser to generate future actions.

To effectively leverage history information within a robot manipulation policy, we propose diffusion state space policy (DSSP), a full-history conditioned diffusion policy (illustrated in Figure˜2). Our approach instantiates the diffusion model with an SSM backbone and a dual-level conditioning mechanism that integrates high-level context with low-level state representations. The context representation serves as a compact encoding of historical multi-modal observations. To ensure this representation captures temporal dependencies, we introduce an auxiliary dynamics-aware loss focused on future state prediction. Section˜4.1 introduces long-horizon context learning, including causal history encoding and dynamics-aware context learning; and Section˜4.2 formalizes the hierarchical conditioning mechanism and diffusion-based action generation policy.

### 4.1 Long-Horizon Context Learning

Long-horizon manipulation requires historical context to resolve task-level ambiguities; otherwise, policies may experience perceptual aliasing, such as getting trapped in loops during repetitive wiping tasks. Because existing conditioning approaches like observation stacking or keyframe extraction often suffer from computational redundancy or overlook subtle causal events, we introduce a learnable history encoder that compresses the full history into a compact, task-relevant context representation.

To this end, our design is guided by two key principles. First, we employ a causal SSM to efficiently process the streaming observation history and extract a temporally integrated memory representation. Second, to ensure that this representation preserves historical cues useful for future decisions, we shape the latent space with a dynamics-aware auxiliary objective.

Causal History Encoding. We first introduce how to build the context representation for history encoding. Let $o_{t}$ denote the multi-modal observation at timestep $t$, comprising visual inputs and robot proprioceptive states. We project this raw observation into a state representation $z_{t}$:

$$
z_{t}=E_{\mathrm{obs}}(o_{t}),
$$

where $E_{\mathrm{obs}}$ contains parallel visual and proprioceptive encoders. To capture long-horizon temporal dependencies, a causal history encoder $G$ processes the sequence of step-wise state representations $z_{1:t}$ into the temporally-integrated context representation $\tilde{z}_{1:t}$:

$$
\tilde{z}_{1:t}=G_{\psi}(z_{1:t}).
$$

A critical design choice is the architecture of the history encoder $G$, which must produce a compact yet effective history representation. This design must satisfy two primary criteria: (1) maintaining a scalable computation when processing extended temporal horizons, and (2) distilling a representation that selectively retains salient events rather than passively memorizing the entire observation stream.

To meet these requirements, we instantiate the history backbone using a State-Space Model (SSM) and define the context representation $c_{t}$ as the final output token of the encoded sequence:

$$
c_{t}=\tilde{z}_{t}.
$$

SSMs support streaming histories with linear-time complexity, and we use Mamba as the history encoder for input-dependent selective updates. This allows the encoder to filter redundant observations while preserving sparse task-relevant events, such as object-state transitions, contact changes, or subgoal completion. The resulting context $c_{t}$ serves as a compressed memory for action generation.

Dynamics-Aware Context Learning. Compressing the observation history into $c_{t}$ does not by itself guarantee that the representation preserves historical information relevant to future decisions. For long-horizon manipulation, the context representation should encode not only past observations, but also history-dependent cues that are predictive of future state evolution. To encourage this property, we introduce a dynamics-aware auxiliary objective applied to the context representation at each timestep. Given the context representation $c_{t}$ and the executed action $a_{t}$ at time $t$, we train a lightweight dynamics predictor $g_{\phi}$ to predict the next state representation:

$$
z_{t+1}=E_{\mathrm{obs}}(o_{t+1}),\qquad\hat{z}_{t+1}=g_{\phi}(c_{t},a_{t}).
$$

We supervise this prediction with a cosine similarity loss:

$$
\mathcal{L}_{\mathrm{dyn}}(\psi,\phi)=\mathbb{E}_{\zeta\sim\mathcal{D}_{E},t\sim[0,T-1]}\left[1-\cos\left(g_{\phi}(c_{t},a_{t}),\mathrm{sg}(z_{t+1})\right)\right],
$$

where $\mathrm{sg}(\cdot)$ denotes the stop-gradient operation, and the expectation is taken over expert trajectories $\zeta\sim\mathcal{D}_{E}$ and trajectory timesteps $t$. This objective encourages the context representation $c_{t}$ to retain action-relevant historical information by requiring it to support prediction of the next state under the executed action. The learned context representation therefore provides a more informative conditioning signal for downstream action generation.

### 4.2 Hierarchical Context-Conditioned Denoising

Given the learned history representation, the remaining question is how to inject it into a policy. Long-horizon manipulation requires both long-term task-progress information and recent local observations: the former resolves visual ambiguity across task stages, while the latter provides fine-grained control cues. Therefore, we propose a diffusion policy utilizing a hierarchical conditioning mechanism that integrates the context representation with immediate state observations. We organize these signals as a causal prefix to the noisy action sequence, allowing long-term context to contextualize recent observations before being propagated to the action tokens during denoising. For efficiency, we instantiate the diffusion backbone with a compact SSM, yielding a lightweight policy while maintaining a unified SSM-based architecture for both history encoding and action denoising.

Hierarchical Prefix Conditioning. To preserve long-term progress information and local control cues, we condition the denoising model on a hierarchical prefix. Let $x_{0}=\mathbf{a}_{t:t+H_{a}-1}$ denote the clean future action trajectory, and let $x_{\tau}$ denote its noisy version at diffusion step $\tau$. We construct the condition sequence as

$$
C_{t}=[c_{t},z_{t-N+1},\dots,z_{t}],
$$

where $c_{t}$ is the context representation produced by the history encoder, $z_{t-N+1:t}$ are the most recent state tokens, and $N$ is the local observation window size. We prepend $C_{t}$ to the noisy action sequence as a prefix condition. The resulting sequence is processed by the SSM denoising backbone:

$$
\hat{x}_{0}=f_{\theta}(x_{\tau},\tau,C_{t}).
$$

In this design, $c_{t}$ captures long-horizon progress and $z_{t}$ retains local geometric and proprioceptive details. Thus, the policy leverages historical context without losing manipulation precision.

Causal Action Denoising. The hierarchical prefix formulation casts action denoising as a causal prefix-conditioned sequence modeling. Ordering the sequence as $[c_{t},z_{t-N+1},\dots,z_{t},x_{\tau}]$ allows long-term context to contextualize recent states before propagating information to noisy action tokens. We instantiate $f_{\theta}$ with a Mamba backbone, whose recurrent selective state updates naturally match this left-to-right conditioning flow while providing linear scaling for iterative diffusion sampling.

Timestep-Decoupled Action Denoising. To provide a stable conditioning signal throughout iterative denoising, we decouple timestep modulation from prefix conditioning. Since the diffusion timestep $\tau$ describes the noise level of the action trajectory, we inject the timestep embedding through AdaLN only into the action tokens while keeping the prefix condition unchanged. This keeps $C_{t}$ as a stable representation of history and local observations throughout the denoising process. Meanwhile, the actions remain aware of the current noise level and can progressively refine the predicted action sequence. This design separates task conditioning from diffusion-step modulation.

Training. The policy is optimized with two objectives: the diffusion reconstruction loss for action generation and the dynamics-aware auxiliary loss for context learning. For an action window starting at time $t$, we construct the hierarchical condition $C_{t}$ causally from observations up to time $t$ and train the denoising model to predict the clean action trajectory:

$$
\mathcal{L}_{\mathrm{diff}}(\theta,\psi)=\mathbb{E}_{\zeta\sim\mathcal{D}_{E},t,\tau,\epsilon}\left[\left\|f_{\theta}(x_{\tau},\tau,C_{t})-x_{0}\right\|_{2}^{2}\right],
$$

where the expectation is taken over expert trajectories $\zeta\sim\mathcal{D}_{E}$, trajectory timesteps $t\sim\mathcal{U}(0,T-H_{a})$, diffusion steps $\tau\sim\mathcal{U}(1,L)$, and Gaussian noise $\epsilon\sim\mathcal{N}(0,\mathbf{I})$. Accordingly, the overall objective is:

$$
\mathcal{L(\theta,\psi,\phi)}=\mathcal{L}_{\mathrm{diff}}(\theta,\psi)+\lambda\mathcal{L}_{\mathrm{dyn}}(\psi,\phi)
$$

where $\lambda$ balances action denoising and context representation learning.

![Refer to caption](https://arxiv.org/html/2605.14598v2/x3.png)

Figure 3: Overview of Experimental Environments. The figure summarizes representative environments from both simulation and real-world experiments. In each row, the left three panels visualize the RoboTwin tasks, the middle three columns present representative MetaWorld tasks, the next column shows Adroit tasks, and the rightmost column shows our real-world tasks.

### 4.3 Theoretical Analysis

Here we provide a theoretical analysis of the benefits of history conditioning for imitation learning in partially observable environments. Our goal is to characterize how the integration of temporal context mitigates the information loss inherent in POMDPs. To analyze the impact of information sets on performance, we denote $\mathcal{L}_{\mathrm{diff}}(\pi;X_{t})$ as the expected diffusion loss for a policy $\pi$ conditioned on variable $X_{t}$. With the formalization of POMDP and observation aliasing in section˜3, and the imitation objective defined in eq.˜9, we present two core propositions.

###### Proposition 4.1.

The minimum achievable diffusion-based imitation loss for a history-conditioned policy is always less than or equal to that of a reactive policy. That is,

$$
\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};h_{t})\leq\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};o_{t}).
$$

###### Proposition 4.2.

In the presence of observation aliasing, history conditioning strictly reduces the minimum achievable imitation loss compared to a reactive policy:

$$
\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};h_{t})<\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};o_{t}),\quad\text{whenever}\quad I_{E}(a_{t};h_{t}\mid o_{t})>0.
$$

This condition implies that history resolves state ambiguity by capturing mutual information that is inaccessible to a reactive policy.

Proposition˜4.1 establishes that history conditioning never degrades the theoretical performance limit of the policy, while proposition˜4.2 proves that history conditioning strictly improves performance in the presence of observation aliasing. These results demonstrate that history acts as a sufficient statistic for the belief state, allowing the policy to disambiguate latent environmental configurations through the capture of action-relevant mutual information. We summarize the high-level logic here and defer the complete mathematical derivations to the Appendix E.

## 5 Experiments

### 5.1 Experimental Setup

Table 1: Main Results on RoboTwin 2.0. Tasks are grouped into Short, Mid, and Long horizons. “Obs.” denotes the observation modality, including RGB and point-cloud observations. DSSP achieves the best overall performance with the smallest model size.

<table><tbody><tr><th rowspan="2">Method</th><th rowspan="2">Obs.</th><td colspan="4">Success Rate by Horizon <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td rowspan="2">Params <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Short (18)</td><td>Mid (15)</td><td>Long (17)</td><td>Average</td></tr><tr><th>ACT (RSS’23)</th><th>RGB</th><td>29.94</td><td>26.40</td><td>32.47</td><td>29.74</td><td>80.0M</td></tr><tr><th>DP (RSS’23)</th><th>RGB</th><td>31.28</td><td>26.00</td><td>26.41</td><td>28.04</td><td>96.8M</td></tr><tr><th><math><semantics><msub><mi>π</mi> <mn>0</mn></msub> <annotation>\pi_{0}</annotation></semantics></math> (arXiv’24)</th><th>RGB</th><td>44.56</td><td>44.07</td><td>50.47</td><td>46.42</td><td>3.3B</td></tr><tr><th>RDT (ICLR’25)</th><th>RGB</th><td>36.72</td><td>32.47</td><td>33.94</td><td>34.50</td><td>1.2B</td></tr><tr><th>SeedPolicy (arXiv’26)</th><th>RGB</th><td>43.39</td><td>36.53</td><td>47.59</td><td>42.76</td><td>147.3M</td></tr><tr><th>DP3 (RSS’24)</th><th>P.C.</th><td>59.83</td><td>52.53</td><td>52.76</td><td>55.24</td><td>264.4M</td></tr><tr><th>FlowPolicy (AAAI’25)</th><th>P.C.</th><td>54.89</td><td>37.40</td><td>29.59</td><td>41.04</td><td>264.4M</td></tr><tr><th>DSSP (Ours)</th><th>P.C.</th><td><math><semantics><mn>64.78</mn> <annotation>\mathbf{64.78}</annotation></semantics></math></td><td><math><semantics><mn>57.33</mn> <annotation>\mathbf{57.33}</annotation></semantics></math></td><td><math><semantics><mn>64.06</mn> <annotation>\mathbf{64.06}</annotation></semantics></math></td><td><math><semantics><mn>62.30</mn> <annotation>\mathbf{62.30}</annotation></semantics></math></td><td><math><semantics><mrow><mn>44.3</mn> <mo></mo><mi>𝐌</mi></mrow> <annotation>\mathbf{44.3M}</annotation></semantics></math></td></tr></tbody></table>

Datasets. We evaluate DSSP on diverse simulation benchmarks and real-world memory-dependent manipulation tasks, as shown in Figure˜3. For simulation, we use 87 tasks across three benchmarks: 50 bimanual tasks from RoboTwin 2.0 [^6], 34 single-arm tabletop tasks from MetaWorld [^41], and 3 dexterous in-hand tasks from Adroit [^32]. To analyze performance across task horizons, we group the 50 RoboTwin tasks into short-, mid-, and long-horizon categories based on average episode length. Detailed grouping criteria, task lists, and subset selection are provided in Appendix C.3.

For real-world evaluation, we use an AgileX robotic platform equipped with a fixed Intel RealSense L515 camera for point-cloud observations. We design three tabletop tasks that require long-horizon execution or progress tracking: (i) Put Bottles, where the robot sequentially places two bottles into a basket; (ii) Object Swap, where the robot swaps two objects through an intermediate buffer slot; and (iii) Morse Tapping, where the robot taps a target three times before returning to the initial position.

Baselines. We compare DSSP with representative imitation-learning baselines on each benchmark. DP3 [^42] is the most direct baseline, as it uses the same 3D point-cloud observation and action interfaces as DSSP, while DP [^7] serves as a standard diffusion-policy baseline. On RoboTwin, we include ACT [^43], RDT \[liu\_rdt-1b\_2025\], and $\pi_{0}$ [^1] from the official benchmark evaluation, together with recent methods SeedPolicy \[gui\_seedpolicy\_2026\], and FlowPolicy \[zhang2025flowpolicy\]. On MetaWorld and Adroit, we further compare with recent policy-learning methods, including AdaFlow \[hu\_adaflow\_2024\], CP \[prasad\_consistency\_2024\], and MP1 [^33].

Evaluation Metric and Protocol. We use task success rate as the primary metric. For RoboTwin, we evaluate 100 test episodes under the in-distribution demo clean setting, defining success as completion within the maximum horizon. For MetaWorld and Adroit, we evaluate 20 episodes every 200 training epochs and report the average of the top five success rates. For real-world experiments, each method undergoes 20 trials per task with randomized initial configurations based on task-specific criteria.

Implementation Details. All models are trained with AdamW on NVIDIA RTX 4090 GPUs. We use 50 demonstrations per task for RoboTwin and real-world experiments, and 10 demonstrations per task for MetaWorld and Adroit. DSSP employs trajectory-wise batching for causal history encoding. Training schedules, architectural details, and hyperparameters are specified in Appendix C.

### 5.2 Simulation Experiments

Main Results. We first evaluate DSSP on tasks from RoboTwin and report the success rate on categorized tasks in Table 1. On RoboTwin, DSSP achieves a 12.8% relative improvement compared to DP3 on average, with the largest improvement on long-horizon tasks (21.4%), indicating the benefit of long-horizon historical context. Beyond RoboTwin, we further evaluate DSSP on the shorter-horizon Adroit and MetaWorld benchmarks, with task horizons of 100 and 200 steps, respectively. As shown in Table˜2, DSSP achieves the best overall average among all compared methods.

Table 2: Results on Adroit and MetaWorld. A comprehensive comparison across 37 tasks using 3 random seeds. Numbers in parentheses indicate the number of tasks in each benchmark.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Adroit (3)</th><th colspan="4">MetaWorld</th><th rowspan="2">Average</th></tr><tr><th>Easy (21)</th><th>Medium (4)</th><th>Hard (4)</th><th>Very Hard (5)</th></tr></thead><tbody><tr><th>DP (RSS’23)</th><td><math><semantics><mrow><mn>21.0</mn> <mo>±</mo> <mn>7.7</mn></mrow> <annotation>21.0{\pm}7.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>50.7</mn> <mo>±</mo> <mn>6.1</mn></mrow> <annotation>50.7{\pm}6.1</annotation></semantics></math></td><td><math><semantics><mrow><mn>11.0</mn> <mo>±</mo> <mn>2.5</mn></mrow> <annotation>11.0{\pm}2.5</annotation></semantics></math></td><td><math><semantics><mrow><mn>5.25</mn> <mo>±</mo> <mn>2.5</mn></mrow> <annotation>5.25{\pm}2.5</annotation></semantics></math></td><td><math><semantics><mrow><mn>22.0</mn> <mo>±</mo> <mn>5.0</mn></mrow> <annotation>22.0{\pm}5.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>35.2</mn> <mo>±</mo> <mn>5.3</mn></mrow> <annotation>35.2{\pm}5.3</annotation></semantics></math></td></tr><tr><th>Adaflow (NeurIPS’24)</th><td><math><semantics><mrow><mn>30.0</mn> <mo>±</mo> <mn>7.7</mn></mrow> <annotation>30.0{\pm}7.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>49.4</mn> <mo>±</mo> <mn>6.8</mn></mrow> <annotation>49.4{\pm}6.8</annotation></semantics></math></td><td><math><semantics><mrow><mn>12.0</mn> <mo>±</mo> <mn>5.0</mn></mrow> <annotation>12.0{\pm}5.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>5.75</mn> <mo>±</mo> <mn>4.0</mn></mrow> <annotation>5.75{\pm}4.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>24.0</mn> <mo>±</mo> <mn>4.8</mn></mrow> <annotation>24.0{\pm}4.8</annotation></semantics></math></td><td><math><semantics><mrow><mn>35.6</mn> <mo>±</mo> <mn>6.1</mn></mrow> <annotation>35.6{\pm}6.1</annotation></semantics></math></td></tr><tr><th>CP (arXiv’24)</th><td><math><semantics><mrow><mn>29.7</mn> <mo>±</mo> <mn>6.7</mn></mrow> <annotation>29.7{\pm}6.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>69.3</mn> <mo>±</mo> <mn>4.2</mn></mrow> <annotation>69.3{\pm}4.2</annotation></semantics></math></td><td><math><semantics><mrow><mn>21.2</mn> <mo>±</mo> <mn>6.0</mn></mrow> <annotation>21.2{\pm}6.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>17.5</mn> <mo>±</mo> <mn>3.9</mn></mrow> <annotation>17.5{\pm}3.9</annotation></semantics></math></td><td><math><semantics><mrow><mn>30.0</mn> <mo>±</mo> <mn>4.9</mn></mrow> <annotation>30.0{\pm}4.9</annotation></semantics></math></td><td><math><semantics><mrow><mn>50.1</mn> <mo>±</mo> <mn>4.7</mn></mrow> <annotation>50.1{\pm}4.7</annotation></semantics></math></td></tr><tr><th>DP3 (RSS’24)</th><td><math><semantics><mrow><mn>67.3</mn> <mo>±</mo> <mn>5.0</mn></mrow> <annotation>67.3{\pm}5.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>87.3</mn> <mo>±</mo> <mn>2.2</mn></mrow> <annotation>87.3{\pm}2.2</annotation></semantics></math></td><td><math><semantics><mrow><mn>44.5</mn> <mo>±</mo> <mn>8.7</mn></mrow> <annotation>44.5{\pm}8.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>32.7</mn> <mo>±</mo> <mn>7.7</mn></mrow> <annotation>32.7{\pm}7.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>39.4</mn> <mo>±</mo> <mn>9.0</mn></mrow> <annotation>39.4{\pm}9.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>68.7</mn> <mo>±</mo> <mn>4.7</mn></mrow> <annotation>68.7{\pm}4.7</annotation></semantics></math></td></tr><tr><th>FlowPolicy (AAAI’25)</th><td><math><semantics><mrow><mn>71.0</mn> <mo>±</mo> <mn>2.3</mn></mrow> <annotation>71.0{\pm}2.3</annotation></semantics></math></td><td><math><semantics><mrow><mn>84.8</mn> <mo>±</mo> <mn>2.2</mn></mrow> <annotation>84.8{\pm}2.2</annotation></semantics></math></td><td><math><semantics><mrow><mn>58.2</mn> <mo>±</mo> <mn>7.9</mn></mrow> <annotation>58.2{\pm}7.9</annotation></semantics></math></td><td><math><semantics><mrow><mn>40.2</mn> <mo>±</mo> <mn>4.5</mn></mrow> <annotation>40.2{\pm}4.5</annotation></semantics></math></td><td><math><semantics><mrow><mn>52.2</mn> <mo>±</mo> <mn>5.0</mn></mrow> <annotation>52.2{\pm}5.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>71.6</mn> <mo>±</mo> <mn>3.5</mn></mrow> <annotation>71.6{\pm}3.5</annotation></semantics></math></td></tr><tr><th>MP1 (AAAI’26)</th><td><math><semantics><mrow><mn>75.7</mn> <mo>±</mo> <mn>2.3</mn></mrow> <annotation>\mathbf{75.7{\pm}2.3}</annotation></semantics></math></td><td><math><semantics><mrow><mn>88.2</mn> <mo>±</mo> <mn>1.1</mn></mrow> <annotation>88.2{\pm}1.1</annotation></semantics></math></td><td><math><semantics><mrow><mn>68.0</mn> <mo>±</mo> <mn>3.1</mn></mrow> <annotation>\mathbf{68.0{\pm}3.1}</annotation></semantics></math></td><td><math><semantics><mrow><mn>58.1</mn> <mo>±</mo> <mn>5.0</mn></mrow> <annotation>\mathbf{58.1{\pm}5.0}</annotation></semantics></math></td><td><math><semantics><mrow><mn>67.2</mn> <mo>±</mo> <mn>2.7</mn></mrow> <annotation>67.2{\pm}2.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>78.9</mn> <mo>±</mo> <mn>2.1</mn></mrow> <annotation>78.9{\pm}2.1</annotation></semantics></math></td></tr><tr><th>DSSP (ours)</th><td><math><semantics><mrow><mn>73.0</mn> <mo>±</mo> <mn>2.9</mn></mrow> <annotation>73.0{\pm}2.9</annotation></semantics></math></td><td><math><semantics><mrow><mn>90.5</mn> <mo>±</mo> <mn>2.1</mn></mrow> <annotation>\mathbf{90.5{\pm}2.1}</annotation></semantics></math></td><td><math><semantics><mrow><mn>67.4</mn> <mo>±</mo> <mn>4.1</mn></mrow> <annotation>67.4{\pm}4.1</annotation></semantics></math></td><td><math><semantics><mrow><mn>54.6</mn> <mo>±</mo> <mn>2.4</mn></mrow> <annotation>54.6{\pm}2.4</annotation></semantics></math></td><td><math><semantics><mrow><mn>71.3</mn> <mo>±</mo> <mn>3.8</mn></mrow> <annotation>\mathbf{71.3{\pm}3.8}</annotation></semantics></math></td><td><math><semantics><mrow><mn>80.1</mn> <mo>±</mo> <mn>2.6</mn></mrow> <annotation>\mathbf{80.1{\pm}2.6}</annotation></semantics></math></td></tr></tbody></table>

Analysis of History Encoding. To better understand how DSSP uses historical information, we conduct controlled studies on the six-task long-horizon subset defined in Appendix C.3. We analyze three aspects: (1) how temporal backbone and history length affect performance, (2) how efficiently the history encoder scales to full-history conditioning, and (3) whether the policy truly relies on context representation when recent observations are corrupted. Together, these studies show that the gains of DSSP come from effective long-history utilization, while maintaining practical efficiency.

Table 3: Comparison of temporal backbones across history lengths. We report average success over six long-horizon tasks and history-prefix encoding cost in p95 latency and peak GPU memory.

<table><thead><tr><th>Encoder Backbone</th><th>History Length (<math><semantics><msub><mi>T</mi> <mi>h</mi></msub> <annotation>T_{h}</annotation></semantics></math>)</th><th>Success Rate (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Encoding Cost <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><th rowspan="3">Transformer</th><th>10</th><td>68.00</td><td>1.43 ms / 176.5 MB</td></tr><tr><th>20</th><td>61.83</td><td>1.43 ms / 176.7 MB</td></tr><tr><th>Full history</th><td>66.00</td><td>3.61 ms / 586.2 MB</td></tr><tr><th rowspan="3">Mamba</th><th>10</th><td>53.00</td><td>1.87 ms / 181.1 MB</td></tr><tr><th>20</th><td>60.00</td><td>1.90 ms / 181.5 MB</td></tr><tr><th>Full history (Ours)</th><td>71.33</td><td>1.97 ms / 238.5 MB</td></tr></tbody></table>

We first compare the impact of history length on different encoder backbones (Table 3). While the Transformer encoder performs best with a short window ($T_{h}=10$) and stagnates with longer histories, the state-space encoder scales effectively with increasing context. Mamba achieves its peak success rate of 71.33% using full history, which is an 8.1% relative improvement over the full-history Transformer. These results confirm that the recurrent formulation of Mamba is better suited for aggregating varying length observations into a compact context representation.

Beyond improving success rates, DSSP also scales more efficiently to long histories. Full-history Transformer encoding costs 3.61 ms and a larger peak GPU memory (586.2 MB), whereas our full-history Mamba encoder requires only 1.97 ms and 238.5 MB, reducing latency by 45.4% and peak memory by 59.3%. This advantage stems from the linear-time state-space formulation, which aggregates history through recurrent state updates instead of pairwise attention over all historical tokens. Meanwhile, during streaming inference, DSSP only maintains a compact hidden-state cache, making the per-step encoding overhead nearly independent of accumulated history length.

Table 4: Robustness comparison under Gaussian perturbations to recent state representations. We report the six tasks’ average success rate (%) under different perturbation scales ($\sigma$) during inference.

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Noise Scale (<math><semantics><mi>σ</mi> <annotation>\sigma</annotation></semantics></math>)</th></tr><tr><th>0.00 (Clean)</th><th>0.05</th><th>0.10</th><th>0.15</th></tr></thead><tbody><tr><th>DP3 (Baseline)</th><td>58.67</td><td>43.00</td><td>15.83</td><td>3.17</td></tr><tr><th>DSSP (w/o full history)</th><td>60.00</td><td>41.67</td><td>23.33</td><td>11.33</td></tr><tr><th>DSSP (Full history)</th><td>71.33</td><td>52.33</td><td>37.00</td><td>20.83</td></tr></tbody></table>

We further test whether the policy uses historical context when recent observations are unreliable by perturbing the most recent three state tokens during inference: $z_{i}^{\mathrm{pert}}=z_{i}+\sigma\epsilon_{i}$, where $\epsilon_{i}\sim\mathcal{N}(0,\mathbf{I})$. As shown in Table 4, DP3 and our short-window ($T_{h}=10$) variant degrade sharply as the perturbation scale increases, while DSSP with full history maintains substantially higher success rates. These results indicate that the learned context token incorporates earlier information, preventing the policy from degenerating into merely relying on recent observations for action generation.

Table 5: Ablation on six history-sensitive RoboTwin tasks under the demo clean setting. Hist., TD, Recent, Dyn., and Trans. denote the history encoder, timestep-decoupled denoising, recent-state conditioning, dynamics-aware loss, and Transformer denoising backbone, respectively.

| Metric | DP3 | w/o Hist. | w/o TD | w/o Recent | w/o Dyn. | w/ Trans. | Full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Success Rate (%) | 58.67 | 64.33 | 66.67 | 68.17 | 69.50 | 68.50 | 71.33 |
| Relative Improvement (%) | – | +9.65 | +13.64 | +16.21 | +18.46 | +16.75 | +21.56 |

Ablation Study. We ablate the key components of DSSP on the six-task history-sensitive subset defined in Section 5.1. As shown in Table 5, the full model achieves the best success rate of 71.33%, a 21.56% relative improvement over DP3. Removing the history encoder leads to the largest drop, highlighting the importance of long-horizon context. Replacing the Mamba action denoising backbone with a Transformer-based one remains stronger than DP3 but underperforms the full model, showing that Mamba further improves temporal action generation. Without timestep-decoupled conditioning, performance drops to 66.67%, validating the effectiveness of our decoupled conditioning design. Finally, removing recent-state conditioning or the dynamics-aware loss also degrades performance, confirming the importance of local grounding and predictive context learning.

### 5.3 Real-World Experiments

We evaluate DSSP on three real-world manipulation tasks that require long-horizon memory or progress tracking, as shown in Table 6. DSSP substantially outperforms DP3 across all tasks, increasing the average success rate from 30% to 70% (a 133.3% relative improvement). The gains are particularly significant on Morse Tapping, where DSSP improves the success rate from 15% to 85%. These results demonstrate the effectiveness of full-history context for real-world memory-dependent manipulation, where the decision making often depends on previous interactions rather than the current observation alone. To better understand where the gains come from, we provide a failure-mode analysis for each real-world task in Appendix D.2 and the limitations are discussed in Appendix˜F.

Table 6: Real-world long-horizon task (avg. steps) evaluation.

| Method | Put Bottles (666) | Object Swap (713) | Morse Tapping (366) |
| --- | --- | --- | --- |
| FlowPolicy (AAAI’25) | 40% | 30% | 10% |
| MP1 (AAAI’26) | 45% | 30% | 25% |
| DP3 (RSS’24) | 40% | 35% | 15% |
| DSSP (ours) | 60% | 65% | 85% |

## 6 Conclusion

In this paper, we introduce DSSP, an efficient full-history conditioned diffusion state-space policy for long-horizon robot manipulation. Our results show that compactly encoding the full observation history improves temporal disambiguation and task-progress tracking in history-sensitive tasks. The proposed state-space history encoder, dynamics-aware objective, and hierarchical timestep-decoupled conditioning jointly integrate long-horizon context with recent observations for action generation.

## References

## Appendix A Preliminaries

Diffusion Policy. Diffusion Policy [^7] adapts Denoising Diffusion Probabilistic Models (DDPMs) [^16] to action generation. The policy treats a future action sequence $a_{t:t+H_{a}-1}$ of horizon $H_{a}$ as the clean sample $x_{0}$. We parameterize the model to predict the clean action sequence directly, conditioning the denoising process on the history representation $h_{t}$. During training, we optimize the reconstruction objective over $L$ diffusion steps:

$$
\mathcal{L}_{x_{0}}=\mathbb{E}_{x_{0},\tau,\epsilon}\left[\left\|f_{\theta}(x_{\tau},\tau,h_{t})-x_{0}\right\|_{2}^{2}\right],
$$

where $\tau\in\{1,\dots,L\}$ is the uniform diffusion step and $x_{\tau}$ is the noise-corrupted action sequence.

State Space Models (SSMs). We employ Mamba as the sequence backbone for both history encoding and action diffusion. A standard discrete SSM updates a hidden state $s_{t}\in\mathbb{R}^{N}$ and outputs $y_{t}$ via time-invariant parameters:

$$
s_{t}=\bar{A}s_{t-1}+\bar{B}x_{t},\qquad y_{t}=Cs_{t}.
$$

Mamba introduces a selective mechanism where parameters become input-dependent: $(\Delta_{t},B_{t},C_{t})=f_{\theta}(x_{t})$, with $\Delta_{t}$ serving as the discretization step size. The discretized parameters are:

$$
\bar{A}_{t}=\exp(\Delta_{t}A),\qquad\bar{B}_{t}=(\Delta_{t}A)^{-1}\big(\exp(\Delta_{t}A)-I\big)\Delta_{t}B_{t}.
$$

The recurrent update becomes $s_{t}=\bar{A}_{t}s_{t-1}+\bar{B}_{t}x_{t}$ and $y_{t}=C_{t}s_{t}$. This input-dependent selection enables the model to efficiently compress long observation histories and generate temporally coherent actions.

## Appendix B Additional Related Works

### B.1 Vision-Language-Action Models

Vision-language-action (VLA) models have recently become a central direction for scalable robot learning. RT-2 [^2] transfers vision-language knowledge to robotic control by representing actions as language-like tokens. Octo [^35] and OpenVLA [^23] further introduce open-source generalist policies trained on large cross-embodiment robot datasets. Recent generative VLA models move beyond discrete action tokenization toward continuous action generation, including flow-matching-based policies such as $\pi_{0}$ [^1] and large-scale diffusion policies such as RDT-1B \[liu\_rdt-1b\_2025\].

Recent work has further improved VLA policies along several directions. OpenVLA-OFT [^22] studies effective fine-tuning recipes with continuous actions and action chunking, while VITA-VLA [^12] equips pretrained vision-language models with action-generation capability through action expert distillation. Other works improve inference efficiency and temporal consistency through asynchronous generation, coarse-to-fine action generation, or action coherence guidance [^20] [^34]. In parallel, SpatialVLA [^31] and GraphCoT-VLA [^17] enhance spatial reasoning and embodiment-aware manipulation.

### B.2 World Models for Robotics

World models provide a complementary direction for robot learning by predicting future environment states for simulation, planning, or policy improvement. In robotic manipulation, action-conditioned video generation has been widely used to model robot-object dynamics. IRASim [^45] learns an interactive real-robot action simulator conditioned on observations and robot actions. RoboMaster [^11] improves trajectory-controlled robotic video generation by modeling robot-object interactions, while Ctrl-World [^15] studies controllable multi-view world modeling for evaluating and improving generalist robot policies.

Recent works further introduce structured spatial representations for world modeling. FlowDreamer [^14] uses RGB-D observations and 3D scene flow for action-conditioned future prediction. GAF [^4] and GWM [^26] represent dynamic manipulation scenes with Gaussian-based formulations, enabling future scene prediction and action refinement. Dream2Flow [^9] connects video generation with robot control by extracting 3D object flow from generated videos. World4RL [^21] and PlayWorld [^39] further use learned world models for policy evaluation, reinforcement learning, or policy refinement.

## Appendix C Implementation Details

### C.1 Trajectory-Wise Training

Since DSSP conditions action generation on causal history context, we use trajectory-wise batch construction during training. At each optimization step, we randomly load one complete demonstration trajectory and compute its state representations and causal context representations following Equations˜2 and 3. We then sample $B$ valid action-window start indices from this trajectory, where $B$ is the training batch size.

For each sampled start time $t_{i}$, we construct the hierarchical condition using Equation˜7, with the context representations and recent state representations available up to $t_{i}$. The corresponding diffusion target is the future action chunk starting from $t_{i}$. This ensures that every training sample is conditioned only on observations before its prediction time, without accessing future observations.

For fair comparison with window-based baselines, we keep the same effective batch size $B$ and comparable optimization budget across methods. Thus, trajectory-wise training does not introduce additional demonstrations or a larger denoising batch; it only changes how causal history conditions are constructed for DSSP. The training objectives follow Equation˜10.

Table 7: Hyperparameters used for RoboTwin simulation experiments.

| Hyperparameter | Value |
| --- | --- |
| Recent Observation Horizon ($N$) | 3 |
| Horizon ($H$) | 8 |
| Action Execution Horizon ($H_{a}$) | 6 |
| Expert Demonstrations per Task | 50 |
| Optimizer | AdamW |
| Betas $(\beta_{1},\beta_{2})$ | $(0.95,0.999)$ |
| Learning Rate | $1.0\times 10^{-4}$ |
| Weight Decay | $1.0\times 10^{-6}$ |
| Diffusion Training Steps | 100 |
| Inference Steps | 10 |
| Learning Rate Scheduler | Cosine |
| Warmup Steps | 500 |
| Prediction Type | Sample prediction |
| Dynamics Loss Weight ($\lambda$) | 0.05 |
| Dynamics Loss Type | Cosine distance |
| Mamba Backbone Layers | 8 |
| Mamba History Encoder Layers | 2 |
| Hidden Dimension | 512 |
| SSM State Dimension | 64 |

Table 8: Hyperparameters used for Adroit simulation experiments.

| Hyperparameter | Value |
| --- | --- |
| Recent Observation Horizon ($N$) | 2 |
| Horizon ($H$) | 4 |
| Action Execution Horizon ($H_{a}$) | 3 |
| Expert Demonstrations per Task | 10 |
| Optimizer | AdamW |
| Betas $(\beta_{1},\beta_{2})$ | $(0.95,0.999)$ |
| Learning Rate | $1.0\times 10^{-4}$ |
| Weight Decay | $1.0\times 10^{-6}$ |
| Epoch | 3000 |
| Learning Rate Scheduler | Cosine |
| Warmup Steps | 500 |
| Diffusion Training Steps | 100 |
| Inference Steps | 10 |
| Prediction Type | Sample prediction |
| Dynamics Loss Weight ($\lambda$) | 0.05 |
| Dynamics Loss Type | Cosine distance |
| Mamba Backbone Layers | 8 |
| Mamba History Encoder Layers | 2 |
| Hidden Dimension | 512 |
| SSM State Dimension | 64 |
| Evaluation Episodes | 20 |
| Maximum Evaluation Steps | 300 |

Table 9: Hyperparameters used for MetaWorld simulation experiments.

| Hyperparameter | Value |
| --- | --- |
| Recent Observation Horizon ($N$) | 2 |
| Horizon ($H$) | 4 |
| Action Execution Horizon ($H_{a}$) | 3 |
| Expert Demonstrations per Task | 10 |
| Optimizer | AdamW |
| Betas $(\beta_{1},\beta_{2})$ | $(0.95,0.999)$ |
| Learning Rate | $1.0\times 10^{-4}$ |
| Weight Decay | $1.0\times 10^{-6}$ |
| Epoch | 1000 |
| Learning Rate Scheduler | Cosine |
| Warmup Steps | 500 |
| Diffusion Training Steps | 100 |
| Inference Steps | 10 |
| Prediction Type | Sample prediction |
| Dynamics Loss Weight ($\lambda$) | 0.05 |
| Dynamics Loss Type | Cosine distance |
| Mamba Backbone Layers | 8 |
| Mamba History Encoder Layers | 2 |
| Hidden Dimension | 512 |
| SSM State Dimension | 64 |
| Evaluation Episodes | 20 |
| Maximum Evaluation Steps | 1000 |

### C.2 Hyperparameters

To account for the varying difficulty levels and unique characteristics of different benchmarks, we tailor our hyperparameter configurations to each individual dataset. The final settings, which include additional configurations for the Mamba-based history encoder, Mamba diffusion backbone, and dynamics-aware auxiliary objective and are summarized in Tables 7,8,9, are informed by established practices in prior literature [^42] [^6].

Table 10: RoboTwin task grouping by average episode length. Short, mid, and long horizons correspond to average episode lengths of $<150$, $150$ – $250$, and $>250$ steps, respectively.

| Horizon Group | Tasks |
| --- | --- |
| Long (17) | Put Bottles Dustbin, Open Microwave, Stack Blocks Three, Stack Bowls Three, Blocks Ranking Rgb, Blocks Ranking Size, Hanging Mug, Stack Blocks Two, Stack Bowls Two, Place Cans Plasticbox, Handover Block, Shake Bottle Horizontally, Put Object Cabinet, Dump Bin Bigbin, Open Laptop, Place Can Basket, Place Object Basket |
| Mid (15) | Shake Bottle, Place Burger Fries, Place Bread Basket, Place Dual Shoes, Handover Mic, Place Shoe, Place Empty Cup, Scan Object, Place Bread Skillet, Place Container Plate, Place A2B Left, Rotate Qrcode, Move Stapler Pad, Stamp Seal, Move Can Pot |
| Short (18) | Place Mouse Pad, Place Fan, Adjust Bottle, Move Pillbottle Pad, Place Object Scale, Place A2B Right, Press Stapler, Place Object Stand, Place Phone Stand, Pick Dual Bottles, Pick Diverse Bottles, Move Playingcard Away, Beat Block Hammer, Lift Pot, Turn Switch, Grab Roller, Click Bell, Click Alarmclock |

### C.3 RoboTwin Horizon Grouping

For horizon-wise analysis, we partition the 50 RoboTwin tasks according to their average episode length. Tasks with average length below 150 steps are categorized as short-horizon tasks, tasks between 150 and 250 steps are categorized as mid-horizon tasks, and tasks above 250 steps are categorized as long-horizon tasks. The resulting groups are listed in Table 10.

For ablation and diagnostic analysis, we further define a six-task long-horizon analysis subset from the above grouping. This subset is selected from tasks with long execution horizons and temporally dependent manipulation behaviors. The purpose of this subset is not to replace the full benchmark evaluation, but to provide a controlled set of tasks for studying how different architectural components affect history utilization. Unless otherwise specified, all ablation studies and history-utilization analyses are conducted on this subset. The selected tasks and their task-level results are shown in Table 11.

Table 11: Six-task history-sensitive analysis subset used for ablation and diagnostic studies. The table reports task-level success rates under the in-distribution (demo clean) evaluation setting.

| Task | Avg. Steps | DP3 (%) | DSSP (%) |
| --- | --- | --- | --- |
| Put Bottles Dustbin | 637 | 60 | 83 |
| Stack Bowls Three | 476 | 57 | 80 |
| Put Object Cabinet | 274 | 72 | 70 |
| Place Dual Shoes | 228 | 13 | 19 |
| Stack Bowls Two | 313 | 83 | 93 |
| Place Can Basket | 255 | 67 | 83 |
| Average | 364 | 58.67 | 71.33 |

## Appendix D Additional Results and Failure-Mode Analysis

### D.1 Per-task RoboTwin results.

LABEL:tab:appendix\_robotwin\_dp3\_vs\_ours\_50task reports the per-task success rates of DP3 and DSSP on all 50 RoboTwin tasks under the clean setting. Overall, DSSP improves the average success rate from 55.24% to 62.30%, with particularly large gains on tasks requiring sequential progress tracking, such as Open Microwave, Place Cans Plasticbox, Put Bottles Dustbin, and Stack Bowls Three.

Table 12: Per-task success rate comparison between RoboTwin DP3 and DSSP on 50 RoboTwin tasks under the clean setting.

| Task | DP3 | DSSP |
| --- | --- | --- |
| Adjust Bottle | 99.00 | 96.00 |
| Beat Block Hammer | 72.00 | 79.00 |
| Blocks Ranking RGB | 3.00 | 6.00 |
| Blocks Ranking Size | 2.00 | 4.00 |
| Click Alarmclock | 77.00 | 99.00 |
| Click Bell | 90.00 | 100.00 |
| Dump Bin Bigbin | 85.00 | 84.00 |
| Grab Roller | 98.00 | 98.00 |
| Handover Block | 70.00 | 95.00 |
| Handover Mic | 100.00 | 93.00 |
| Hanging Mug | 17.00 | 24.00 |
| Lift Pot | 97.00 | 96.00 |
| Move Can Pot | 70.00 | 86.00 |
| Move Pillbottle Pad | 41.00 | 58.00 |
| Move Playingcard Away | 68.00 | 71.00 |
| Move Stapler Pad | 12.00 | 16.00 |
| Open Laptop | 82.00 | 88.00 |
| Open Microwave | 61.00 | 97.00 |
| Pick Diverse Bottles | 52.00 | 53.00 |
| Pick Dual Bottles | 60.00 | 66.00 |
| Place A2B Left | 46.00 | 40.00 |
| Place A2B Right | 49.00 | 52.00 |
| Place Bread Basket | 26.00 | 29.00 |
| Place Bread Skillet | 19.00 | 39.00 |
| Place Burger Fries | 72.00 | 81.00 |
| Place Can Basket | 67.00 | 83.00 |
| Place Cans Plasticbox | 48.00 | 88.00 |
| Place Container Plate | 86.00 | 95.00 |
| Place Dual Shoes | 13.00 | 19.00 |
| Place Empty Cup | 65.00 | 86.00 |
| Place Fan | 36.00 | 40.00 |
| Place Mouse Pad | 4.00 | 4.00 |
| Place Object Basket | 65.00 | 62.00 |
| Place Object Scale | 15.00 | 10.00 |
| Place Object Stand | 60.00 | 61.00 |
| Place Phone Stand | 44.00 | 60.00 |
| Place Shoe | 58.00 | 49.00 |
| Press Stapler | 69.00 | 66.00 |
| Put Bottles Dustbin | 60.00 | 83.00 |
| Put Object Cabinet | 72.00 | 70.00 |
| Rotate QRcode | 74.00 | 66.00 |
| Scan Object | 31.00 | 29.00 |
| Shake Bottle | 98.00 | 100.00 |
| Shake Bottle Horizontally | 100.00 | 100.00 |
| Stack Blocks Three | 1.00 | 2.00 |
| Stack Blocks Two | 24.00 | 30.00 |
| Stack Bowls Three | 57.00 | 80.00 |
| Stack Bowls Two | 83.00 | 93.00 |
| Stamp Seal | 18.00 | 32.00 |
| Turn Switch | 46.00 | 57.00 |
| Average | 55.24 | 62.30 |

### D.2 Failure-Mode Analysis on Real-World Tasks

We further analyze the failure modes of each real-world task to better understand where the gains of DSSP come from.

Morse Tapping. DSSP achieves the largest relative improvement on this task. The policy is required to tap the target three times before returning, which demands accurate tracking of the number of completed taps. However, the visual observation before each tap is nearly identical, making the task progress ambiguous when only short-term context is available. As a result, DP3 often stops at an incorrect stage or performs redundant taps. By maintaining a history context, DSSP better tracks the tapping progress and executes the correct number of taps. The remaining failures of DSSP mainly come from inaccurate target localization, which can cause missed or imprecise contacts.

Put Bottles. This long-horizon task requires the robot to sequentially place multiple bottles into the basket. A common failure mode of DP3 is premature termination after placing only one bottle. This happens because intermediate states, where one bottle has already been placed, can appear visually similar to the final completion state when the policy only observes a short recent context. With historical context, DSSP better infers the overall task progress and distinguishes intermediate subgoals from true task completion. Its failures are more often caused by local manipulation errors, such as inaccurate grasping or placement, rather than losing track of the task stage.

Object Swap. This task requires swapping two objects through an intermediate buffer slot, with demonstrations collected in both swap directions. When an object is located in the buffer, the current observation alone is insufficient to determine whether it should be moved to the left or to the right. Consequently, DP3 may move the object back to its original location, leading to a reversal of progress. In contrast, DSSP uses the historical context to infer the previous movement direction and resolve this ambiguity. This allows the policy to maintain consistent task progress across visually aliased intermediate states.

## Appendix E Theoretical Analysis

We provide the formal proofs for the propositions presented in the main text regarding the theoretical advantages of history conditioning in imitation learning. Our analysis is grounded in the framework of Partially Observable Markov Decision Processes (POMDPs), where we demonstrate that the interaction history $h_{t}$ serves as a sufficient statistic for the underlying belief state. We establish two primary results: first, a safety guarantee showing that conditioning on history never increases the theoretical minimum loss (proposition˜4.1); and second, a proof of performance gain showing that history conditioning strictly improves performance in environments subject to observation aliasing (proposition˜4.2). These derivations utilize the law of total variance and information-theoretic principles to quantify the performance gap between reactive and history-dependent policies.

### E.1 Adding history information does not hurt the performance.

###### Proposition E.1.

The minimum achievable diffusion-based imitation loss for a history-conditioned policy is always less than or equal to that of an reactive policy. That is,

$$
\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};h_{t})\leq\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};o_{t}).
$$

###### Proof.

For notational brevity, we denote the minimum achievable loss for a given conditioning variable $X$ as $\mathcal{L}^{*}(X)=\underset{\theta}{\min}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};X)$. Consider the diffusion-based imitation loss defined in Eq. 9. For a conditioning variable $X$, the minimum achievable Mean Squared Error (MSE) loss is the expected conditional variance of the expert action $a_{t}$ calculated over the expert dataset $\mathcal{D}_{E}$:

$$
\mathcal{L}^{*}(X)=\mathbb{E}_{(X,a_{t})\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid X)].
$$

Specifically, we denote the optimal losses for reactive and history-conditioned policies as:

$$
\mathcal{L}^{*}(o_{t})=\mathbb{E}_{o_{t}\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid o_{t})]\quad\text{and}\quad\mathcal{L}^{*}(h_{t})=\mathbb{E}_{h_{t}\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid h_{t})].
$$

By definition, the history $h_{t}$ contains the current observation $o_{t}$ as its final element ($o_{t}\subset h_{t}$). This inclusion allows us to apply the law of total variance to decompose the variance of the action $a_{t}$ given $o_{t}$ as:

$$
\text{Var}(a_{t}\mid o_{t})=\mathbb{E}_{h_{t}}[\text{Var}(a_{t}\mid h_{t})\mid o_{t}]+\text{Var}_{h_{t}}(\mathbb{E}[a_{t}\mid h_{t}]\mid o_{t}).
$$

Intuitively, the first term represents the inherent uncertainty that remains even if we know the full history, which is also the minimum possible error of a history-conditioned policy. The second term represents the aliasing penalty, which measures how much the average expert action changes depending on which history led to the current observation. Since the variance of the conditional expectation (the second term) is non-negative, it follows that $\text{Var}(a_{t}\mid o_{t})\geq\mathbb{E}_{h_{t}}[\text{Var}(a_{t}\mid h_{t})\mid o_{t}]$.

To find the total loss, we take the expectation over the entire expert distribution $\mathcal{D}_{E}$:

$$
\mathbb{E}_{o_{t}\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid o_{t})]\geq\mathbb{E}_{o_{t}\sim\mathcal{D}_{E}}[\mathbb{E}_{h_{t}\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid h_{t})\mid o_{t}]].
$$

By the law of iterated expectations, the right-hand side simplifies to the total average variance over the dataset:

$$
\mathbb{E}_{o_{t}\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid o_{t})]\geq\mathbb{E}_{h_{t}\sim\mathcal{D}_{E}}[\text{Var}(a_{t}\mid h_{t})],
$$

which is equivalent to $\mathcal{L}^{*}(h_{t})\leq\mathcal{L}^{*}(o_{t})$. This proves that the history-conditioned objective never exceeds the observation-only objective when averaged over the expert demonstrations. ∎

### E.2 Adding history information can improve the performance.

###### Proposition E.2.

In the presence of observation aliasing, history conditioning strictly reduces the minimum achievable imitation loss compared to an reactive policy:

$$
\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};h_{t})<\min_{\theta}\mathcal{L}_{\mathrm{diff}}(\pi_{\theta};o_{t}),\quad\text{whenever}\quad I_{E}(a_{t};h_{t}\mid o_{t})>0.
$$

This condition implies that history resolves state ambiguity by capturing mutual information that is inaccessible to a reactive policy.

###### Proof.

Recall the decomposition from the law of total variance:

$$
\text{Var}(a_{t}\mid o_{t})=\mathbb{E}[\text{Var}(a_{t}\mid h_{t})\mid o_{t}]+\text{Var}(\mathbb{E}[a_{t}\mid h_{t}]\mid o_{t}).
$$

The gap between the optimal observation-only loss and the history-conditioned loss is determined by the second term, $\text{Var}(\mathbb{E}[a_{t}\mid h_{t}]\mid o_{t})$, which represents the variance of the expert’s conditional mean across different histories that share the same current observation.

Under the condition of observation aliasing, there exist histories such that $\mathbb{E}[a_{t}\mid h_{t}^{1}]\neq\mathbb{E}[a_{t}\mid h_{t}^{2}]$ for the same observation $o_{t}=o$. Because the conditional expectation $\mathbb{E}[a_{t}\mid h_{t}]$ is not constant given $o_{t}$, its variance is strictly positive:

$$
\text{Var}(\mathbb{E}[a_{t}\mid h_{t}]\mid o_{t})>0.
$$

This variance reduction is fundamentally linked to the conditional mutual information $I_{E}(a_{t};h_{t}\mid o_{t})$. Formally, this is defined as the reduction in the expert’s action entropy when conditioned on history:

$$
I_{E}(a_{t};h_{t}\mid o_{t})=H_{E}(a_{t}\mid o_{t})-H_{E}(a_{t}\mid h_{t}),
$$

where $H_{E}(\cdot\mid\cdot)$ denotes the conditional entropy. This term quantifies the additional information about the expert action $a_{t}$ contained in the full history $h_{t}$ that is not present in the current observation $o_{t}$ alone.

When observation aliasing exists, the expert’s action depends on the past context; therefore, $a_{t}$ and $h_{t}$ are not conditionally independent given $o_{t}$, leading to $I_{E}(a_{t};h_{t}\mid o_{t})>0$. Consequently, an reactive policy $\pi_{\theta}(a_{t}\mid o_{t})$ must average over these conflicting expert behaviors, resulting in a strictly higher Bayes risk. In contrast, a history-conditioned policy $\pi_{\theta}(a_{t}\mid h_{t})$ utilizes the additional information to disambiguate the states, thereby achieving a strictly lower imitation loss. ∎

## Appendix F Limitations

DSSP is primarily designed to improve temporal disambiguation and task-progress tracking through full-history conditioning. It does not directly eliminate low-level perception or control failures, such as inaccurate target localization, grasping, or placement, which remain a source of real-world failure. Moreover, our real-world evaluation is limited to three tabletop tasks with a fixed-camera point-cloud setup. Extending DSSP to more diverse embodiments, viewpoints, deformable objects, and dynamic environments remains an important direction for future work.

[^1]: J. Bai, Y. Chao, Q. Chen, J. Gu, M. J. Kim, Z. Li, X. Li, T. Lin, M. Liu, N. Ma, K. Mo, D. Qu, S. Sun, H. Xia, F. Wei, and X. Zeng (2025-12) Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge. arXiv. Note: arXiv:2512.10071 \[cs\] External Links: [Link](http://arxiv.org/abs/2512.10071), [Document](https://dx.doi.org/10.48550/arXiv.2512.10071) Cited by: §B.1, §5.1.

[^2]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich (2023) RT-2: vision-language-action models transfer web knowledge to robotic control. External Links: 2307.15818, [Link](https://arxiv.org/abs/2307.15818) Cited by: §B.1.

[^3]: J. Cao, Q. Zhang, J. Sun, J. Wang, H. Cheng, Y. Li, J. Ma, K. Wu, Z. Xu, Y. Shao, W. Zhao, G. Han, Y. Guo, and R. Xu (2025-06) Mamba Policy: Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models. arXiv (en-US). Note: arXiv:2409.07163 \[cs\] External Links: [Link](http://arxiv.org/abs/2409.07163), [Document](https://dx.doi.org/10.48550/arXiv.2409.07163) Cited by: §1, §2.

[^4]: Y. Chai, L. Deng, R. Shao, J. Zhang, K. Lv, L. Xing, X. Li, H. Zhang, and Y. Liu (2025) GAF: gaussian action field as a 4d representation for dynamic world modeling in robotic manipulation. External Links: 2506.14135, [Link](https://arxiv.org/abs/2506.14135) Cited by: §B.2.

[^5]: J. Chen, H. Fang, C. Wang, S. Wang, and C. Lu (2026-03) History-Aware Visuomotor Policy Learning via Point Tracking. arXiv. Note: arXiv:2509.17141 \[cs\] External Links: [Link](http://arxiv.org/abs/2509.17141), [Document](https://dx.doi.org/10.48550/arXiv.2509.17141) Cited by: §1.

[^6]: T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, W. Deng, Y. Guo, T. Nian, X. Xie, Q. Chen, K. Su, T. Xu, G. Liu, M. Hu, H. Gao, K. Wang, Z. Liang, Y. Qin, X. Yang, P. Luo, and Y. Mu (2025-08) RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation. arXiv (en-US). Note: arXiv:2506.18088 \[cs\] External Links: [Link](http://arxiv.org/abs/2506.18088), [Document](https://dx.doi.org/10.48550/arXiv.2506.18088) Cited by: §C.2, §5.1.

[^7]: C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2024-03) Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. arXiv. Note: arXiv:2303.04137 \[cs\]Comment: An extended journal version of the original RSS2023 paper External Links: [Link](http://arxiv.org/abs/2303.04137), [Document](https://dx.doi.org/10.48550/arXiv.2303.04137) Cited by: Appendix A, §1, §2, §5.1.

[^8]: T. Dao and A. Gu (2024-05) Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. arXiv (en-US). Note: arXiv:2405.21060 \[cs\] External Links: [Link](http://arxiv.org/abs/2405.21060), [Document](https://dx.doi.org/10.48550/arXiv.2405.21060) Cited by: §2.

[^9]: K. Dharmarajan, W. Huang, J. Wu, L. Fei-Fei, and R. Zhang (2025) Dream2Flow: bridging video generation and open-world manipulation with 3d object flow. External Links: 2512.24766, [Link](https://arxiv.org/abs/2512.24766) Cited by: §B.2.

[^10]: H. Fang, Y. Huang, Y. Zhao, P. Weng, X. Li, and Y. Ban (2026) OMP: one-step meanflow policy with directional alignment. External Links: 2512.19347, [Link](https://arxiv.org/abs/2512.19347) Cited by: §2.

[^11]: X. Fu, X. Wang, X. Liu, J. Bai, R. Xu, P. Wan, D. Zhang, and D. Lin (2026) Learning video generation for robotic manipulation with collaborative trajectory control. External Links: 2506.01943, [Link](https://arxiv.org/abs/2506.01943) Cited by: §B.2.

[^12]: D. Gao, B. Zhao, A. Lee, I. Chuang, H. Zhou, H. Wang, Z. Zhao, J. Zhang, and I. Soltani (2026) VITA: vision-to-action flow matching policy. External Links: 2507.13231, [Link](https://arxiv.org/abs/2507.13231) Cited by: §B.1.

[^13]: A. Gu and T. Dao (2024-05) Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv (en-US). Note: arXiv:2312.00752 \[cs\] External Links: [Link](http://arxiv.org/abs/2312.00752), [Document](https://dx.doi.org/10.48550/arXiv.2312.00752) Cited by: §1, §2.

[^14]: J. Guo, X. Ma, Y. Wang, M. Yang, H. Liu, and Q. Li (2025) FlowDreamer: a rgb-d world model with flow-based motion representations for robot manipulation. External Links: 2505.10075, [Link](https://arxiv.org/abs/2505.10075) Cited by: §B.2.

[^15]: Y. Guo, L. X. Shi, J. Chen, and C. Finn (2026) Ctrl-world: a controllable generative world model for robot manipulation. External Links: 2510.10125, [Link](https://arxiv.org/abs/2510.10125) Cited by: §B.2.

[^16]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. External Links: 2006.11239, [Link](https://arxiv.org/abs/2006.11239) Cited by: Appendix A.

[^17]: H. Huang, M. Cen, K. Tan, X. Quan, G. Huang, and H. Zhang (2025) GraphCoT-vla: a 3d spatial-aware reasoning vision-language-action model for robotic manipulation with ambiguous instructions. External Links: 2508.07650, [Link](https://arxiv.org/abs/2508.07650) Cited by: §B.1.

[^18]: H. Jang, S. Yu, H. Kwon, H. Jeon, Y. Seo, and J. Shin (2025-10) ContextVLA: Vision-Language-Action Model with Amortized Multi-Frame Context. arXiv. Note: arXiv:2510.04246 \[cs\] External Links: [Link](http://arxiv.org/abs/2510.04246), [Document](https://dx.doi.org/10.48550/arXiv.2510.04246) Cited by: §1.

[^19]: X. Jia, Q. Wang, A. Donat, B. Xing, G. Li, H. Zhou, O. Celik, D. Blessing, R. Lioutikov, and G. Neumann (2024-11) MaIL: Improving Imitation Learning with Mamba. arXiv (en-US). Note: arXiv:2406.08234 \[cs\] External Links: [Link](http://arxiv.org/abs/2406.08234), [Document](https://dx.doi.org/10.48550/arXiv.2406.08234) Cited by: §1, §2.

[^20]: Y. Jiang, S. Cheng, Y. Ding, F. Gao, and B. Qi (2025) AsyncVLA: asynchronous flow matching for vision-language-action models. External Links: 2511.14148, [Link](https://arxiv.org/abs/2511.14148) Cited by: §B.1.

[^21]: Z. Jiang, K. Liu, Y. Qin, S. Tian, Y. Zheng, M. Zhou, C. Yu, H. Li, and D. Zhao (2026) World4RL: diffusion world models for policy refinement with reinforcement learning for robotic manipulation. External Links: 2509.19080, [Link](https://arxiv.org/abs/2509.19080) Cited by: §B.2.

[^22]: M. J. Kim, C. Finn, and P. Liang (2025) Fine-tuning vision-language-action models: optimizing speed and success. External Links: 2502.19645, [Link](https://arxiv.org/abs/2502.19645) Cited by: §B.1.

[^23]: M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn (2024-09) OpenVLA: An Open-Source Vision-Language-Action Model. arXiv. Note: arXiv:2406.09246 \[cs\] External Links: [Link](http://arxiv.org/abs/2406.09246), [Document](https://dx.doi.org/10.48550/arXiv.2406.09246) Cited by: §B.1.

[^24]: A. Lahoti, K. Y. Li, B. Chen, C. Wang, A. Bick, J. Z. Kolter, T. Dao, and A. Gu (2026-03) Mamba-3: Improved Sequence Modeling using State Space Principles. arXiv (en-US). Note: arXiv:2603.15569 \[cs\] External Links: [Link](http://arxiv.org/abs/2603.15569), [Document](https://dx.doi.org/10.48550/arXiv.2603.15569) Cited by: §2.

[^25]: H. Li, S. Yang, Y. Chen, X. Chen, X. Yang, Y. Tian, H. Wang, T. Wang, D. Lin, F. Zhao, and J. Pang (2025-10) CronusVLA: Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling. arXiv. Note: arXiv:2506.19816 \[cs\] External Links: [Link](http://arxiv.org/abs/2506.19816), [Document](https://dx.doi.org/10.48550/arXiv.2506.19816) Cited by: §1.

[^26]: G. Lu, B. Jia, P. Li, Y. Chen, Z. Wang, Y. Tang, and S. Huang (2025) GWM: towards scalable gaussian world models for robotic manipulation. External Links: 2508.17600, [Link](https://arxiv.org/abs/2508.17600) Cited by: §B.2.

[^27]: Y. Lu, Y. Tian, Z. Yuan, X. Wang, P. Hua, Z. Xue, and H. Xu (2025-05) H$^{\\mathbf{3}}$DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning. arXiv (en-US). Note: arXiv:2505.07819 \[cs\] version: 1 External Links: [Link](http://arxiv.org/abs/2505.07819), [Document](https://dx.doi.org/10.48550/arXiv.2505.07819) Cited by: §1, §2.

[^28]: J. Ma, Y. Qin, Y. Li, X. Liao, Y. Guo, and R. Zhang (2025-08) CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion. arXiv. Note: arXiv:2506.14769 \[cs\] External Links: [Link](http://arxiv.org/abs/2506.14769), [Document](https://dx.doi.org/10.48550/arXiv.2506.14769) Cited by: §2.

[^29]: N. Oh, J. Jang, M. Jung, and D. Park (2026) DiSPo: diffusion-ssm based policy learning for coarse-to-fine action discretization. External Links: 2409.14719, [Link](https://arxiv.org/abs/2409.14719) Cited by: §2.

[^30]: W. Peebles and S. Xie (2023-03) Scalable Diffusion Models with Transformers. arXiv (en-US). Note: arXiv:2212.09748 \[cs\] External Links: [Link](http://arxiv.org/abs/2212.09748), [Document](https://dx.doi.org/10.48550/arXiv.2212.09748) Cited by: §1.

[^31]: D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, and X. Li (2025) SpatialVLA: exploring spatial representations for visual-language-action model. External Links: 2501.15830, [Link](https://arxiv.org/abs/2501.15830) Cited by: §B.1.

[^32]: A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine (2018-06) Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. arXiv. Note: arXiv:1709.10087 \[cs\] External Links: [Link](http://arxiv.org/abs/1709.10087), [Document](https://dx.doi.org/10.48550/arXiv.1709.10087) Cited by: §5.1.

[^33]: J. Sheng, Z. Wang, P. Li, and M. Liu (2025-10) MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation. arXiv. Note: arXiv:2507.10543 \[cs\] External Links: [Link](http://arxiv.org/abs/2507.10543), [Document](https://dx.doi.org/10.48550/arXiv.2507.10543) Cited by: §1, §2, §5.1.

[^34]: J. Tang, Y. Sun, Y. Zhao, S. Yang, Y. Lin, Z. Zhang, J. Hou, Y. Lu, Z. Liu, and S. Han (2025) VLASH: real-time vlas via future-state-aware asynchronous inference. External Links: 2512.01031, [Link](https://arxiv.org/abs/2512.01031) Cited by: §B.1.

[^35]: O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine (2024) Octo: an open-source generalist robot policy. External Links: 2405.12213, [Link](https://arxiv.org/abs/2405.12213) Cited by: §B.1.

[^36]: Y. Wei, H. Liao, Y. Lin, P. Wang, Z. Liang, G. Liu, and W. Zheng (2025) CycleManip: enabling cyclic task manipulation via effective historical perception and understanding. arXiv preprint arXiv:2512.01022. Cited by: §1.

[^37]: J. Xie, X. Luo, H. Wu, J. Zhang, Y. Xing, L. Gao, and J. Song In-context adaptation for generalizable imitation learning. In CoRL 2025 Workshop RemembeRL, Cited by: §2.

[^38]: G. Yan, J. Zhu, Y. Deng, S. Yang, R. Qiu, X. Cheng, M. Memmel, R. Krishna, A. Goyal, X. Wang, and D. Fox (2025-09) ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training. arXiv (en-US). Note: arXiv:2509.01819 \[cs\] External Links: [Link](http://arxiv.org/abs/2509.01819), [Document](https://dx.doi.org/10.48550/arXiv.2509.01819) Cited by: §1, §2.

[^39]: T. Yin, Z. Mei, Z. Zheng, M. Yamane, D. Wang, J. Sceats, S. M. Bateman, L. Zha, A. Badithela, O. Shorinwa, and A. Majumdar (2026) PlayWorld: learning robot world models from autonomous play. External Links: 2603.09030, [Link](https://arxiv.org/abs/2603.09030) Cited by: §B.2.

[^40]: Y. Yoo, J. Hu, Y. Zhu, B. Liu, Q. Liu, R. Martín-Martín, and P. Stone (2025-09) RoboSSM: Scalable In-context Imitation Learning via State-Space Models. arXiv (en-US). Note: arXiv:2509.19658 \[cs\] External Links: [Link](http://arxiv.org/abs/2509.19658), [Document](https://dx.doi.org/10.48550/arXiv.2509.19658) Cited by: §1.

[^41]: T. Yu, D. Quillen, Z. He, R. Julian, A. Narayan, H. Shively, A. Bellathur, K. Hausman, C. Finn, and S. Levine (2021-06) Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning. arXiv. Note: arXiv:1910.10897 \[cs\] External Links: [Link](http://arxiv.org/abs/1910.10897), [Document](https://dx.doi.org/10.48550/arXiv.1910.10897) Cited by: §5.1.

[^42]: Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu (2024-09) 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations. arXiv (en-US). Note: arXiv:2403.03954 \[cs\]Comment: Published at Robotics: Science and Systems (RSS) 2024. Videos, code, and data: https://3d-diffusion-policy.github.io External Links: [Link](http://arxiv.org/abs/2403.03954), [Document](https://dx.doi.org/10.48550/arXiv.2403.03954) Cited by: §C.2, §1, §2, §5.1.

[^43]: T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023-04) Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. arXiv (en-US). Note: arXiv:2304.13705 \[cs\] External Links: [Link](http://arxiv.org/abs/2304.13705), [Document](https://dx.doi.org/10.48550/arXiv.2304.13705) Cited by: §1, §2, §5.1.

[^44]: R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daumé III, A. Kolobov, F. Huang, and J. Yang (2024) Tracevla: visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345. Cited by: §1.

[^45]: F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong (2025) IRASim: a fine-grained world model for robot manipulation. External Links: 2406.14540, [Link](https://arxiv.org/abs/2406.14540) Cited by: §B.2.