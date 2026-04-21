Yihong Liu <sup>1</sup> Dongyeop Kang <sup>2</sup> Sehoon Ha <sup>1</sup> <sup>1</sup> YL and SH are with Georgia Institute of Technology, Atlanta, GA, USA {yliu3518,sehoonha}@gatech.edu <sup>2</sup> DK is with Electronics and Telecommunications Research Institute, Daegu, Korea kang@etri.re.kr

###### Abstract

Autonomous robotic wiping is an important task in various industries, ranging from industrial manufacturing to sanitization in healthcare. Deep reinforcement learning (Deep RL) has emerged as a promising algorithm, however, it often suffers from a high demand for repetitive reward engineering. Instead of relying on manual tuning, we first analyze the convergence of quality-critical robotic wiping, which requires both high-quality wiping and fast task completion, to show the poor convergence of the problem and propose a new bounded reward formulation to make the problem feasible. Then, we further improve the learning process by proposing a novel visual-language model (VLM) based curriculum, which actively monitors the progress and suggests hyperparameter tuning. We demonstrate that the combined method can find a desirable wiping policy on surfaces with various curvatures, frictions, and waypoints, which cannot be learned with the baseline formulation. The demo of this project can be found at: [https://sites.google.com/view/highqualitywiping](https://sites.google.com/view/highqualitywiping)

## I INTRODUCTION

Robotic surface wiping is an important manipulation task with wide domains, such as automation and healthcare. Active research areas involve state detection, trajectory planning, and the low-level interaction skills with surfaces. Our work focuses on learning surface interaction skills with a blind policy. A blind wiping policy is often required and cost-effective for scenarios without obstacles, such as wiping tables or car surfaces and handling workpieces. This problem has been commonly approached by classical model-based approaches, which often leverage operational space control and impedance control [^1] [^2] [^3], particularly on a flat surface. However, it is not straightforward to design a model-based controller that works on a variety of surfaces with different curvatures and friction parameters [^4] [^5] [^6].

Our work investigates learning-based algorithms to take uncertainties into consideration. Unlike traditional approaches that rely on predefined models, learning-based algorithms often demonstrate robust performance in such uncertain environments by leveraging a massive amount of simulation samples. We utilize deep reinforcement learning (deep RL) to generate high-level policies through simulation without prior demonstrations, for dynamic adaptation to complex environmental variables. As a result, deep RL will allow us to obtain an autonomous hybrid pose/force controller for precise navigation and force control during wiping tasks on surfaces with varying curvatures and friction.

Our research addresses a critical challenge in applying RL to real-world robotic tasks: the inadequacy of off-the-shelf RL approaches for quality-critical tasks. During RL training, we observed that navigational wiping with quality control is essentially a “quality-critical” Markov Decision Process (MDP) problem, demanding a critical balance between fast task execution and high-quality wiping. This duality makes the task very sensitive to hyperparameters. The naive formulation of step-wise rewards for quality instruction and episodic sparse rewards for completion, can easily lead to either degrading work qualities or incentivizing the avoidance of task completion. In fact, the sensitive hyperparameter tuning would be a common issue for many real-world robotic tasks, which has been approached by extensive, labor-intensive manual tuning through repeated trial-and-error.

![Refer to caption](https://arxiv.org/html/2502.12599v1/extracted/6212486/figs/force-profile-0915.png)

Figure 1: The example trajectories of the learned wiping policy on surfaces with different curvatures and frictions.

To address this parameter-sensitive, multi-task learning in RL training, we first demonstrate the infeasibility of the naive formulation, and developed two techniques that we believe are generalizable to tasks facing similar challenges of balancing procedure qualities control and rapid task completion: (i) a bounded reward design with concentric circular checkpoints, which is theoretically grounded, proving that desired behaviors inherently lead to maximal rewards; and (ii) a novel visual-language model (VLM) based curriculum system that simulates human reward engineering, leveraging semanticmulti-modal understanding and proposing new reward weights. These methods combined, make the convergence supported by thorough analysis while reducing the laborious efforts of fine-tuning required from human researchers.

We show that our novel framework with two novel inventions, bounded reward and VLM-based curriculum, can practically improve the learning process by performing evaluations in a MuJoCo-based environment with variable curvatures, frictions, and waypoint positions. For a 2-points navigation task with a target force of 60N, following 800k training steps, our method yielded a 98% success rate (+69%) in navigation, and an average Integral Absolute Error (IAE) of 243 (-9%), over 25 (-34%) average completion steps.

To summarize, our main contributions are as follows:

1. We formally analyze the convergence of quality-critical robotic wiping and prove the infeasibility of the naive formulation.
2. We propose a new bounded reward function that makes the problem feasible.
3. We propose a novel VLM-based curriculum for automated and effective parameter tuning.
4. We demonstrate the effectiveness of the combined learning framework.

## II Related Work

### II-A Robotics Surface Wiping

Recent works leverage visual observations to generate synthesizing cleaning plans [^7] [^8], bounding box and litter classification [^9], dense waypoints [^10] [^11], or high-level waypoints with crumbs/spill dynamics modeling [^12].

The need for contact force control in robot manipulators, beyond simple position control, is detailed in a survey paper [^13] and its references. Several studies utilize dynamic models or sensor feedback for constant contact force and pose correction on unknown curved surfaces [^1] [^2] [^3] [^14]. Others use learning-based methods for better generalization to different tools and surfaces. Existing works include learning from demonstration (LfD) and applying motion to different flat, rectangular and horizontal surfaces [^15]; using reinforcement learning [^16] for tangential angle estimation and constant force tracking; using deep learning network to learn the surface material embedding [^17], image embedding of different 3D objects (e.g., cubes, rounds) [^18] [^19] and subsequent motion control.

### II-B Deep Reinforcement Learning for Robotics Manipulation

Deep Reinforcement Learning (DRL) has become pivotal for robotic tasks, complemented by Learning from Demonstration (LfD) which has shown promising outcomes (e.g., [^11] [^20]). Significant progress in robotic manipulation pre-training via demonstrations has been reported [^21]. Yet RL remains critical for autonomously enhancing simulated demonstrations and subsequent refinement for adaptations.

Our approach diverges in two key aspects from each. Firstly, unlike Zhang et al. [^16] ’s focus on tangential angle estimation and constant force tracking, our emphasis lies on integrating force control within navigational tasks. Secondly, unlike Lew et al. [^12] concentrates on crumb collection and spill cleaning on a fixed surface, we train wiping control policies across environments of varying curvatures and smoothness; in contrast to [^12] ’s use of admittance control with a pre-set normal force, which may falter or prove costly in dynamically changing environments, we gain force control through learning in varied training environments, adaptively determining control inputs.

### II-C Language to Reward

Recent efforts have integrated large language models (LLM) with robotics for plan generation, skill bootstrapping, state representation and language-conditioned manipulation. Our work on a visual-language model (VLM) curriculum contributes to the Language to RL Reward initiative, which focuses on converting language into actionable robotic rewards. Notably, EUREKA [^22] automates reward code evolution from environmental and task descriptions through evolutionary optimization based on RL feedback [^22]; TEXT2REWARD [^23] takes in similar inputs but incorporates human feedback after each RL cycle [^23]. Yu et al. [^24] uses heuristic templates to transform task descriptions into reward parameters for model predictive control (MPC) [^24].

Our VLM-based curriculum can be viewed as an extension of EUREKA [^22] adapted for our learning purpose: Eureka has a LLM agent update the whole reward function and retrains from scratch for each iteration; we start with a structured RL reward formula to avoid known undesired behaviors, and only update the reward weights during the training process to balance different learning goals. In addition, we add a separate vision-language model (VLM) agentVLM to get visual policy replay feedback without extensive logging, analogy to human experiences.

## III Robotic Wiping as Quality-critical MDP

We will first formalize the problem of robotic wiping as a common Markov Decision Processes (MDPs) with dense rewards provided per step and sparse reward per episode. Then, we will show the infeasibility of the given wiping task because it is a quality-critical task. Then, we propose a new bounded formulation that makes the problem feasible.

### III-A Initial Formulation of Markov Decision Process

We formulate robotic wiping as a Partially Observable Markov Decision Process (POMDP), which is a tuple of the state space $S$, the observation space $O$, the action space $A$, the reward function $r$, the initial state distribution $\rho_{0}$, the transition function $P(\mathbf{s}_{t+1}|\mathbf{s}_{t},\mathbf{a}_{t})$, and the discount factor $\gamma$. Our problem is partially observable because certain information, such as the tabletop’s curvature and smoothness, is inaccessible due to limited sensory feedback. Then our goal is to find the optimal policy $\pi:{O}\mapsto{A}$ that maximizes the expected episodic reward: $E_{\mathbf{s}_{0}\sim D}[\sum_{t=0}^{T}\gamma^{t}r(\mathbf{s}_{t},\mathbf{a}_{%
t})]$.

State: the state $\mathbf{s}\in S$ is defined as the internal state of the physics-based simulation.

Observation: A $46$ dimensional observation vector $\mathbf{o}\in O$ includes waypoint information, joint positions and velocities encoded as sine and cosine of their values, end-effector position and orientation, and force/torque sensor values.

Action: we use a six dimensional pose control to directly adjust the precise position and orientation of the end-effector, which also indirectly adjusts the forces.

Reward: The reward function $r$ is defined as a weighted sum of the five terms:

$$
r(\mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{t+1})=\begin{cases}r_{\text{col}}&%
\text{if collides},\\
r_{\text{con}}+r_{\text{force}}+r_{\text{way}}+r_{\text{ac}}&\text{otherwise},%
\end{cases}
$$

where we omit their arguments for brevity. We also encapsulate all the weights inside of the terms. If collision happens, agent will receive a negative scalar reward $r_{\text{col}}=-w_{col}$ to penalize collision with the episode terminates immediately. Otherwise, we consider four terms that are contact flag, contact force, waypoint, and acceleration penalization rewards. First, the contact flag reward is defined as $r_{\text{con}}=w_{\text{con}}\mathbf{I}_{\text{con}}$, where $\mathbf{I}_{\text{con}}$ is a zero or one binary flag whether the end effector makes any contact with the table. The second force term, $r_{\text{force}}$ encourages force control while moving towards the target, which is defined as:

$$
r_{\text{force}}=w_{\text{force}}\exp\left(-\frac{(f_{z}-\mu)^{2}}{2\sigma^{2}%
}\right)\mathbf{I}_{\text{align}},
$$

Where $w_{\text{force}}$ is the weight, $f_{z}$ is the upward/downward force applied at the force sensor at EE, and $e^{-\frac{(f_{z}-\mu)^{2}}{2\sigma^{2}}}$ is a Gaussian shape reward centering at target force $\mu$ (in our case, $\mu=60\text{N}$). $\mathbf{I}_{\text{align}}$ is a binary flag which checks the alignment between EE’s movement direction and the direction toward the next way point, which returns one when their cosine similarity is greater than 0.8.

The waypoint reward $r_{\text{way}}=w_{\text{way}}\mathbf{I}_{\text{way}}$ denotes the positive episodic reward agent receives for wiping each way point, as $\mathbf{I}_{\text{way}}$ indicates the completion of the waypoint. If the last waypoint is wiped, an extra sparse episodic reward will be provided, end the episode ends. Finally, the term $r_{ac}(\mathbf{a}_{t}):=w_{\textbf{ac}}(|a_{x}|+|a_{y}|+|a_{z}|)$ penalizes excessive actions, where $a_{x}$, $a_{y}$, $a_{z}$ are agent’s accelerations at $x$, $y$, $z$ axis respectively.

### III-B Convergence Analysis of Quality-critical MDP

The reward formulation in the previous section consists of common terms in robot learning: dense stepwise feedback to promote desired behaviors and substantial completion rewards to encourage the fast completion. In practice, many researchers typically tune the ratios with many rounds of trial and error to obtain the desirable behaviors. However, tuning hyperparameters for tasks requiring both in-process quality and rapid completion presents significant challenges.

Let us simplify two rewards: a continuous quality reward $W_{q}$ and an episodic terminal reward for wiping all waypoints, $W_{T}$. In our case, $W_{q}$ considers $r_{\text{con}}$, $r_{\text{force}}$, and $r_{\text{ac}}$ while $W_{T}$ corresponds to the waypoint reward $r_{way}$. We have $W_{q}^{\text{max}}>0$ for constant contact with target force and small accelerations, $W_{q}^{\text{poor}}<W_{q}^{\text{max}}$ for all other undesired qualities, and $W_{T}>0$ to encourage completion. Then, the agent can learn one of three possible strategies, and get respective accumulated rewards:

- optimal: takes the best quality wipe and terminates at minimum required time $\text{T}_{2}$ steps: $\sum_{t=0}^{T_{2}}\gamma^{t}W_{q}^{\text{max}}+\gamma^{T_{2}}W_{T}$.
- lazy: suboptimal, finishes episode as early as possible without maintaining wiping qualities (e.g., jumping between waypoints with high accelerations and no constant contacts): $\sum_{t=0}^{T_{1}}\gamma^{t}W_{q}^{\text{poor}}+\gamma^{T_{1}}W_{T}$.
- forever: suboptimal, keeps getting a quality reward without task completion: $\sum_{t=0}^{\infty}\gamma^{t}W_{q}^{\text{max}}=W_{q}^{\text{max}}/(1-\gamma)$.

For stable learning, it’s crucial to establish a feasible relationship between $W_{T}$ and $W_{q}^{\max}$ so that accumulated rewards meet the constraints for episodes of varying lengths $\text{T}_{1}<\text{T}_{2}$ and for any $W_{q}^{\text{poor}}<W_{q}^{\text{max}}$. From $\textbf{R}_{\textbf{optimal}}\gg\textbf{R}_{\textbf{lazy}}$, we get the relation below.

$$
W_{T}\ll(\sum_{t=0}^{T_{2}}\gamma^{t}W_{q}^{\text{max}}-\sum_{t=0}^{T_{1}}%
\gamma^{t}W_{q}^{\text{poor}})/(\gamma^{T_{1}}-\gamma^{T_{2}}),
$$

And from $\textbf{R}_{\textbf{optimal}}\gg\textbf{R}_{\textbf{forever}}$, we get the relation below.

$$
W_{T}\gg(\sum_{t=T_{2+1}}^{\infty}\gamma^{t}W_{q}^{\text{max}})/\gamma^{T_{2}},
$$

By combining Eqs. 3 and 4, we want to find a feasible range of $W_{T}$ regarding $W_{q}^{\text{max}}$:

$$
\text{L}(\text{W}_{q}^{\text{max}})\ll W_{T}\ll\text{U}(\text{W}_{q}^{\text{%
max}})
$$

Where $\text{U}(\text{W}_{q}^{\text{max}})=(\sum_{t=0}^{T_{2}}\gamma^{t}W_{q}^{\text{%
max}}-\sum_{t=0}^{T_{1}}\gamma^{t}W_{q}^{\text{poor}})/(\gamma^{T_{1}}-\gamma^%
{T_{2}})$ and $\text{L}(\text{W}_{q}^{\text{max}})=(\sum_{t=T_{2+1}}^{\infty}\gamma^{t}W_{q}^%
{\text{max}})/\gamma^{T_{2}}$. Finding the lower bound of $\text{U}(\text{W}_{q}^{\text{max}})$ is more straightforward, as $\text{T}_{2}/\text{T}_{1}$ predominantly influences the exponential terms, while $\text{W}_{q}^{\text{poor}}/\text{W}_{q}^{\text{max}}$ affects only the linear terms. We can approximate the lower bound of $\text{U}(\text{W}_{q}^{\text{max}})$ by setting $\text{T}_{1}=1$ and $\text{T}_{2}=\text{H}$, where H denotes the episode horizon (in our case, $\text{H}=200$). Then $\text{U}(\text{W}_{q}^{\text{max}})\in[99.02\text{W}_{q}^{\text{max}},101.30%
\text{W}_{q}^{\text{max}}]$ for $\text{W}_{q}^{\text{poor}}/\text{W}_{q}^{\text{max}}\in[0.01,0.99]$.

On the contrary, finding a feasible $\text{L}(\text{W}_{q}^{\text{max}})$ applicable for all $\text{T}_{2}$ is more challenging and prevents the the feasible range of current formulation, which motivates the next section.

### III-C Bounded Reward Design for Improved Feasibility

![Refer to caption](https://arxiv.org/html/2502.12599v1/extracted/6212486/figs/checkpoint_region.jpg)

Figure 2: Illustration of Checkpoint Regions.

To address $\text{L}(\text{W}_{q}^{\text{max}})$, we introduce concentric circular checkpoint regions between waypoints to promote navigation, inspired by the horizontal checkpoints in the Google research football environment [^25]. This setup introduces a bounded reward mechanism for target force control $r_{\text{force}}$ and constant contact $r_{\text{con}}$, as outlined in equation (1).

Fig. 2 illustrates these checkpoint regions around a waypoint, with the next waypoint marked by a green point at the center of equally distanced concentric circles. Rewards $r_{\text{force}}$ and $r_{\text{con}}$ are granted per checkpoint region rather than per step. The updated reward function is similar to equation (1) but with an additional indicator function:

$$
r=\begin{cases}r_{\text{col}}&\text{if collides},\\
\mathbf{I}_{\text{check}}(r_{\text{con}}+r_{\text{force}})+r_{\text{way}}+r_{%
\text{ac}}&\text{otherwise}.\end{cases}
$$

Now, two positive terms, $r_{\text{con}}$ and $r_{\text{force}}$ are controlled by the checkpoint indicator $\mathbf{I}_{\text{check}}$, which limits the occurrence of those terms to the number of the checkpoints. This gives us a direct way to bound the cumulative reward $\text{R}_{\text{forever}}$, which is re-defined from $\sum_{t=0}^{\infty}\gamma^{t}W_{q}^{\text{max}}$ to $\sum_{t=0}^{T_{2}}\gamma^{t}W_{q1}^{\text{max}}+\sum_{t={T_{2}+1}}^{\infty}%
\gamma^{t}W_{q2}^{\text{max}}$, where $\text{T}_{2}$ is approximated by the time to traverse each checkpoint region only once. $W_{q1}^{\text{max}}$ is identical to $W_{q}^{\text{max}}$ within checkpoint regions, but $W_{q2}^{\text{max}}<0$ only contains acceleration penalties when all checkpoint regions have been visited. And hence $\text{L}(\text{W}_{q}^{\text{max}})$ is re-defined as:

$$
\text{L}(\text{W}_{q}^{\text{max}})=(\sum_{t={T_{2}+1}}^{\infty}\gamma^{t}W_{q%
2}^{\text{max}})/\gamma^{T_{2}}
$$

Given $\text{L}(\text{W}_{q}^{\text{max}})\ll 0$, equation 5 holds for $0<\text{W}_{T}\ll 99\text{W}_{q}^{\text{max}}$, altering the policy convergence landscape. Our experiments demonstrate this effectively prevents the convergence to perpetual wiping, as elaborated in Section V-B.

## IV Visual-Language Model based Curriculum

![Refer to caption](https://arxiv.org/html/2502.12599v1/extracted/6212486/figs/diagram-0915.png)

Figure 3: Diagram of the Proposed VLM Algorithms, simulating human decision process on reward scale engineering.

While the new formulation makes the quality-critical problem feasible, learning is still hyperparameter sensitive. To ensure successful trajectories exist and hence can be learned subsequently, we propose a novel Vision-Language Models (VLM) based curriculum learning system, which automatically monitors training metrics and adjusts relative weights of reward terms during the learning process, which resembles the parameter tuning process of human experts.

### IV-A VLM-based Curriculum

Our learning framework calls the VLM-based curriculum every K steps after the initial M training steps, where K and M are hyper parameters. The curriculum module auto-adjusts reward weights for the next cycle with following steps:

Step1: Inspection. In this step, our goal is to collect the initial set of information, which includes success rates, landing pressure profiles, and navigation pressure stats. These stats can be collected by expanding the rollout of the current policy $\pi$ for $N$ times. We maintain the history of the previous information for reference. Once the information is collected, the system checks the pre-defined predicates (e.g., force variance decreased without a significant reduction in navigation success rates) to see if it wants to call the VLM-based hyperparameter tuning.

Step2: Update In this step, there are two large model agents involved: a LLM agent and a VLM agent. The LLM takes in provided metric from Step 1 and updates reward weights. Depending on the training progress, the LLM could request for different extra information before updating. If the completion rate is low, vision feedback of ending scene summarized by a separate VLM will be provided to describe failure reasons (e.g., no contact, or close to endpoint without finish wiping). If the force metrics require improvements, detailed force percentiles will be provided. This step is desired with multiple purposes: 1) Only providing necessary details into prompts to avoid LLM’s catastrophic forgetting on important information. 2) Navigation failures can arise from various scenarios. Leveraging VLM’s semantic capabilities allows us to understand the causes of failures, reducing the need for labor-intensive monitoring and iterative metric development. 3) This hierarchical approach enhances system’s extensibility. 4) Separating LLM and VLM optimizes reasoning and visual data interpretation respectively.

The final metrics and extra information will be feed to the LLM. The output from LLM consists of two parts: 1) A 1-2 sentences step-by-step analysis on logs and focus-learning areas; 2) python code for updated reward parameters.

Detailed prompts can be found at our website noted in the abstract. The high-level description is summarized in Algorithm 1 and Figure 3.

Algorithm 1 VLM-based Curriculum Learning

Data: pre-trained LLM $L$ and VLM $V$

Data: a RL policy $\pi$

Data: a reward weights parameter vector $\mathbf{w}$

$d\leftarrow$ dict(), $i\leftarrow 0$

while not converged do

$\pi\leftarrow$ learn($\pi$, $\mathbf{w}$, K) $\triangleright$ Learn a policy for $K$ steps

$\triangleright$ Step 1. Inspection

$d\leftarrow$ eval($d$, $\pi$, $i$) $\triangleright$ Eval $\pi$ and update $i$ th iter data

if not $\text{maintain}()$ then

$\triangleright$ Step 2. Update

$d\leftarrow$ request\_extra\_info\_if\_needed($V$, $d$, $i$)

$\mathbf{w}\leftarrow$ update\_reward\_params($L$ $V$, $d$)

end if

 $i+=1$

end while

### IV-B Implementation details

We used gpt-4 [^26] as LLM and gpt-4-vision-preview [^27] as VLM.We implemented GPT-4 [^26] as the VLM. To ensure thorough exploration of initial parameters, we initiate our module at 300k steps. We evaluate $N=50$ episodes every $K=100k$ steps and invoke the LLM curriculum module unless evaluation metrics meet the maintenance criteria: an improvement in force profiles—defined by a mean force deviation from the target of less than 5N with reduced variance—without significantly compromising the navigation completion rate (a permissible change of less than 15%). Initially, $\text{W}_{T}=1000$ and $\text{W}_{q}^{\text{max}}=29$, which is far from the upper limit of the feasible range $0<\text{W}_{T}\ll 99\text{W}_{q}^{\text{max}}$ outlined in Section III-C. Optionally, researchers can clip the reward weights for each goal do not exceed twice the weight of any other goal for safeguard. Throughout the fine-tuning process, the ratio consistently remains within this range.

## V Results

We design our experiments to answer the questions below.

1. Can our methodology effectively train a quality-critical wiping policy for various surfaces?
2. Can (i) Bounded Reward Design, and (ii) VLM-based curriculum improve the learning effectiveness?

### V-A Experiment Setup

Our simulation environments are built on top of Mujoco [^28] and robosuite [^29]. We use the 7-DoF Panda as our robot model, a common choice for both simulated and real-robot research. The trained policies control a robosuite pose controller module using OSC\_POSE option at 20 Hz.

Fig. 1 illustrates the robot arm performing wiping tasks in various simulated environments. Utilizing domain randomization [^30] for effective Sim2Real Transfer, we generate diverse simulated settings randomly sampled at the beginning of each training episode.with randomized attributes to train a model effective across all. Key properties varied include:

1. Curvature: Six tabletops with varying curvature (1 flat, 5 curved) to cover a range of surface shapes. The most curved one was created first, and scaled down the z-axis uniformly (flat, 0.2x, 0.4x, 0.6x, 0.8x).
2. Textures: Sliding ($\mathcal{N}(0.30,0.05)$), torsional ($\mathcal{N}(0.06,0.02)$), rolling ($\mathcal{N}(0.0125,0.005)$) frictions are modeled as Gaussian distributions.
3. Waypoints: We randomize the location of two waypoints on the tabletop.

We do not include these randomization parameter when we designed curriculum learning.

For analysis, we run experiments for three methods:

1. non-bounded-reward: The baseline formulation without the bounded reward defined in Fig 2. To balance both objectives, the reward for navigation completion is scaled to match the cumulative wiping quality rewards of the expected completion steps.The baseline formulation without the bounded reward defined by checkpoint regions in Fig 2. Rewards are carefully crafted to balance navigation completion and force learning.
2. bounded-reward: The formulation inherits the same reward scales from non-bounded-reward, but incorporated the checkpoint regions as shown in Fig 2.
3. bounded-llm-curr (ours): An extended formulation from bounded-reward with VLM-based curriculum discussed in Section IV. We initialize the learning with the same reward scales, which are adjusted by language models during trainingsubsequently modified by VLM agents to enhance learning outcomes.

### V-B Main Results

Our approach effectively trains a wiping policy to navigate waypoints on surfaces with varied curvature and smoothness, while ensuring force remains centered around a target of 60N. Fig 4 demonstrates the successful training outcomes of bounded-llm-curr. It achieves a high navigation completion rate, maintaining stable force control. To illustrate the quality of wiping, we visualize two examples of successful trajectories with different table properties in Fig 1, which are nicely centered around our target pressure values $60$ N. After 800k steps of training, the policy is able to achieve an average 98% navigation success rate, and 243 Integral Absolute Error (defined as $\text{IAE}=\int_{0}^{\infty}|e(f)|\,dt,$) with an average of 25 steps.

![Refer to caption](https://arxiv.org/html/2502.12599v1/extracted/6212486/figs/complete-v3.png)

(a) Navigation Complete

![Refer to caption](https://arxiv.org/html/2502.12599v1/extracted/6212486/figs/reasoning_examples.png)

Figure 5: Examples of VLM-based curriculum adjustment based on the training progresses. Each performance segment includes navigation success rate, average landing pressure (up) and navigational pressure (down). The target pressure is 60N.

| Method | Success | Steps | $\textbf{f}_{z}$ IAE |
| --- | --- | --- | --- |
| non-bounded-reward | 58% | 38 | 267 |
| bounded-reward | 92% | 29 | 333 |
| bounded-llm-curr | 98% | 25 | 243 |

TABLE I: Evaluation metrics averaged across 5 random seeds. From left to right: navigation success rate; completion steps; IAE of navigational forces.

Fig 4(a) and Table I show the non-bounded-reward method yields around 60% navigation completion rates, primarily due to suboptimal convergence in four of five seeds, demonstrating as persistent wiping behavior (Section III-B) in half the cases. We observe no policy converged to such behavior once we introduce bounded reward design as we intended, and hence the navigation success raised significantly from 58% to 92%. Further enhancements via a VLM-based curriculum (bounded-llm-curr) increased this rate to 98%, also optimizing average navigational force accuracy to the target value (60N), reducing Integral Absolute Error ($\text{IAE}=\int_{0}^{\infty}|e(f)|dt,$), shortening completion times, and decreasing landing forces. This strategy effectively trained policies to achieve force control comparable to non-bounded-reward, which prioritized quality at the expense of completion rates, without compromising on the latter.

### V-C Updates and Benefits of VLM Based CurriculumAnalysis

#### V-C1 Efficient fine-tuning with reasoning

This section discusses how the system responds to various input metrics and avoids potential local optima for superior solutions, using Fig 5 as examples. In Scenario 1, when the navigation completion rate is low, the LLM agent increases navigation rewards, enhancing the gradient signals for this metric at the expense of increased landing forces - potentially encouraging successful landings regardless of costs. However, since this occurs early in the training, the RL agent can dedicate the remaining episodes to mastering force control. In Scenario 2, landing force is challenging to learn due to sparse sampling (one per episode). In later training stages, the LLM agent adjusts the penalty multiplier for landing forces, significantly reducing them without adversely affecting other metrics. Combined adjustments lead to better results in Table I. To further validate the system, we initiated a set of experiments with imbalanced weight initialization where navigation completion rewards were only 10% of wiping quality rewards. With bounded-reward, success rates remained near zero even after 600k steps. However, bounded-llm-curr effectively corrected this undesired initialization during the early exploration phase, included successful trajectories, and increased the success rate to 40% by 500k steps.

![Refer to caption](https://arxiv.org/html/2502.12599v1/extracted/6212486/figs/visual-description.png)

Figure 6: An example of automatic visual feedback

#### V-C2 Visual Monitoring over Failed Behaviors

Fig 6 illustrates how the VLM component summarizes failure reasons. Typically, identifying such open-ended failures requires domain knowledge, iterative monitoring, and extensive logging. In this case, VLM identified the failure occurred early, before contact with the table, leading to a subsequent increase in the intermediate reward for wiping the first waypoint. This example demonstrates the potential of VLMs to enhance understanding in scenarios where the fundamental learning tasks are more complex.

## VI Conclusion

This paper presents two techniques for learning effective wiping policies: bounded reward formulation and VLM-based curriculum learning. Initially, we demonstrate the infeasibility of the naive step reward formulation and introduce a bounded approach that improves feasibility. Our novel VLM system actively monitors and adjusts reward weights during learning. Experimental results confirm the efficacy of these methods. We aim to follow up and address current limitations: 1) enhancing the VLM system’s generalizability in complex scenarios beyond wiping; 2) deploying policies to hardware to validate real-world performance; and 3) autonomously generating waypoints from observations, thus eliminating the assumption of available waypoints.

## Acknowledgement

This work was supported by grants from the Electronics and Telecommunications Research Institute (ETRI) \[24ZD1130/24BD1300\]. We also want to thank Jiachen Yang for his thorough proofreading and insightful feedback.

[^1]: Y. Qian, J. Yuan, S. Bao, and L. Gao, “Sensorless hybrid normal-force controller with surface prediction,” in *2019 IEEE International Conference on Robotics and Biomimetics (ROBIO)*. IEEE, 2019.

[^2]: C.-Y. Lin, C.-C. Tran, S. H. Shah, and A. R. Ahmad, “Real-time robot pose correction on curved surface employing 6-axis force/torque sensor,” *IEEE Access*, vol. 10, pp. 90 149–90 162, 2022.

[^3]: J. Li, Y. Guan, H. Chen, B. Wang, T. Zhang, J. Hong, and D. Wang, “Real-time normal contact force control for robotic surface processing of workpieces without a priori geometric model,” *The International Journal of Advanced Manufacturing Technology*, pp. 1–15, 2022.

[^4]: M. Amersdorfer, J. Kappey, and T. Meurer, “Real-time freeform surface and path tracking for force controlled robotic tooling applications,” *Robotics and Computer-Integrated Manufacturing*, vol. 65, p. 101955, 2020.

[^5]: M. Iskandar, C. Ott, A. Albu-Schäffer, B. Siciliano, and A. Dietrich, “Hybrid force-impedance control for fast end-effector motions,” *IEEE Robotics and Automation Letters*, vol. 8, no. 7, pp. 3931–3938, 2023.

[^6]: L. F. Vázquez-Alberto and M. A. Arteaga, “A continuous terminal sliding mode algorithm for robot manipulators: an application to force control,” *International Journal of Control*, vol. 96, no. 11, pp. 2812–2826, 2023.

[^7]: J. Hess, J. Sturm, and W. Burgard, “Learning the state transition model to efficiently clean surfaces with mobile manipulation robots,” in *Proc. of the Workshop on Manipulation under Uncertainty at the IEEE International Conference on Robotics and Automation (ICRA)*, 2011.

[^8]: S. Elliott and M. Cakmak, “Robotic cleaning through dirt rearrangement planning with learned transition models,” in *2018 IEEE International Conference on Robotics and Automation (ICRA)*, 2018.

[^9]: J. Yin, K. G. S. Apuroop, Y. K. Tamilselvam, R. E. Mohan, B. Ramalingam, and A. V. Le, “Table cleaning task by human support robot using deep learning technique,” *Sensors*, vol. 20, no. 6, p. 1698, 2020.

[^10]: N. Cauli, P. Vicente, J. Kim, B. Damas, A. Bernardino, F. Cavallo, and J. Santos-Victor, “Autonomous table-cleaning from kinesthetic demonstrations using deep learning,” in *2018 Joint IEEE 8th International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob)*. IEEE, 2018, pp. 26–32.

[^11]: J. Kim, N. Cauli, P. Vicente, B. Damas, F. Cavallo, and J. Santos-Victor, ““icub, clean the table!” a robot learning from demonstration approach using deep neural networks,” in *2018 IEEE international conference on autonomous robot systems and competitions (ICARSC)*.

[^12]: T. Lew, S. Singh, M. Prats, J. Bingham, J. Weisz, B. Holson, X. Zhang, V. Sindhwani, Y. Lu, F. Xia, *et al.*, “Robotic table wiping via reinforcement learning and whole-body trajectory optimization,” in *2023 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2023, pp. 7184–7190.

[^13]: M. Suomalainen, Y. Karayiannidis, and V. Kyrki, “A survey of robot manipulation in contact,” *Robotics and Autonomous Systems*, 2022.

[^14]: C. S. Zapico, Y. Petillot, and M. S. Erden, “Semi-autonomous surface-tracking tasks using omnidirectional mobile manipulators,” in *2024 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2024, pp. 2176–2182.

[^15]: S. Elliott, Z. Xu, and M. Cakmak, “Learning generalizable surface cleaning actions from demonstration,” in *2017 26th IEEE international symposium on robot and human interactive communication (RO-MAN)*. IEEE, 2017, pp. 993–999.

[^16]: T. Zhang, M. Xiao, Y.-b. Zou, J.-d. Xiao, and S.-y. Chen, “Robotic curved surface tracking with a neural network for angle identification and constant force control based on reinforcement learning,” *International Journal of Precision Engineering and Manufacturing*, 2020.

[^17]: K. Kawaharazuka, N. Kanazawa, K. Okada, and M. Inaba, “Learning-based wiping behavior of low-rigidity robots considering various surface materials and task definitions,” in *2022 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids)*. IEEE, 2022, pp. 919–924.

[^18]: N. Saito, D. Wang, T. Ogata, H. Mori, and S. Sugano, “Wiping 3d-objects using deep learning model based on image/force/joint information,” in *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2020, pp. 10 152–10 157.

[^19]: N. Saito, T. Shimizu, T. Ogata, and S. Sugano, “Utilization of image/force/tactile sensor data for object-shape-oriented manipulation: Wiping objects with turning back motions and occlusion,” *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 968–975, 2021.

[^20]: A. Gams, T. Petrič, M. Do, B. Nemec, J. Morimoto, T. Asfour, and A. Ude, “Adaptation and coaching of periodic motion primitives through physical and visual interaction,” *Robotics and Autonomous Systems*, vol. 75, pp. 340–351, 2016.

[^21]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, *et al.*, “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” *arXiv preprint arXiv:2307.15818*, 2023.

[^22]: Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward design via coding large language models,” *arXiv preprint arXiv:2310.12931*, 2023.

[^23]: T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu, “Text2reward: Automated dense reward function generation for reinforcement learning,” *arXiv preprint arXiv:2309.11489*, 2023.

[^24]: W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, M. G. Arenas, H.-T. L. Chiang, T. Erez, L. Hasenclever, J. Humplik, *et al.*, “Language to rewards for robotic skill synthesis,” *preprint arXiv:2306.08647*, 2023.

[^25]: K. Kurach, A. Raichuk, P. Stanczyk, M. Zajkac, O. Bachem, L. Espeholt, C. Riquelme, D. Vincent, M. Michalski, O. Bousquet, *et al.*, “Google research football: A novel reinforcement learning environment,” in *Proceedings of the AAAI conference on artificial intelligence*, vol. 34, no. 04, 2020, pp. 4501–4510.

[^26]: J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, *et al.*, “Gpt-4 technical report,” *arXiv preprint arXiv:2303.08774*, 2023.

[^27]: OpenAI, “Gpt-v system card,” OpenAI, Tech. Rep., 2023, accessed: INSERT-DATE-OF-ACCESS. \[Online\]. Available: [https://cdn.openai.com/papers/GPTV˙System˙Card.pdf](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

[^28]: E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in *2012 IEEE/RSJ international conference on intelligent robots and systems*. IEEE, 2012, pp. 5026–5033.

[^29]: Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, S. Nasiriany, and Y. Zhu, “robosuite: A modular simulation framework and benchmark for robot learning,” *preprint arXiv:2009.12293*, 2020.

[^30]: J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from simulation to the real world,” in *2017 IEEE/RSJ international conference on intelligent robots and systems (IROS)*. IEEE, 2017, pp. 23–30.