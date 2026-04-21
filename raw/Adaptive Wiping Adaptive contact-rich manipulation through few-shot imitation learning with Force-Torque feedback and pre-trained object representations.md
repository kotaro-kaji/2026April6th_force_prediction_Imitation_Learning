Chikaha Tsuji <sup>1∗</sup>, Enrique Coronado <sup>2</sup>, Pablo Osorio <sup>3</sup> and Gentiane Venture <sup>4,2∗</sup> <sup>1</sup> Department of Mechano-Informatics, The University of Tokyo, Japan. tsujchik@g.ecc.u-tokyo.ac.jp <sup>2</sup> National Institute of Advanced Industrial Science and Technology, Japan <sup>3</sup> Department of Mechanical Systems Engineering, Tokyo University of Agriculture and Technology, Japan. <sup>4</sup> Department of Mechanical Engineering, The University of Tokyo, Japan. venture@g.ecc.u-tokyo.ac.jp \*Corresponding authors

###### Abstract

Imitation learning offers a pathway for robots to perform repetitive tasks, allowing humans to focus on more engaging and meaningful activities. However, challenges arise from the need for extensive demonstrations and the disparity between training and real-world environments. This paper focuses on contact-rich tasks like wiping with soft and deformable objects, requiring adaptive force control to handle variations in wiping surface height and the sponge’s physical properties. To address these challenges, we propose a novel method that integrates real-time force-torque (FT) feedback with pre-trained object representations. This approach allows robots to dynamically adjust to previously unseen changes in surface heights and sponges’ physical properties. In real-world experiments, our method achieved 96% accuracy in applying reference forces, significantly outperforming the previous method that lacked an FT feedback loop, which only achieved 4% accuracy. To evaluate the adaptability of our approach, we conducted experiments under different conditions from the training setup, involving 40 scenarios using 10 sponges with varying physical properties and 4 types of wiping surface heights, demonstrating significant improvements in the robot’s adaptability by analyzing force trajectories. [https://sites.google.com/view/adaptive-wiping](https://sites.google.com/view/adaptive-wiping)

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/task.png)

Refer to caption

## I Introduction

Robots are crucial for handling mundane tasks, but pre-programming each task is impractical, leading to increased interest in imitation learning [^1]. Despite its benefits, challenges like the need for extensive demonstrations and discrepancies between training and real-world environments persist [^2]. Thus, robots must not merely mimic but adapt to new environments, even with limited demonstration data.

A challenging aspect of robotic manipulation is executing contact-rich tasks, which involve extensive physical interactions. Interestingly, those involving deformable objects pose particular challenges due to the need for precise force control and adaptation to changes [^3]. Wiping tasks, for example, demand careful force adjustments based on wiping surface height and sponge’s physical properties.

Therefore, in this paper, we address the challenge – *Could robots learn a versatile manipulation policy via few-shot imitation learning capable of adapting to environmental changes: the height of manipulating surface and the physical properties of manipulated objects?*

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/overview.png)

Figure 2: Overview of our proposed framework. First, we pre-train the sponge properties encoder ϕ sponge subscript italic-ϕ \\phi\_{\\text{sponge}} italic\_ϕ start\_POSTSUBSCRIPT sponge end\_POSTSUBSCRIPT using simulated unlabeled data (Pre-training step III-A ). Then, we train the motion trajectory decoder θ traj 𝜃 \\theta\_{\\text{traj}} italic\_θ start\_POSTSUBSCRIPT traj end\_POSTSUBSCRIPT and the FT feedback loop ft − height \\phi\_{\\text{ft}}-\\theta\_{\\text{height}} italic\_ϕ start\_POSTSUBSCRIPT ft end\_POSTSUBSCRIPT - italic\_θ start\_POSTSUBSCRIPT height end\_POSTSUBSCRIPT to obtain the wiping policy with the active inference of applied force using few-shot human demonstration data (Training step III-B ). Finally, we deploy the acquired policy on real robot hardware (Deployment III-C ).

## II Related Works and Contribution

Learning-based methods have succeeded in addressing contact-rich tasks. Reinforcement learning is one of the key learning-based methods for acquiring desired behaviors by defining a reward function. Martín-Martín et al. succeeded in completing a contact-rich task based on variable impedance control in end-effector space via reinforcement learning [^4]. Spector et al. proposed a residual admittance policy that is learned to correct the difference from the reference policy using reinforcement learning, achieving a contact-rich assembly task [^5]. However, reinforcement learning-based methods rely heavily on the reward function’s design and are inefficient with few samples.

In contrast, imitation learning offers a different approach by acquiring desired behaviors through demonstrations, yielding higher sample efficiency without carefully designing the reward function. Rozo et al. used Gaussian mixture models and variable impedance control to accomplish human-robot cooperative transportation tasks [^6]. Yamane et al. used bilateral control-based imitation learning to decouple the applied force from humans and the reaction force from the environment, enabling a robot to grasp various objects with a custom-made cross-structure hand [^7].

Tasks involving deformable objects are especially challenging due to the need for precise force control and adaptability to changing conditions. To address this, multiple studies combined representation learning to acquire object property embeddings prior to demonstrations. [^8] and [^9] pre-trained representations based on visual observations using self-supervised learning, while Guzey et al. used tactile observation for representation learning [^10]. In contrast, several works suggest the importance of haptic time-series information in capturing objects’ physical properties [^11].

Real-world data collection is costly and time-consuming, while simulation offers a more efficient alternative. However, Sim2Real transfer poses challenges due to differences between simulated and real-world environments. Domain randomization mitigates this gap by introducing variability in parameters like lighting and object textures during simulation, improving model robustness [^12]. Tobin et al. used domain randomization to train an object detector in simulation for robotic grasping [^13]. Beyond visual domain randomization, dynamics randomization, which involves randomizing physical properties like mass and friction, has been explored to improve real-world generalization [^14]. Domain randomization has also been applied to manipulation tasks with deformable objects in [^15].

Aoyama et al. used self-supervised learning on force and torque data, along with dynamics domain randomization, to capture the physical properties of deformable objects. They successfully transferred these representations from simulation to reality, enabling effective force control via few-shot imitation learning [^16].

However, they controlled the wiping motions in an open-loop manner. Thus, the approach could not adapt to environmental changes, such as variations in wiping surface height. In contrast, methods like impedance control [^17] or AC [^18] are well-established for closed-loop force control. However, in our context, both the target position (affected by changes in surface heights) and the target force to be applied (influenced by variations in sponge properties) are unknown, rendering these methods unsuitable. Therefore, a different approach is needed to apply the appropriate force while adapting to changes in wiping surface heights and sponge physical properties.

We addressed these challenges with three contributions:

- We propose a framework that combines pre-training to represent the physical properties of manipulated objects with real-time feedback of time-series force-torque (FT) information, enabling the robot’s adaptation to environmental changes from a small number of human demonstrations.
- In contrast to the open-loop control method used in [^16] for wiping motions, our approach extends this by incorporating a closed-loop control strategy. This advancement allows a robot to dynamically adapt to environmental variations, such as changes in the height of the wiping surface. Unlike admittance and impedance control methods, our approach is particularly advantageous for handling deformable and elastic objects, as it can adapt to the physical properties of unseen sponges and surface height variations without requiring prior information like target force or position.
- We validate our approach on real hardware by altering the height of the wiping surface and the physical properties of the sponge in a wiping task, showcasing the ability to adapt to unseen environmental conditions by analyzing force measurements.

## III Methods

The proposed method consists of two steps: a pre-training step using a simulator and a training step using a real robot, before being deployed (Fig. 2), each step is detailed below.

### III-A Pre-training step

We pre-train the sponge properties encoder $\phi_{\text{sponge}}$ on simulated unlabeled data $D_{\text{sim}}=\{(\tau^{\text{exp}})_{1},\ldots,(\tau^{\text{exp}})_{M}\}$, collected by performing pre-defined exploratory actions (detailed in IV-C1), to capture the sponges’ physical properties as the latent space $Z_{\text{sponge}}$ covering a wide range of the underlying distribution. We use a self-supervised learning framework inspired by [^16] but with a modified architecture.

Using a Variational Autoencoder (VAE) [^19] approach, the VAE encoder-decoder model $\phi_{\text{sponge}}-\theta_{\text{sponge}}$ takes FT trajectory $\tau^{\text{exp}}$ from $D_{\text{sim}}$ as inputs and outputs reconstructed FT trajectory $\hat{\tau}^{\text{exp}}$, treating the latent space $Z_{\text{sponge}}$ as a Gaussian distribution with five dimensions.

The VAE encoder-decoder model $\phi_{\text{sponge}}-\theta_{\text{sponge}}$ consists of 2 fully connected encoder layers, 1 sampling step, and 2 fully connected decoder layers. To flatten the six sensors’ time-series data $\tau^{\text{exp}}\in\mathbb{R}^{400\times 6}$, we employ 2 fully connected layers each for the encoder $\phi_{\text{sponge}}$ and decoder $\theta_{\text{sponge}}$. The encoder $\phi_{\text{sponge}}$ comprises 1 fully connected layer of 5 hidden dimensions followed by the flattening step and 1 fully connected layer. Whereas the decoder $\theta_{\text{sponge}}$ comprises 1 fully connected layer with Rectified Linear Unit (ReLU) as an activation function and a dropout rate of 0.1 followed by a reshaping step and 1 fully connected layer of 5 hidden dimensions. The latent space dimension $Z_{\text{sponge}}\in\mathbb{R}^{5}$ is designed to capture sponges’ stiffness, friction, and other non-intuitive physical properties. We adopt a loss function $L_{ssl}$ shown in Eq. (1), with $\beta=0.06$.

$$
\displaystyle L_{ssl}
$$
 
$$
\displaystyle=\;E_{\text{MSE}}(\hat{\tau}^{\text{exp}},\;\tau^{\text{exp}})\hfill
$$
 
$$
\displaystyle+\;\beta D_{\text{KL}}(q_{\phi_{\text{sponge}}}(z\;|\;\tau^{\text%
{exp}})\;||\;p_{\phi_{\text{sponge}}}(z))\hfill
$$

### III-B Training step

We train the motion trajectory decoder $\theta_{\text{traj}}$ and the FT feedback loop $\phi_{\text{ft}}-\theta_{\text{height}}$ on real-world unlabeled data $D_{\text{real}}=\{\tau^{\text{exp}}\}$, collected by the same pre-defined exploratory actions with III-A, and few-shot human demonstration data $D_{\text{demo}}=\{(x^{\text{demo}},\Delta h^{\text{demo}},\tau^{\text{demo}})_%
{1},\ldots,(x^{\text{demo}},\Delta h^{\text{demo}},\tau^{\text{demo}})_{N}\}$.

#### III-B1 Motion trajectory decoder θtrajsubscript𝜃traj\\theta\_{\\text{traj}}italic\_θ start\_POSTSUBSCRIPT traj end\_POSTSUBSCRIPT

We train the wiping motion trajectory decoder $\theta_{\text{traj}}$ using Learning from Demonstration (LfD) [^16] to generate the wiping motion $\hat{x}^{\text{task}}$ according to the manipulated sponge properties.

The encoder-decoder model $\phi_{\text{sponge}}-\theta_{traj}$ takes FT trajectory $\tau^{\text{exp}}$ from $D_{\text{real}}$ as inputs and outputs the corresponding motion trajectory $\hat{x}^{\text{demo}}$. Here, the encoder $\phi_{\text{sponge}}$ is pre-trained on simulated data $D_{\text{sim}}$, with its weights frozen during training on real data, and then deployed in the real world (Sim2Real).

The motion trajectory decoder $\theta_{\text{traj}}$ consists of 1 fully connected layer with a dropout rate of 0.1. We adopt the Mean Squared Error $L_{traj}$ between the generated motion trajectory $\hat{x}^{\text{demo}}$ and the demonstrated one $x^{\text{demo}}$ represented in the absolute coordinate from the base link (Eq. (2)).

$$
L_{motion}=E_{\text{MSE}}(\hat{x}^{\text{demo}}\,,\,x^{\text{demo}})
$$

#### III-B2 FT feedback loop ϕft−θheightsubscriptitalic-ϕftsubscript𝜃height\\phi\_{\\text{ft}}-\\theta\_{\\text{height}}italic\_ϕ start\_POSTSUBSCRIPT ft end\_POSTSUBSCRIPT - italic\_θ start\_POSTSUBSCRIPT height end\_POSTSUBSCRIPT

We train an FT feedback loop $\phi_{\text{ft}}-\theta_{\text{height}}$ composed of the FT encoder $\phi_{\text{ft}}$ and the end-effector’s vertical position decoder $\theta_{\text{height}}$ to obtain a control input of the next time step’s vertical position according to the contact state and the manipulated sponge.

The FT encoder $\phi_{\text{ft}}$ processes the FT history from the demonstrations $D_{\text{demo\_ft}}=\{\tau^{\text{demo}}_{\text{t-4}},\ldots,\tau^{\text{demo}%
}_{\text{t}}\}$, encoding it into the latent space $Z_{\text{ft}}\in\mathbb{R}^{6}$, which is designed to represent the forces and torques along the x, y, and z axes. The end-effector’s vertical position decoder $\theta_{\text{height}}$ takes the concatenated latent spaces $Z_{\text{sponge}}$ from the sponge properties encoder $\phi_{\text{sponge}}$ and $Z_{\text{ft}}$ from the FT encoder $\phi_{\text{ft}}$ as inputs, and outputs the next time step’s vertical displacement $\Delta\hat{h}^{\text{demo}}_{\text{t+1}}$.

The FT encoder $\phi_{\text{ft}}$ consists of 2 layers of temporal convolutional network (TCN) [^20] with 25 hidden channels each and a dropout rate of 0.1. Inspired by [^21], which suggests that TCN has advantages in training efficiency and training time over gated recurrent units (GRU) [^22], we adopt TCN as our sequence model. The end-effector’s vertical position decoder $\theta_{\text{height}}$ consists of 2 fully connected layers: the first fully connected layer of 128 hidden dimensions with ReLU as an activation function and a dropout rate of 0.1 followed by the final layer (the second fully connected layer). We adopt the Mean Squared Error $L_{height}$ between the predicted vertical displacement in the next time step $\Delta\hat{h}^{\text{demo}}_{\text{t+1}}$ and that of the ground truth $\Delta h^{\text{demo}}_{\text{t+1}}$ (Eq. (3)).

$$
L_{height}=E_{\text{MSE}}(\Delta\hat{h}^{\text{demo}}_{\text{t+1}}\,,\,\Delta{%
h}^{\text{demo}}_{\text{t+1}})
$$
![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/execution.png)

Figure 3: Manipulation processes of 3 different settings (low, high, sloped) using an unseen sponge that was not included in the training data. The right plots show FT profiles. The baseline simply traces the demonstration and reproduces vertical motion without considering setting changes (gray). In contrast, our method adapts to those changes while maintaining the desired wiping motion (red).

### III-C Deployment

In the task execution, the robot performs a wiping motion by combining offline horizontal (x, y) motion of $\hat{x}^{\text{task}}$ and online vertical (z) motion of $\Delta\hat{h}^{\text{task}}_{\text{t+1}}$. First, the robot collects unlabeled data $D_{\text{task}}=\{\tau^{\text{exp}}\}$ of the sponge being used in the task through pre-defined exploratory actions. Then it generates (x,y) planar motion $\hat{x}^{\text{task}}$ from $D_{\text{task}}$ and replays the motion offline. The FT feedback loop actively infers the next vertical position $\Delta\hat{h}^{\text{task}}_{\text{t+1}}$ from the previous $\sim$ current force and torque history $D_{\text{task\_ft}}=\{\tau^{\text{task}}_{\text{t-4}},\ldots,\tau^{\text{task}%
}_{\text{t}}\}$, and adapts online.

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/sponges.png)

Refer to caption

## IV Experiment Setup

### IV-A Wiping task

To illustrate our method, we use a contact-rich wiping task in which the robot has to adapt its wiping motion to the wiping surface height and the manipulated sponge’s physical properties. We prepare 3 variations of table heights (low, high, and sloped) and 10 sponges (one ready-made sponge (normal sponge) and 9 custom-made sponges (3 stiffness levels $\times$ 3 friction levels)) as shown in Fig. 4. We denote a sponge with a stiffness level $m$ and a friction level $n$ as ’s $m$ f $n$ ’ ($m,n=1,2,3$). For additional verification, we also prepare a vertical wall to replace the horizontal table.

Figure 5: Experimental results: Baselines and Ours under Various Conditions. The contact percentage indicates the proportion of time steps where force was applied to press a sponge and the number in () represents the ratio of the average force in the z-direction to that of the corresponding demonstrations (reference force) shown in Table 6.

<table><tbody><tr><td colspan="2" rowspan="2"><svg height="12.22" width="43.44"><g><path></path><g><g><foreignObject height="6.07" width="21.72"><span><span><span>Sponge</span> </span></span></foreignObject></g></g><g><g><foreignObject height="6.15" width="19.99"><span><span><span>Height</span></span></span></foreignObject></g></g></g></svg></td><td colspan="3">Low</td><td colspan="3">High</td><td colspan="3">Sloped</td><td colspan="3">Average</td></tr><tr><td>Contact</td><td>Average [N]</td><td>Std</td><td>Contact</td><td>Average [N]</td><td>Std</td><td>Contact</td><td>Average [N]</td><td>Std</td><td>Contact</td><td>Average [N]</td><td>Std</td></tr><tr><td rowspan="3">Normal</td><td>Baseline</td><td>12%</td><td>1.64 (-)</td><td>2.38</td><td>32%</td><td>-3.13 (25%)</td><td>8.73</td><td>24%</td><td>1.60 (-)</td><td>4.73</td><td>23%</td><td>8%</td><td>5.28</td></tr><tr><td>AC</td><td>100%</td><td>-6.79 (54%)</td><td>1.22</td><td>100%</td><td>-6.86 (54%)</td><td>1.02</td><td>100%</td><td>-6.65 (53%)</td><td>4.41</td><td>100%</td><td>54%</td><td>2.22</td></tr><tr><td>Ours</td><td>100%</td><td>-13.9 (110%)</td><td>3.50</td><td>100%</td><td>-12.5 (99%)</td><td>5.73</td><td>100%</td><td>-16.9 (133%)</td><td>9.12</td><td>100%</td><td>114%</td><td>6.12</td></tr><tr><td rowspan="3">s1f1</td><td>Baseline</td><td>0%</td><td>1.37 (-)</td><td>0.40</td><td>32%</td><td>-1.36 (6%)</td><td>4.28</td><td>12%</td><td>-0.14 (1%)</td><td>2.57</td><td>15%</td><td>2%</td><td>2.42</td></tr><tr><td>AC</td><td>100%</td><td>-5.84 (26%)</td><td>1.11</td><td>100%</td><td>-5.73 (25%)</td><td>1.24</td><td>100%</td><td>-6.68 (29%)</td><td>5.65</td><td>100%</td><td>27%</td><td>2.67</td></tr><tr><td>Ours</td><td>100%</td><td>-18.0 (80%)</td><td>11.9</td><td>100%</td><td>-22.1 (97%)</td><td>16.7</td><td>100%</td><td>-21.4 (94%)</td><td>15.2</td><td>100%</td><td>90%</td><td>14.6</td></tr><tr><td rowspan="3">s1f2</td><td>Baseline</td><td>0%</td><td>1.38 (-)</td><td>0.36</td><td>12%</td><td>-1.54 (7%)</td><td>6.22</td><td>12%</td><td>0.04 (-)</td><td>2.38</td><td>8%</td><td>2%</td><td>2.99</td></tr><tr><td>AC</td><td>100%</td><td>-5.67 (26%)</td><td>0.83</td><td>100%</td><td>-5.23 (24%)</td><td>0.77</td><td>100%</td><td>-5.25 (25%)</td><td>6.80</td><td>100%</td><td>25%</td><td>2.80</td></tr><tr><td>Ours</td><td>100%</td><td>-28.7 (134%)</td><td>18.0</td><td>100%</td><td>-25.8 (120%)</td><td>20.1</td><td>100%</td><td>-23.1 (108%)</td><td>17.0</td><td>100%</td><td>121%</td><td>18.4</td></tr><tr><td rowspan="3">s1f3</td><td>Baseline</td><td>0%</td><td>1.28 (-)</td><td>0.41</td><td>16%</td><td>-1.12 (5%)</td><td>6.13</td><td>12%</td><td>-0.20 (1%)</td><td>3.08</td><td>9%</td><td>2%</td><td>3.21</td></tr><tr><td>AC</td><td>100%</td><td>-5.23 (25%)</td><td>1.67</td><td>100%</td><td>-4.94 (23%)</td><td>1.57</td><td>100%</td><td>-6.06 (28%)</td><td>5.79</td><td>100%</td><td>25%</td><td>3.01</td></tr><tr><td>Ours</td><td>100%</td><td>-26.5 (124%)</td><td>14.6</td><td>100%</td><td>-20.1 (94%)</td><td>16.1</td><td>100%</td><td>-21.5 (101%)</td><td>16.2</td><td>100%</td><td>106%</td><td>15.6</td></tr><tr><td rowspan="3">s2f1</td><td>Baseline</td><td>0%</td><td>1.43 (-)</td><td>0.38</td><td>40%</td><td>-2.92 (12%)</td><td>7.21</td><td>12%</td><td>0.17 (-)</td><td>2.27</td><td>17%</td><td>4%</td><td>3.29</td></tr><tr><td>AC</td><td>100%</td><td>-11.4 (47%)</td><td>1.54</td><td>100%</td><td>-9.82 (41%)</td><td>2.17</td><td>100%</td><td>-7.44 (31%)</td><td>7.02</td><td>100%</td><td>40%</td><td>3.58</td></tr><tr><td>Ours</td><td>100%</td><td>-19.7 (82%)</td><td>9.97</td><td>100%</td><td>-23.3 (97%)</td><td>8.70</td><td>100%</td><td>-15.2 (63%)</td><td>4.95</td><td>100%</td><td>81%</td><td>7.87</td></tr><tr><td rowspan="3">s2f2</td><td>Baseline</td><td>0%</td><td>1.32 (-)</td><td>0.50</td><td>20%</td><td>-1.58 (5%)</td><td>5.75</td><td>12%</td><td>0.34 (-)</td><td>1.98</td><td>11%</td><td>2%</td><td>2.74</td></tr><tr><td>AC</td><td>100%</td><td>-11.7 (39%)</td><td>1.26</td><td>100%</td><td>-10.2 (34%)</td><td>1.49</td><td>100%</td><td>-11.3 (37%)</td><td>6.26</td><td>100%</td><td>37%</td><td>3.00</td></tr><tr><td>Ours</td><td>100%</td><td>-34.4 (115%)</td><td>14.6</td><td>100%</td><td>-23.3 (77%)</td><td>10.1</td><td>100%</td><td>-21.7 (72%)</td><td>7.80</td><td>100%</td><td>88%</td><td>10.8</td></tr><tr><td rowspan="3">s2f3</td><td>Baseline</td><td>0%</td><td>1.76 (-)</td><td>0.53</td><td>20%</td><td>-1.46 (4%)</td><td>6.74</td><td>12%</td><td>-0.07 (0%)</td><td>3.80</td><td>11%</td><td>1%</td><td>3.69</td></tr><tr><td>AC</td><td>100%</td><td>-12.1 (35%)</td><td>1.53</td><td>100%</td><td>-10.7 (31%)</td><td>1.83</td><td>100%</td><td>-11.1 (32%)</td><td>7.00</td><td>100%</td><td>33%</td><td>3.45</td></tr><tr><td>Ours</td><td>100%</td><td>-29.2 (85%)</td><td>14.1</td><td>100%</td><td>-24.5 (72%)</td><td>12.5</td><td>100%</td><td>-22.2 (65%)</td><td>6.86</td><td>100%</td><td>74%</td><td>11.2</td></tr><tr><td rowspan="3">s3f1</td><td>Baseline</td><td>0%</td><td>1.97 (-)</td><td>0.62</td><td>20%</td><td>-5.42 (18%)</td><td>12.9</td><td>12%</td><td>-0.35 (1%)</td><td>5.83</td><td>11%</td><td>6%</td><td>6.45</td></tr><tr><td>AC</td><td>100%</td><td>-18.7 (61%)</td><td>2.58</td><td>100%</td><td>-19.5 (63%)</td><td>5.90</td><td>100%</td><td>-16.8 (54%)</td><td>9.58</td><td>100%</td><td>59%</td><td>6.02</td></tr><tr><td>Ours</td><td>100%</td><td>-30.0 (97%)</td><td>15.4</td><td>100%</td><td>-24.5 (80%)</td><td>10.5</td><td>100%</td><td>-39.3 (127%)</td><td>13.3</td><td>100%</td><td>101%</td><td>13.1</td></tr><tr><td rowspan="3">s3f2</td><td>Baseline</td><td>0%</td><td>1.18 (-)</td><td>0.37</td><td>20%</td><td>-4.71 (13%)</td><td>11.4</td><td>12%</td><td>-1.07 (3%)</td><td>5.11</td><td>11%</td><td>5%</td><td>5.63</td></tr><tr><td>AC</td><td>100%</td><td>-19.7 (56%)</td><td>3.26</td><td>100%</td><td>-22.1 (63%)</td><td>4.24</td><td>100%</td><td>-19.0 (54%)</td><td>8.31</td><td>100%</td><td>58%</td><td>5.27</td></tr><tr><td>Ours</td><td>100%</td><td>-37.0 (105%)</td><td>8.27</td><td>100%</td><td>-29.6 (84%)</td><td>8.31</td><td>100%</td><td>-31.7 (90%)</td><td>4.85</td><td>100%</td><td>93%</td><td>7.14</td></tr><tr><td rowspan="3">s3f3</td><td>Baseline</td><td>0%</td><td>1.11 (-)</td><td>0.46</td><td>44%</td><td>-7.66 (21%)</td><td>17.3</td><td>24%</td><td>-2.43 (7%)</td><td>7.24</td><td>23%</td><td>9%</td><td>8.33</td></tr><tr><td>AC</td><td>100%</td><td>-21.6 (59%)</td><td>2.35</td><td>100%</td><td>-22.8 (62%)</td><td>2.35</td><td>100%</td><td>-22.0 (60%)</td><td>9.61</td><td>100%</td><td>60%</td><td>4.77</td></tr><tr><td>Ours</td><td>100%</td><td>-45.2 (123%)</td><td>8.97</td><td>100%</td><td>-28.1 (76%)</td><td>9.39</td><td>100%</td><td>-28.1 (77%)</td><td>4.39</td><td>100%</td><td>92%</td><td>7.58</td></tr><tr><td rowspan="3">Average</td><td>Baseline</td><td>1%</td><td>0%</td><td>0.64</td><td>26%</td><td>11%</td><td>8.67</td><td>14%</td><td>1%</td><td>3.90</td><td>14%</td><td>4%</td><td>4.40</td></tr><tr><td>AC</td><td>100%</td><td>43%</td><td>1.74</td><td>100%</td><td>42%</td><td>2.26</td><td>100%</td><td>40%</td><td>7.04</td><td>100%</td><td>42%</td><td>3.68</td></tr><tr><td>Ours</td><td>100%</td><td>106%</td><td>11.9</td><td>100%</td><td>90%</td><td>11.8</td><td>100%</td><td>93%</td><td>9.97</td><td>100%</td><td>96%</td><td>11.2</td></tr></tbody></table>

### IV-B Robot and Setup

We use a 6 DoF UR5 e-series robot arm with a 6-axis FT sensor and a sponge attached to its end-effector for both simulation and real robot experiments. We control the robot by specifying the end-effector position when performing pre-defined exploratory actions for collecting unlabeled data and deploying our proposed method. We conduct demonstrations by moving the robot arm kinesthetically in free drive mode. For the simulation, we use robosuite [^23] with the same setup as the real robot experiments, and for controlling the real hardware, the Robot Operating System (ROS) [^24] with the Universal Robot ROS Driver is used.

### IV-C Dataset

#### IV-C1 Unlabeled data

The robot performs two pre-defined exploratory actions [^16], where each action is carefully designed to capture the characteristics of sponges’ stiffness and friction, effectively bridging the Sim2Real transfer. These actions consist of pressing at $0.01m/s$ for 2 seconds and moving laterally left and right at $0.05m/s$ for 1 second each. During these exploratory actions, we record the 3-axis force and torque for $4s$ at a frequency of $100Hz$ while performing exploratory actions to obtain the FT trajectory $\tau^{\text{exp}}\in\mathbb{R}^{400\times 6}$.

We collect 1000 unlabeled data in simulation for pre-training by varying sponge properties. We randomized the parameters of stiffness, friction, and damping by setting sliding friction $\mu\in[0.0,3.5]$, solref stiffness $k\in[0.5,1000]$ N/m, and solimp width $d\in[0.02,0.3]$ m, to narrow the gap between simulation and reality (dynamics domain randomization). For training, we collect 1 demonstration unlabeled data of a normal sponge. ‘Normal’ refers to typical friction, stiffness, and damping properties in ready-made sponges.

The FT trajectories of the unlabeled data collected both in simulation and in the real world were similar. This is likely due to careful tuning of the dynamics-related parameters of the sponge in the simulator and the inherent elasticity of the sponge, which reduces noise in real-world measurements.

#### IV-C2 Demonstration dataset

A human demonstrator kinesthetically performs the desired wiping motion by moving the robot’s end-effector in free drive mode. The demonstrator is instructed to wipe the inclined table (which differs from the slope used in the validation experiments), applying as much force as possible to maximize cleaning efficiency [^16]. We collected 8 demonstrations using a normal sponge with natural speed, with no errors made by the demonstrator in completing the task. We record the robot’s end-effector’s position, force and torque in the (x, y, z) axis at a rate of $2.5Hz$ for $10s$ to obtain the motion trajectory $x^{\text{demo}}\in\mathbb{R}^{25\times 2}$ (2 absolute positions in (x, y) axis), vertical motion trajectory $\Delta h^{\text{demo}}\in\mathbb{R}^{25}$ (vertical displacements from the previous time step), and FT trajectory $\tau^{\text{demo}}\in\mathbb{R}^{6\times 25}$.

### IV-D Model training

The datasets are pre-processed before training; we apply a Butterworth low-pass filter offline to unlabeled data and online to FT demonstrations data. Subsequently, we normalize all data to \[0.0, 0.9\].

*Pre-training:* We pre-train the sponge properties encoder $\phi_{\text{sponge}}$ as described in III-A using 1000 unlabeled simulation data IV-C1. We adopt the Adam optimizer and train the model for 200 epochs at a learning rate of 0.0001.

*Training:* We train the motion trajectory decoder $\theta_{\text{traj}}$ as described in III-B1 using 8 motion trajectory data $x^{\text{demo}}$ of demonstrations. We adopt the Adam optimizer as an optimizer and train the decoder for 10000 epochs at a learning rate of 0.001. We train the FT feedback loop described in III-B2 using 8 vertical motion trajectory data $\Delta h^{\text{demo}}$ and FT data $\tau^{\text{demo}}$ of demonstrations. We treat FT data as time series data and set the window size as 5. We adopt the Adam optimizer as an optimizer and train the loop for 2000 epochs at a learning rate of 0.001.

## V Results and Discussion

To evaluate our method, we conducted experiments using a robot under varying conditions in a total of 40 scenarios, including different heights of the wiping table (V-A) and different types of sponges (V-B). We compared our method with two state-of-the-art control methods: (1) Aoyama et al. [^16] (baseline), which is an imitation learning-based control method without an FT feedback loop, and (2) an admittance control (AC), which is a non-learning-based method. As noted in Section II, in the same problem setting as the baseline and proposed method, AC cannot be executed due to the absence of necessary target force information. Therefore, during the execution of AC, we defined the target force as the force applied when the sponge is pressed by $1cm$, enabling the implementation of AC. AC attempts to maintain this target force predicting vertical displacement $\Delta{h}$ using Eq. (4) [^25].

$$
\Delta{h}=\frac{FT^{2}+BT\,\Delta{h}_{t-1}+M\left(2\,\Delta{h}_{t-1}-\Delta{h}%
_{t-2}\right)}{M+BT+KT^{2}}
$$

In this equation, $M=0.5[Kg]$, $B=5[N/(m/s)]$, and $K=15[N/m]$ represent the desired inertia, damping and stiffness values, respectively. The variable $t$ denotes the $t$ th sampling period, with $T=0.4[s]$ as the sampling period. Additionally, we tested our model on a completely different setup – wiping a vertical wall instead of a horizontal table – using the same model trained with table wiping demonstration data. For each verification, we compared the contact with the table by examining the ratio of time steps in which the sponge contacted a table. And we examined the force applied to the sponge to compare whether the robot ’wiped’ with the sponge. Specifically, we used the average vertical force applied to the sponge and its standard deviation, referencing data from human demonstrations (Table 6).

Figure 6: Reference force information from Demonstrations

<table><thead><tr><th></th><th colspan="3">Demonstration</th></tr></thead><tbody><tr><td></td><td>Contact</td><td>Average [N]</td><td>Std</td></tr><tr><td>Normal</td><td>100%</td><td>-12.6</td><td>5.39</td></tr><tr><td>s1f1</td><td>100%</td><td>-22.8</td><td>8.11</td></tr><tr><td>s1f2</td><td>100%</td><td>-21.4</td><td>7.76</td></tr><tr><td>s1f3</td><td>100%</td><td>-21.3</td><td>10.1</td></tr><tr><td>s2f1</td><td>100%</td><td>-24.1</td><td>12.2</td></tr><tr><td>s2f2</td><td>100%</td><td>-30.0</td><td>15.8</td></tr><tr><td>s2f3</td><td>100%</td><td>-34.3</td><td>15.9</td></tr><tr><td>s3f1</td><td>100%</td><td>-30.9</td><td>9.27</td></tr><tr><td>s3f2</td><td>100%</td><td>-35.2</td><td>12.4</td></tr><tr><td>s3f3</td><td>100%</td><td>-36.7</td><td>10.5</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/force_boxplot.png)

Refer to caption

### V-A Verification of the ability to adapt to changes in height

We varied the wiping table’s heights (low, high, sloped) from the height used in the demonstrations (inclined table). The results are shown in Table 5 and Fig. 7. Force data during demonstration (Table 6) is used as the reference force.

To adapt wiping motions to changes in the wiping surface height, the robot should apply a consistent force to the sponge regardless of the height. With the same sponge, the robot should wipe with as much force as possible to ensure effective wiping. With the baseline method, the sponge was in contact with the table only 0-44% of the time, and the average force reached merely 4% of the desired reference force (Fig. 7). Specifically, in some cases with the low and sloped tables, the average force turned positive because the sponge did not contact the table, and the influence of gravitational force from the sponge’s own weight became dominant. This indicates that a robot did not effectively ’wipe’ and was unable to adapt to changes in the wiping surface height. In contrast, both AC and our proposed method maintained constant contact in all 30 cases. However, AC applied only an average of 42% of the reference force, whereas our proposed method successfully maintained an appropriate average force on the sponge across all heights, averaging 96% of the reference force (Fig. 7). Furthermore, the applied force did not significantly vary with changes in table height as shown in Fig. 8 (a), with the standard deviation being only about 5% larger than that of human demonstrations. This indicates the robot’s ability to successfully adapt to height variations.

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/height_plot.png)

Refer to caption

### V-B Verification of the ability to adapt to changes in sponge

We varied the sponge properties (3 stiffness levels $\times$ 3 friction levels) from the sponge used in the demonstrations (normal). The results are shown in Table 5 and Fig. 7. Adapting the wiping motions to changes in the sponges’ physical properties requires adjusting the force applied to the sponge accordingly. With the baseline method, the robot failed to maintain contact with the table when using sponges with unseen properties. Specifically, with the low table, the contact ratio was 0% for all 9 unseen sponges. Moreover, the average force applied was less than 25% of the expected force, averaging only 4% of the reference force (Fig. 8 (b)). Therefore, the baseline is unable to adapt to unseen sponges.

In contrast, both AC and our proposed method successfully maintained contact at all time steps in all 30 cases. However, AC merely maintained the predefined target force without considering the sponge’s physical properties, resulting in only 23-63% of the expected force and an average of 42% of the reference force being exerted. Our proposed method, on the other hand, applied an average force comparable to the expected force, achieving over 63% and an average of 96% of the reference force, according to the type of sponge (Fig. 8 (b)). This demonstrates that our method successfully enables the robot to adapt to unseen sponge properties.

Figure 9: Experimental results: Wall wiping

<table><thead><tr><th></th><th colspan="3">Wall Wiping</th></tr></thead><tbody><tr><td></td><td>Contact</td><td>Average [N]</td><td>Std</td></tr><tr><td>Normal</td><td>100%</td><td>-14.5 (115%)</td><td>2.92</td></tr><tr><td>s1f1</td><td>100%</td><td>-23.7 (104%)</td><td>17.1</td></tr><tr><td>s1f2</td><td>100%</td><td>-29.5 (138%)</td><td>21.6</td></tr><tr><td>s1f3</td><td>100%</td><td>-25.4 (119%)</td><td>19.5</td></tr><tr><td>s2f1</td><td>100%</td><td>-29.0 (120%)</td><td>19.4</td></tr><tr><td>s2f2</td><td>100%</td><td>-32.2 (107%)</td><td>19.7</td></tr><tr><td>s2f3</td><td>100%</td><td>-28.9 (84%)</td><td>18.4</td></tr><tr><td>s3f1</td><td>100%</td><td>-27.0 (87%)</td><td>13.8</td></tr><tr><td>s3f2</td><td>100%</td><td>-33.5 (95%)</td><td>17.1</td></tr><tr><td>s3f3</td><td>100%</td><td>-27.3 (74%)</td><td>19.2</td></tr></tbody></table>

### V-C Wall Wiping

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/wall_wiping.png)

Refer to caption

In real-world scenarios, cleaning involves more than just wiping horizontal surfaces like tables; it may include tasks such as wiping walls and other vertical surfaces. A key challenge for robots in these tasks is the ability to adapt to the physical properties of sponges and adjust the applied force in real time as surface conditions change. Our method achieves this adaptiveness independently of gravitational effects. In previous tasks (V-A and V-B), the direction of the forces applied to the sponge was aligned with gravitational acceleration, whether the configuration was low, high, or sloped. To further demonstrate that our method is effective regardless of gravity’s influence, we tested our method in a gravity-neutral setting—wall wiping—where gravitational forces do not affect the applied forces during the task.

We evaluated the same model as V-A and V-B, trained using the same demonstration data of table wiping. Due to the setting changes, the end-effector’s frame rotated 90 degrees and the base-link’s x-axis came vertically to the end-effector. We swapped the position outputs of the x-axis and z-axis based on the base-link, and introduced an offset to the z-axis positions. Although this might appear as a mere transformation of output trajectories, the core challenge lies in the method’s ability to adjust applied forces in a gravity-independent manner. The results are shown in Table 9.

Our method maintained contact with a wall in all 10 cases and the applied forces were comparable to that expected, averaging 104% of the reference force. This indicates that a robot can adapt to wall wiping even with unseen sponges.

## VI Conclusion

This work tackles the challenges of robots adapting to environmental changes in manipulating deformable objects in contact-rich tasks with few human demonstrations. Our method combines real-time FT feedback with pre-trained object representations in closed-loop by treating contact information as time series data. Focusing on a wiping task, we varied table heights and sponge properties. To verify the effectiveness of the proposed method, we also tested the proposed method on a wall-wiping task. Experimental results show that the robot adapts to unseen manipulating surface height and object properties with our method, surpassing performances of the baseline and AC methods.

Although we demonstrated our approach’s adaptiveness, we found that the standard deviations were $5\%$ greater on average than that of the human demonstrations, and about 3 times greater compared to AC. This increased variability suggests that further refinement of the control policy is needed to achieve more consistent results, which is crucial for tasks requiring high precision and consistency but probably enough for daily life tasks. Our method has another limitation: it is designed specifically for tools that are deformable and elastic. The approach follows this premise in [^16], where applying as much force as possible maximizes cleaning efficiency. However, when the robot attempts to wipe with rigid objects (e.g., bricks), our method applies excessive force according to the object’s hardness, causing the robot to trigger safety alarms and stop. In contrast, admittance and impedance control methods can handle such cases by adjusting target force and position.

While our method is more advantageous than admittance and impedance control for deformable and elastic objects, future work will focus on expanding its applicability to a wider range of objects, including non-deformable ones. One idea is to enhance our system by pre-training it on a large dataset, similar to models like PaLM-E [^26], so the robot can adjust its actions based on visual or linguistic prompts. This would allow to create personalized motion for different object types. Our method would be more versatile and adaptive across various real-world scenarios.

TABLE I: The average ratio of the applied force in the z-direction to the reference force exerted by the demonstrator.

<table><tbody><tr><td></td><td colspan="2">Layer</td><td colspan="2">Window Size</td><td colspan="2">Demo</td><td rowspan="2">Standard</td></tr><tr><td></td><td>Fewer</td><td>More</td><td>Smaller</td><td>Larger</td><td>Fewer</td><td>More</td></tr><tr><td>Average (%)</td><td>159</td><td>152</td><td>182</td><td>170</td><td>190</td><td>114</td><td>97</td></tr></tbody></table>

We conducted ablation studies to validate: (1) the number of layers in the FT feedback loop, (2) the window size of TCN, and (3) the number of demonstrations. Additionally, pre-training ablation studies have been conducted in [^16], demonstrating its impact on improving the generation of desired wiping motion for unseen sponges. In each ablation study, we used the values from our proposed method as the standard and compared them with two variants: smaller values and larger values. We evaluated the model’s performance by comparing the force exerted in z-direction by the robot with the reference force exerted by a human during the demonstration. We tested the system under 2 wiping surface heights (low and high) and 2 sponge types—a known sponge (Normal) used in the demonstration and an unknown sponge (s2f1) not used during the demonstration—resulting in $2\times 2=4$ combinations. The results are shown in Table I.

*Number of layers in the FT feedback loop:* We compared the proposed 2-layer model with a fewer-layer model (1 layer) and a more-layer model (5 layers) to analyze the effects of model depth on performance (Fig. 11 (a)). In the fewer-layer model, although the force applied to the sponge was adjusted, the reference force was 1.43 to 1.73 times higher, showing that the model lacked sufficient capacity to capture the necessary force control dynamics. On the other hand, the more-layer model appeared to handle the unknown sponge well at first glance but applied a force around -25N regardless of the sponge’s physical properties. This suggests that the deeper model was too complex and failed to learn force control dynamics and generalize well. These results indicate that neither too shallow nor too deep a model performs well, with the optimal performance achieved at a depth of around 2 layers, where the balance between model capacity and complexity is effectively maintained.

![Refer to caption](https://arxiv.org/html/2505.06451v1/extracted/6426510/imgs/ablation_layer.png)

Refer to caption

*Window size of TCN:* We compared the proposed window size of 5 with a smaller window size (window size of 1) and a larger window size (window size of 10) to analyze the effects of the length of the past history referenced on performance (Fig. 11 (b)). The smaller window size model applied approximately $-33N$ for low surfaces and $-27N$ for high surfaces, regardless of the sponge type. The difference in force applied with the change in surface height exceeded 5N, indicating that the model could not handle either the wiping surface height or sponge properties. These suggest that the window size should neither be too short nor too long. For this wiping task, a window size of 5 is appropriate.

*Number of demonstrations:* We compared the proposed model with 8 demonstrations against a fewer-demo model (4 demonstrations) and a more-demo model (12 demonstrations) to analyze the effect of demonstration quantity on performance (Fig. 11 (c)). The fewer-demo model exerted excessive force of $-30N$ in all conditions, suggesting that it learned to simply push hard regardless of the conditions. The more-demo model showed only small changes in applied force when the surface height changed and applied forces close to the reference force when the sponge type changed. The performance was nearly identical to the standard model. As shown in Table I, the average ratio of applied force to reference force for the more-demo model (114%) was similar to the standard model (97%). Therefore, a minimum of 8 demonstrations is sufficient for the model to learn the relationship between the FT history and the end-effector’s next position.

[^1]: A. Hussein and et al., “Imitation learning: A survey of learning methods,” *ACM Computing Surveys (CSUR)*, vol. 50, no. 2, pp. 1–35, 2017.

[^2]: Y. Duan and et al., “One-shot imitation learning,” in *Advances in Neural Information Processing Systems*, I. Guyon, U. von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds. Curran Associates, Inc.

[^3]: A. Billard and D. Kragic, “Trends and challenges in robot manipulation,” *Science*, vol. 364, no. 6446, p. eaat8414, 2019.

[^4]: R. Martín-Martín and et al., “Variable impedance control in end-effector space: An action space for reinforcement learning in contact-rich tasks,” in *IEEE/RSJ Int. Conf. on intelligent robots and systems*, 2019, pp. 1010–1017.

[^5]: O. Spector and M. Zacksenhouse, “Learning contact-rich assembly skills using residual admittance policy,” in *IEEE/RSJ Int. Conf. on Intelligent Robots and Systems*, 2021, pp. 6023–6030.

[^6]: L. Rozo, D. Bruno, S. Calinon, and D. G. Caldwell, “Learning optimal controllers in human-robot cooperative transportation tasks with position and force constraints,” in *IEEE/RSJ Int. Conf. on Intelligent Robots and Systems*, 2015, pp. 1024–1030.

[^7]: K. Yamane, Y. Saigusa, S. Sakaino, and T. Tsuji, “Soft and rigid object grasping with cross-structure hand using bilateral control-based imitation learning,” *IEEE Robotics and Automation Letters*, 2023.

[^8]: P. Florence and et al., “Self-supervised correspondence in visuomotor policy learning,” *IEEE Robotics and Automation Letters*, vol. 5, no. 2, pp. 492–499, 2019.

[^9]: J. Pari and et al., “The surprising effectiveness of representation learning for visual imitation,” *CoRR*, vol. abs/2112.01511, 2021.

[^10]: I. Guzey and et al., “Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play,” *arXiv preprint arXiv:2303.12076*, 2023.

[^11]: O. Kroemer and et al., “Learning dynamic tactile sensing with robust vision-based training,” *IEEE Trans. on robotics*, vol. 27, no. 3, pp. 545–557, 2011.

[^12]: L. Weng, “Domain randomization for sim2real transfer,” *lilianweng.github.io*, 2019.

[^13]: J. Tobin and et al., “Domain randomization for transferring deep neural networks from simulation to the real world,” in *2017 IEEE/RSJ Int. Conf. on intelligent robots and systems*, 2017, pp. 23–30.

[^14]: X. B. Peng and et al., “Sim-to-real transfer of robotic control with dynamics randomization,” in *2018 IEEE Int. Conf. on robotics and automation*, 2018, pp. 3803–3810.

[^15]: Y. Wu, W. Yan, T. Kurutach, L. Pinto, and P. Abbeel, “Learning to manipulate deformable objects without demonstrations,” in *Robotics: Science and Systems*, 2020. \[Online\]. Available: [https://doi.org/10.15607/RSS.2020.XVI.065](https://doi.org/10.15607/RSS.2020.XVI.065)

[^16]: M. Y. Aoyama and et al., “Few-shot learning of force-based motions from demonstration through pre-training of haptic representation,” in *Proc. of the IEEE Int. Conf. on Robotics and Automation*, 2023.

[^17]: N. Hogan, “Impedance control: An approach to manipulation: Part ii—implementation,” 1985.

[^18]: H. Seraji, “Adaptive admittance control: An approach to explicit force control in compliant motion,” in *Proceedings of the 1994 IEEE Int. Conf. on Robotics and Automation*, 1994, pp. 2705–2712.

[^19]: D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” *arXiv preprint arXiv:1312.6114*, 2013.

[^20]: S. Bai and et al., “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” *arXiv:1803.01271*, 2018.

[^21]: J. Lee and et al., “Learning quadrupedal locomotion over challenging terrain,” *Science Robotics*, vol. 5, no. 47, p. eabc5986, 2020.

[^22]: J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical evaluation of gated recurrent neural networks on sequence modeling,” in *Proc. NeurIPS Workshop Deep Learn.*, 2014.

[^23]: Y. Zhu and et al., “robosuite: A modular simulation framework and benchmark for robot learning,” in *arXiv preprint arXiv:2009.12293*, 2020.

[^24]: M. Quigley and et al., “Ros: an open-source robot operating system,” in *ICRA workshop on open source software*, vol. 3, no. 3.2. Kobe, Japan, 2009, p. 5.

[^25]: P. Song, Y. Yu, and X. Zhang, “A tutorial survey and comparison of impedance control on robotic manipulation,” *Robotica*, vol. 37, no. 5, p. 801–836, 2019.

[^26]: D. Driess and et al., “Palm-e: An embodied multimodal language model,” in *arXiv preprint arXiv:2303.03378*, 2023.