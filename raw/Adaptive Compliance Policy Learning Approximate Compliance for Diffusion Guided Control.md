references.bib

Yifan Hou <sup>1</sup>  Zeyi Liu <sup>1</sup>  Cheng Chi <sup>1</sup>  Eric Cousineau <sup>2</sup>  Naveen Kuppuswamy <sup>2</sup>    
Siyuan Feng <sup>2</sup>  Benjamin Burchfiel <sup>2</sup>  Shuran Song <sup>1</sup>  
<sup>1</sup> Stanford University <sup>2</sup> Toyota Research Institute  
[adaptive-compliance.github.io](https://arxiv.org/html/2410.09309v2/adaptive-compliance.github.io)

###### Abstract

Compliance plays a crucial role in manipulation, as it balances between the concurrent control of position and force under uncertainties. Yet compliance is often overlooked by today’s visuomotor policies that solely focus on position control. This paper introduces Adaptive Compliance Policy (ACP), a novel framework that learns to dynamically adjust system compliance both spatially and temporally for given manipulation tasks from human demonstrations, improving upon previous approaches that rely on pre-selected compliance parameters or assume uniform constant stiffness. However, computing full compliance parameters from human demonstrations is an ill-defined problem. Instead, we estimate an approximate compliance profile with two useful properties: avoiding large contact forces and encouraging accurate tracking. Our approach enables robots to handle complex contact-rich manipulation tasks and achieves over 50% performance improvement compared to state-of-the-art visuomotor policy methods. Project website with result videos: [adaptive-compliance.github.io](https://arxiv.org/html/2410.09309v2/adaptive-compliance.github.io).

![[Uncaptioned image]](https://arxiv.org/html/2410.09309v2/x1.png)

Figure 1: Compliance Requirements. \[Left\] Flipping an item requires the robot to follow an arc trajectory (blue) while maintaining contact force. This demands low stiffness in pushing directions ($K_{2}$) and high stiffness elsewhere ($K_{1}$). \[Right\] Wiping a vase necessitates 3D compliance adjustments in both end-effectors to 1) hold the vase, 2) trace the marking, and 3) apply appropriate force without damage. Our algorithm aims to model these spatial-, temporal-, and task-dependent compliance requirements from human demonstration data.

![Refer to caption](https://arxiv.org/html/2410.09309v2/x2.png)

Figure 2: Method Comparisons. \[LEFT\] shows the comparison between a) a typical visuomotor policy \[ chi2023diffusionpolicy \], b) a typical force-based compliant policy lee2019making, and c) Adaptive Compliance Policy. \[Right\] Visualization of virtual target (orange sqaures) and reference poses (yellow circles) inferred by Adaptive Compliance Policy. The directional difference (orange arrows) between the virtual and reference poses encodes compliance direction.

## I INTRODUCTION

Manipulation often requires the concurrent control of both position and force to achieve the desired outcome. This joint objective can be captured by the concept of mechanical compliance \[mason1981compliance, raibert1981hybrid, hogan1984impedance\], where a low compliance prioritizes position accuracy regardless of external forces, while a high compliance allows large position deviation in response to external forces, making the system “soft” during interaction.

The desired compliance for a robotics system is not a static property; rather, it varies drastically depending on the task objectives and the system’s state. For instance, consider the flipping task in Fig. 1, the desired compliance:

- Varies temporally. For example, The system needs to be less compliant before contact to prioritize precise position tracking while becoming compliant upon contact.
- Varies spatially. For example, during the pivoting stage, the system should be compliant only in the pushing directions (i.e., $K_{2}$ direction) while maintaining high stiffness in other directions to follow the arc motion (e.g., low compliance in $K_{1}$ direction).
- Varies from task to task. If we change to a different task, such as wiping a vase in Fig. 1 \[Right\], both temporal and spatial properties of the compliance will change in order to satisfy the unique 3D motion and force requirements.

While the desired compliance can be obtained from optimization given physical measurements of the manipulation problem \[hou2019robust, hou2020manipulation\], it remains a challenge to obtain compliance parameters directly from human demonstrations. Compliance describes how force and motion variations in all directions are related, a typical demonstration trajectory does not contain all the information. Prior work often requires detailed known dynamics parameter \[grioli2011non\] or multiple demonstration trajectories on the exact same task to statistically estimate one robot’s compliance \[calinon2010learning, kronander2012online, duan2018learning, duan2019sequential\]. These approaches cannot handle new scene configurations or unexpected perturbations. As a result, compliant policies either rely on pre-selected compliance parameters for the target tasks \[kamijo2024learning\] or assume uniform constant stiffness across all directions \[lee2019making\].

In this work, we introduce Adaptive Compliance Policy (ACP), a sensorimotor policy that learns to dynamically adjust the system compliance both spatially and temporally for a given manipulation task from human demonstrations.

Specifically, the algorithm represents the compliance profile by an additional stiffness value and a virtual target pose in addition to the original reference pose (e.g., robot end-effector pose) predicted by the policy. The directional difference between the virtual and reference targets encodes the spatial distribution of stiffness. Finally, the predicted reference pose and stiffness can be executed by a standard low-level high-rate compliance controller to achieve robust and adaptive compliance behaviors.

Instead of estimating the exact human compliance, we derive a simple rule to obtain a useful compliance profile that guarantees the avoidance of large internal forces while encourages precise tracking, under mild assumptions about the tasks. This simple rule allows us to approximate varying stiffness for every demonstration episode with different object variations and scene configurations. Then, by learning from a collection of demonstrations, the policy can summarize the typical compliance profile for a specific task and quickly adjust compliance based on visual and force feedback.

We systematically evaluate the performance of our algorithm on two real world contact-rich manipulation tasks: object flipping and vase wiping. Our method achieves over 50% increase in performance compared to state-of-the-art visuomotor policy methods. In summary, the main contribution of the paper includes:

- Adaptive Compliance Policy formulation that is able to dynamically adjust compliance to maintain desired contact modes despite uncertainties and disturbances.
- A kinesthetic teaching system that allows demonstrations with varying compliance profiles.
- A method to compute spatial-, temporal-varying compliance labels from human demonstrations, making ACP training practical and scalable.

## II RELATED WORK

Contact-rich manipulation with active compliance. There is a long history of work on utilizing robot compliance for robust contact-rich manipulation \[lozano1984automatic, uchiyama1988symmetric, sawasaki1991tumbling, hou2018fast\]. Given modeling information such as contact geometry and friction, the stiffness that provides the maximum robustness can be computed efficiently \[hou2019robust, hou2021efficient, hou2020manipulation\]. Although proved in many occasions, these methods typically require careful modeling work to setup. Our work utilizes similar mechanical modelings but derives a model-free method that is easier to scale.

Learning compliance from Reinforcement Learning (RL). RL can learn compliance controllers by exploring force-motion variations \[portela2024learning, kalakrishnan2011learning, beltran2020learning, beltran2020variable, noseworthy2024forge, chang2022impedance, martin2019variable\]. However, these policies need to be retrained for any scene variations. Moreover, most of existing work uses fixed-parameter low-level compliance controllers like impedance \[lee2019making\] or admittance \[kohler2024symmetric\] controllers, which lack robustness against disturbances.

Learning compliance from human Human stiffness during manipulation can be estimated from sufficiently repeated same motions \[deng2016learning, duan2018learning, duan2019sequential, wang2020framework, yamane2023soft, zeng2021generalization\]. The stiffness is either proportional to the force covariance \[duan2018learning, duan2019sequential\] or inversely proportional to the position covariance \[calinon2010learning, kronander2012online\]. These methods do not work for visuomotor policy learning since every demonstration is different. \[grioli2011non\] proposed a stiffness estimator that works with a single demonstration. However, it requires perfect knowledge of human mass and damping, which is impractical. Prior work also attempted to give the demonstrator the ability to specify uniform stiffness \[kronander2013learning, kamijo2024learning\]. A concurrent work \[liu2024forcemimic\] computes compliance profiles based on heuristics from manually divided stages of tasks.

Learning visuomotor policies with force feedback. To effectively encode the temporal relations in the force/torque data, Prior work has explored many different encoding methods such as causal convolution \[lee2019making\], TCN \[beltran2020variable\], LSTM \[ding2019transferable\], VAE \[aoyama2024few\], and self-attention \[kim2023training, kohler2024symmetric\]. However, despite taking force as input, most of the existing visuomotor policies are still solely focused on predicting the position of the robot \[liu2024maniwav, li2022see\] with uniform constant compliance \[lee2019making\] that fails to capture the spatial- and temporal- variations of the compliance requirements for delicate manipulation task.

## III Method

In this section, we introduce our pipeline, from collecting human demonstrations to designing the Adaptive Compliance Policy. We first introduce related physical concepts in §III-A, then describe a kinesthetic teaching system with compliance control §III-B. Next, §III-C introduces how we annotate compliance labels for the human demonstration data. Finally, §III-D explain the Adaptive Compliance Policy design.

### III-A Preliminaries: Modeling

Robot Compliance Modeling. Compliance expands the action space of a robot \[khatib1987unified, mason1981compliance\]. Consider a $N$ dimensional system described by position $x\in\mathbb{R}^{N}$ and force $f\in\mathbb{R}^{N}$. Compliance is the elastic behavior between force and motion, which is typically modeled by a spring-mass-damper system:

$$
\vspace{-2mm}f=M\ddot{x}+K(x-x_{ref})+K_{D}\dot{x}.
$$

The three terms on the right-hand side represent inertia force, spring force, and damping force, respectively. $x_{ref}$ is a reference position, at which the spring force is zero. The compliant behavior is described by the inertia matrix $M\in R^{N\times N}$, stiffness matrix $K\in R^{N\times N}$ and damping matrix $K_{D}\in R^{N\times N}$, which can be user-specified if the compliance 1 is implemented by control. In other words, they can be added to the action space of a high level policy \[beltran2020learning, beltran2020variable, deng2016learning\], while a low level compliance controller implements the ”virtual” stiffness/damping/inertia. This is beneficial when the high level policy cannot be fast enough to exhibit compliance.

We use admittance control \[maples1986experiments\] for our compliance controller, which takes in force feedback and outputs position targets. This works for our high accuracy position-controlled robot (UR5e). Robots with force control interface may also use impedance control \[hogan1984impedance\], or some hybrid force-motion control schemes \[mason1981compliance, raibert1981hybrid\].

Manipulation Modeling. Next we model how a manipulation system interacts with a robot, treating the robot as a black box. We make the following assumption:

> \-2em-2em Assumption I: Contact force dominates. Other types of force, such as inertia force, friction and gravity, are negligible comparing with contact force.

This is not to be confused with the compliance control of the robot itself, which is fast and dynamic. We ensure Assumption I by avoiding fast robot motion and using lightweight objects. Consider a robot with $N$ degrees of freedom, making $n$ contacts with the environment. Denote $\lambda\in\mathbb{R}^{n}$ as the vector of contact normal forces, the Newton’s Second Law can be written as following with Assumption I:

$$
\vspace{-2mm}J^{T}\lambda=f,\vspace{-1mm}
$$

where $J$ is the Contact Jacobian matrix that maps contact force into the robot generalized force space. While used in derivation, our method does not need to compute it. Denote $v\in\mathbb{R}^{N}$ as the generalized velocity vector, the Jacobian $J$ can also describe the velocity constraint imposed by contacts:

$$
\vspace{-2mm}Jv\geq 0.\vspace{-2mm}
$$

![Refer to caption](https://arxiv.org/html/2410.09309v2/x3.png)

Figure 3: Data Collection with Haptic Feedback. We designed a kinesthetic teaching system with low-stiffness compliance that allows the operator to demonstrate variable compliance behavior with direct haptic feedback.

### III-B Demonstration Collection

We choose kinesthetic teaching instead of teleoperation to collect human demonstrations in order for the operator to easily demonstrate variable compliance behavior under direct haptic feedback (see Fig. 3). The setup for one arm includes one robot manipulator to provide accurate position feedback, one RGB camera to record visual information, and one force torque sensor mounted near the robot hand.

During demonstration, we specify low stiffness, low damping and low mass for the robot compliance controller so the operator can move the robot freely. Low damping and mass are achievable during demonstration because the human hand provides a natural external stabilization. During testing, we increase robot damping and virtual mass to maintain stability of the admittance controller. We also found it necessary to set the tool center point (TCP) near the handle, so the robot can rotate under external force in a way intuitive to the operator.

### III-C Estimating Compliance from Demonstrations

Human uses varying stiffness during manipulation. High stiffness provides position accuracy under force disturbances. Low stiffness is also necessary, since the high stiffness controls impose velocity constraints that might conflict with the contact constraints defined in Eq. 3, during which a huge internal force may be generated \[hou2021efficient\].

However, it is often an ill-conditioned problem to estimate compliance parameters from a single human demonstration due to the lack of variations \[calinon2010learning\]. Previous work on stiffness estimation \[grioli2011non\] assumes perfect knowledge of damping and virtual mass and sufficient variations in force-motion signals. These requirements are not met in kinesthetic teaching, where the human hand changed the effective damping and mass of the robot hand, and the demonstration could have constant force or position for a period of time.

To simplify the problem, we also use pre-selected values for mass and damping. Then instead of estimating the true human stiffness, we propose to find a stiffness matrix with the following properties:

- It avoids huge internal forces in manipulation.
- It encourages accurate tracking of the desired motion.

C.1 Stiffness direction: We propose the following simple strategy to choose stiffness direction in the generalized space: Use a low stiffness $k_{low}$ in the direction of the force feedback, and a high stiffness $k_{high}$ in all other directions.

Next we explain why this simple rule works. From rigid body mechanics \[murray2017mathematical\], we know that the rows of Contact Jacobian $J$ represents the directions of contact normal forces, which forms a polyhedral convex cone in the generalized force space. We make two more assumptions:

> \-2em-2em Assumption II: Nonzero contact force: all made contacts should have nonzero contact forces.  
> Assumption III: No pinching contacts: the cone formed by rows of the Contact Jacobian $J$ is contained in its dual cone.

Assumption II can be satisfied by making contacts clearly in demonstrations. Assumption III means the contacts on the robot are not too restrictive. Fig. 4 shows some examples:

![Refer to caption](https://arxiv.org/html/2410.09309v2/x4.png)

Figure 4: Pinching Examples. Grey shape represents a robot tool, blue shape represents a frictionless environment. First three examples are not pinching contact, the last one is.

We propose the following theorem under the assumptions:

###### Theorem 1.

For a robot under external contact described by Eq. 2, there exists a solution $v$ that satisfies the contact constraint 3 as long as it does not control its velocity in the direction of feedback force $f$ in the generalized space.

Proof. Not controlling velocity in the force direction means the velocity has a free component:

$$
v=v_{0}+kf=v_{0}+kJ^{T}\lambda,
$$

where $k$ is an arbitrary scaling factor, $v_{0}$ denotes the components of generalized velocity in other directions. Due to Assumption II, the contact force $\lambda$ must have all positive components, $J^{T}\lambda$ represents a ray strictly inside the cone formed by rows of $J$. Assumption III says this cone is contained in its dual cone $\{x\in\mathbb{R}^{N}|Jx\geq 0\}$, so

$$
JJ^{T}\lambda>0.
$$

Then $JV=JV_{0}+kJJ^{T}\lambda>0$ for large enough $k$.

Theorem 1 shows that one-dimensional low stiffness control suffices to avoid constraint violation, so we can use high stiffness in other directions to improve position tracking. Let $K_{0}\in\mathbb{R}^{N}$ be a diagonal matrix with $[k_{low},k_{high},...,k_{high}]$ on its diagonal, and $S\in\mathbb{R}^{N}$ be a matrix whose columns form an orthonormal basis of $\mathbb{R}^{N}$ with its first column as $f/|f|$. The stiffness matrix can be written as:

$$
K=SK_{0}S^{-1}
$$

We use $k_{high}$ in all directions when $|f|$ is small.

C.2 Stiffness Magnitude The high stiffness $k_{high}$ should support accurate position tracking in those directions, which can be set empirically. The low stiffness value should be zero, however, since the low stiffness direction is estimated from noisy force signal, we found it helpful to let the stiffness decrease continuously with the force magnitude:

$$
k_{low}=\left\{\begin{matrix}k_{max},&|f|<f_{min}\\
k_{max}-(k_{max}-k_{min})\frac{|f|-f_{min}}{f_{max}-f_{min}},&f_{min}\leq|f|%
\leq f_{max}\\
k_{min},&|f|>f_{max}\\
\end{matrix}\right.
$$

where $k_{max},k_{min},f_{max}$ and $f_{min}$ are parameters determined by the hardware system.

### III-D Adaptive Compliance Policy

We formulate the policy as a diffusion process for both reference action and target stiffness. The policy runs in a receding-horizon manner \[chi2023diffusionpolicy\], where an action trajectory is predicted using recent observations: 1) fisheye RGB images, 2) robot end-effector poses, and 3) force/torque data.

#### III-D1 Inputs and Encoding

We implement two encoding strategies for the force/torque data: 1) temporal encoding via causal convolution \[oord2016wavenet\], which helps capture causal relations from sequential data like force. We use force readings from the past 32 timesteps and pass them through a 5-layer causal convolution network, outputting a 768-dimensional vector; 2) FFT encoding, where we convert each dimension of the 6D force/torque readings into a 2D spectrogram. These spectrograms (6 $\times$ 30 $\times$ 17) are passed to a ResNet-18 model with a modified input channel of 6. A coordinate convolution layer \[liu2018intriguing\] is added at the beginning to handle translational invariances. The images from the past two timesteps are resized to 224 $\times$ 224 with random cropping then encoded using a CLIP-pretrained ViT-B/16 model \[dosovitskiy2020image\].

Both image and force encodings are passed to a transformer encoder layer, using self-attention to learn adaptive visual-force representations. This representation is concatenated with robot end-effector poses from the past 3 timesteps and fed to the downstream diffusion policy head as a condition following \[chi2023diffusionpolicy\].

#### III-D2 Outputs and decoding

Our policy output encodes the position target, the stiffness matrix, and the reference force in a 19-dimensional vector per robot arm:

- Reference pose: 9D pose vector following convention in \[chi2023diffusionpolicy\], where the last six elements are the top two rows of a rotation matrix;
- Virtual target pose: 9D pose representing the actual target for the low level compliance controller to track;
- A scalar value representing the stiffness magnitude in the low stiffness direction.

The virtual target pose is computed such that the robot will exert the reference force if it reaches the reference pose while tracking the virtual target with the given stiffness. It essentially changes a force target into a position target. The benefit is to have a uniform target representation across different robots: an impedance controlled robot without FT sensor can also execute the virtual target.

During training, we first pass the whole episode of wrench data through a moving average filter with one second window size, then compute the stiffness from it using Eq. 6, and finally compute the virtual target following a 3D mechanical spring. The heavy wrench filtering has two benefits: 1) it makes the virtual target smooth; 2) it gives the action labels hindsight information about contacts about to be made, which is crucial for smooth contact engaging motions.

At inference time, the full stiffness matrix is reconstructed following Eq. 6 by replacing the force direction with the direction from the reference pose to the virtual target. Finally both the stiffness matrix and the virtual target are sent to the low level compliance controller for execution.

## IV Experiments

We evaluate our method in two contact-rich manipulation tasks whose success depends on the maintenance of suitable contact modes. We use the UR5e robot with one GoPro RGB camera and one ATI mini-45 force torque sensor. The GoPro camera streams images at 60Hz, the robot receives Cartesian pose commands and send pose feedback at 500Hz, while the ATI sensor streams force-torque data at up to 7000Hz.

We evaluate the following four policies, all trained on the same dataset with the same number of epochs:

- ACP: Adaptive Compliance Policy, our approach;
- ACP w.o. FFT: same as ACP but with force encoded using temporal convolution \[oord2016wavenet, lee2019making\] instead of FFT.
- Stiff policy: Diffusion policy \[chi2023diffusionpolicy, chi2024diffusionpolicy\] with additional force input. Outputs target positions.
- Compliant policy: Same as the \[Stiff policy\] except that the low level controller has a uniform stiffness $k=500$ N/m. Relying on low level robot compliance is common in visuomotor policies \[lee2019making, wang2023mimicplay, zhu2023viola, kohler2024symmetric\].

Both ACP variations uses compliance in 3D translational space, i.e. $N=3$, though our method formulations works for 6D compliance too if needed. For all polices, we use two frames of RGB image and three frames of end-effector poses. The policy is not sensitive to these numbers when they are small. \[ACP w.o. FFT\] uses 32 frames of wrench data sampled at 250Hz, while all other three policies uses one second of data at 7000Hz.

### IV-A Task I: Item Flipping

The task is to flip up an item with a point finger by pivoting it against a corner of a fixture (i.e., a wall), as exemplified in Fig. 2. The task has two challenges: 1) The finger needs to consistently maintain contact force during the flipping motion regardless of the item’s shape, weight, and fixture locations; 2) Larger contact force will cause the fixture to slide on the floor, making it harder to maintain good contact. We trained each method on 230 episodes of demonstrations collected on 15 different items for 300 epochs.

![Refer to caption](https://arxiv.org/html/2410.09309v2/extracted/6259248/figures/flipup_settings_v2.jpg)

Figure 5: Flipping Scenarios. We test the policy under a variety of settings that require the policy to adapt to different and unseen object geometries (a,b), configuration (c,d) and react to unexpected perturbations caused by fixture movements (e).

Test Scenarios. (Fig. 5) We ran 20 tests in each of the five scenarios below, making 100 tests per algorithm:

- Training Items: Items appeared in training data.
- Unseen Items: Items not seen in training.
- Push&Flip: Items start 5cm away from the fixture. This scenario requires the policy to first push the free-standing item towards the fixture then start the pivoting action.
- Varied Fixture Pose: Two different fixture poses.
- Unstable Fixture: Lighter fixture that causes unstable movements during the pivoting process that require the policy to quickly adapt.

![Refer to caption](https://arxiv.org/html/2410.09309v2/x5.png)

Figure 6: Flipping Results. \[Top\] Predicted stiffness in the world frame. The robot first decreases X-axis stiffness while approaching from the same direction, then gradually shifts the compliance to Z-axis, aligning with the contact normal. \[Bottom\] Predicted reference pose (yellow dot), virtual target poses (orange dot) and compliance direction (yellow line).

Results. The success rate is shown in Tab. I. Success is defined as the item being rotated greater than 70 degrees. Typical failure cases include items falling off the finger, motion getting stuck, or triggering robot force violation.

TABLE I: Flipping-up Success Rates (%)

|  | Train | Unseen | Push | Fixture | Unstable | All |
| --- | --- | --- | --- | --- | --- | --- |
|  | Items | Items | &Flip | Pose | Fixture |  |
| ACP | 90 | 95 | 95 | 100 | 100 | 96 |
| ACP w.o. FFT | 90 | 100 | 100 | 95 | 90 | 95 |
| Compliant Policy | 80 | 15 | 15 | 5 | 0 | 23 |
| Stiff Policy | 20 | 0 | 5 | 35 | 10 | 14 |

Findings. The two baselines \[Compliant Policy\] and \[Stiff Policy\] have a few successes when they can exploit the passive compliance in the system. They effectively applies a force when the predicted trajectory is in collision with a deformable item. When the item is rigid, or when the position uncertainty is not in a convenient direction, the baseline polices break the contacts and fail the task. On the contrary, both variations of ACP tolerates a large range of position uncertainties while maintain the needed contacts.

In this task, \[ACP w.o FFT\] has similar performance (95/100) as ACP (96/100), indicating that the force spectrum encoding is similarly useful to convolution in this task. This makes sense as the force signal does not contain much high frequency component in the flipping motion.

### IV-B Task II: Vase Wiping

The vase is randomly placed in front of the bimanual robot. The right upper side of the vase is marked by a random drawing using random colored dry markers. For this task, we equip each robot arm with a 3D-printed tool with two piece of kitchen sponges as wipers. The demonstrated motion uses the left arm to hold the vase while the right arm performs the wiping. We collected 200 demonstrations with various vase poses, marking shapes, and colors. Each demonstration includes one to five wipes to fully clean the markings.

Test Scenarios. We compared with the same set of algorithm alternatives as in the Item Flipping task. For each algorithm, we ran a total of 16 tests in the following four scenarios:

- Small Mark $\times 5$: easier cases that need only one wipe.
- Large Mark $\times 5$: require multiple wipes.
- Perturbation before contact (PbC) $\times 4$: move the vase right before the tool comes in contact with the vase to disturb the contact-engaging motion.
- Perturbation after contact (PaC) $\times 2$: move the vase after the tool is engaged to disturb the wiping motion.

Results. The quantitative results are summarized in Tab. II. All policies demonstrated wiping behaviors. However, the effects of the wipes are different. We define success as the mark being cleaned within three wipes, where we consider a vase “clean” if the remaining marks are within 1cm $\times$ 1cm.

![Refer to caption](https://arxiv.org/html/2410.09309v2/x6.png)

Figure 7: Wiping Results. \[Top\] Predicted stiffness of the wiping arm in world frame. \[Bottom\] Predicted reference (yellow dot) and virtual poses (orange dot). The Wiping arm’s compliant direction roughly aligns with the contact normal. Despite some errors, the wiping succeeded as the virtual target pose still pulls arm towards the vase.

TABLE II: Wiping Success Rates (%)

|  | Small | Large | PbC | PaC | All |
| --- | --- | --- | --- | --- | --- |
| ACP | 100 | 80 | 100 | 100 | 93.75 |
| ACP w.o. FFT | 100 | 60 | 75 | 100 | 81.25 |
| Compliant Policy | 60 | 20 | 25 | 100 | 43.75 |

![Refer to caption](https://arxiv.org/html/2410.09309v2/x7.png)

Figure 8: Wiping Comparisons. \[Top\] APC: maintains contact and follows desired trajectory. \[Middle\] Stiff Policy: Position noise causes excessive force that breaks the tool. \[Bottom\] Compliant Policy: Safe contact, but friction hinders wiping position accuracy and eventually loses contact.

Findings. Compared with the \[Stiff Policy\], \[ACP\] safely engages and maintains contacts during the wipes for its compliance. \[Stiff Policy\]’s contact force magnitude varies greatly with the accuracy of the position action. While the first few tests were successful, the robot broke its tool during the fourth test, as shown in Fig. 8, middle row.

Compared with the \[Compliant Policy\], \[ACP\] maintains accurate tracking of the desired motion, as shown in Fig. 8, top row. The wiping motion of the \[Compliant Policy\] deviates from the position target because it is sensitive to the friction from the vase. As a result, the sliding motion is insufficient to make a high quality wipe. The comparison of wiping results is shown in Tab. II, which excludes the \[Stiff Policy\] since it broke the robot tool frequently.

Compared with \[ACP w.o. FFT\], our policy with the FFT encoding performs better and finishes the task with fewer wipes. We observed that the FFT encoding, when used together with RGB encoding via cross-attention, makes better decision on the next best wiping location.

We also compare with a policy that predicts force and position, which is less robust without a suitable compliance profile. See our website for details.

## V CONCLUSIONS

In this work, we show that Adaptive Compliance Policy is an effective visuomotor policy for compliant manipulation. Extensive real-world results show that our approach is able to extract useful compliance from human demonstration, and thereby significantly improve the success rate of two contact-rich manipulation tasks.

## ACKNOWLEDGMENT

This work was supported in part by the Toyota Research Institute, NSF Award #2143601, #2037101, and #2132519. We would like to thank Google and TRI for the UR5 robot hardware. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors.