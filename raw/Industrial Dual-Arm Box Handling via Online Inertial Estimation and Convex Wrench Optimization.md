Kenzhi Iskandar Wong <sup>1</sup> Lin Yang <sup>2</sup> Qian Ying Lee Domenico Campolo [d.campolo@ntu.edu.sg](https://arxiv.org/html/2605.22021v1/mailto:d.campolo@ntu.edu.sg)

###### Abstract

Industrial robotic object handling often involves boxes and packages whose mass and center of mass are not known in advance. These uncertainties affect the force–moment balance required for stable lifting, and improper regulation of contact wrenches can lead to slip, object drop, orientation deviation, or excessive squeezing. This paper presents a friction-aware dual-arm box-handling framework for objects with unknown inertial properties. The proposed approach estimates the object mass and center of mass online from measured contact wrenches, and computes friction-feasible contact forces and torsional moments through a second-order cone program (SOCP) under ellipsoidal friction-limit-surface constraints. An offline trajectory refinement stage is also included to reduce undesired object–environment contact when geometric constraints are present. By enforcing friction feasibility as a hard constraint and minimizing contact effort within the feasible region, the framework achieves stable lifting without treating slip avoidance and excessive squeezing as separately tuned objectives. Experiments on a real dual-arm robotic system under different center-of-mass configurations demonstrate that the method lifts objects with unknown inertial properties while maintaining stable frictional contact.

###### keywords:

dual-arm manipulation, online inertial parameter estimation, friction limit surface, torsional friction, second-order cone programming

\[ntu\] organization=School of Mechanical and Aerospace Engineering, Nanyang Technological University, country=Singapore

## 1 Introduction

Industrial robotic box handling is a central operation in modern manufacturing and warehouse automation, where robots are increasingly expected to move boxes, containers, and packages across shelves, workstations, and storage systems [^25] [^7]. In these environments, boxes often have simple external geometry but uncertain physical properties: the mass may not be known in advance, and the center of mass may vary depending on the distribution of contents inside the box. These uncertainties directly affect the force–moment balance required for stable lifting. If not accounted for, they can cause object tilt, slip, drop, or excessive squeezing during manipulation.

In practical settings, industrial manipulation systems are commonly performed using position-controlled parallel-jaw grippers [^11] or suction-based grippers [^20] [^6], which are widely used in industrial and warehouse automation [^7]. While effective for objects with compatible geometry, these end-effectors depend on dedicated grasp conditions: parallel-jaw grippers require accessible grasp surfaces, while suction grippers require a reliable seal. These assumptions are difficult to satisfy for common packaging items such as boxes that are large relative to the gripper, have permeable surfaces, or contain uneven mass distributions.

An alternative is dual-arm frictional handling, where two arms support and transport an object through frictional contacts on opposite sides rather than through dedicated grasp closures [^29] [^16], as illustrated in Fig. 1. This approach is attractive for industrial box handling because it can accommodate a wider range of object geometries and contact conditions. However, because support is provided entirely through friction, improper regulation of the contact wrench can lead to slip, object drop, orientation deviation, or excessive squeezing. This makes explicit reasoning about contact forces and torques essential.

![Refer to caption](https://arxiv.org/html/2605.22021v1/figures/intro_overview.png)

Figure 1: Dual-arm box handling whose mass and center of mass are unknown to the robotic system. The system estimates these properties online and generates appropriate contact forces and torques to achieve stable lifting through frictional contacts without slip or excessive squeezing.

To build physical intuition, consider the case where the object CoM is known. When the CoM coincides with the geometric center and lies along the line connecting the two contact points, the lifting problem can be treated as planar, as shown in Fig. 2(a). In this case, gravity is symmetrically shared by the two manipulators, and lifting can be achieved by enforcing force equilibrium and friction cone constraints at each contact [^21] [^1], without requiring explicit contact torques. When the CoM is offset, the required force-torque balance changes. In planar lifting, the gravitational moment can still be compensated by redistributing the contact forces, as shown in Fig. 2(b). In three-dimensional lifting, however, an offset CoM can induce moments that cannot be balanced by contact forces alone. When the line of action of gravity does not intersect the line connecting the contact points, contact torques are required in addition to contact forces, as shown in Fig. 2(c). Thus, stable dual-arm box handling in 3D requires contact wrench distribution under friction constraints, online estimation of inertial parameters, and physical execution of the resulting wrench. We review prior work along these axes.

![Refer to caption](https://arxiv.org/html/2605.22021v1/x1.png)

Figure 2: Dual-arm lifting under different known CoM configurations. (a) Centered CoM: force equilibrium and friction constraints are sufficient. (b) Planar off-center CoM: the gravitational moment is balanced by redistributing contact forces. (c) 3D off-center CoM: contact torques are required when gravity does not intersect the contact line.

Existing approaches to cooperative manipulation have largely focused on object-level impedance, compliance, and internal-force control strategies for coordinating multiple manipulators while maintaining grasp stability [^24] [^4] [^10]. These methods have shown strong practical effectiveness for coordinated transport and disturbance rejection, but they generally do not explicitly optimize the full contact wrench under friction constraints, and instead rely on prescribed impedance behaviors or user-specified internal force references [^31] [^4]. More recent work has also considered cooperative manipulation under input constraints and underactuation using switching or constraint-aware control strategies [^17].

Related work on quasi-static grasp analysis and force distribution studies how to balance a given external load while satisfying contact constraints [^14] [^1]. However, these formulations typically assume that object properties and external loads are known a priori. In contrast, the framework proposed here estimates the object mass and CoM online from contact wrench measurements and uses the estimates during wrench distribution.

In parallel, learning-based grasping methods have also demonstrated strong empirical performance in selecting feasible grasps from sensory observations [^2] [^15], and end-to-end methods can map perception directly to grasping and lifting actions [^19]. However, mass and CoM are inertial properties that cannot be reliably inferred from visual appearance alone, since identical visual appearance does not imply identical mass distribution. Identifying these properties therefore requires physical interaction [^9]. Our approach follows this interaction-driven viewpoint and combines online inertial-parameter estimation with analytic, friction-aware wrench optimization.

In this work, we address dual-arm handling of objects with unknown mass and CoM. Under a quasi-static assumption, we estimate object properties online and compute friction-feasible contact forces and torques through convex optimization. The optimized wrenches are then executed on a real dual-arm robotic system equipped with force–torque sensors. Experiments show that the proposed method can lift objects with different CoM configurations while avoiding slip, object drop, and excessive squeezing. The results also show the importance of contact torques in three-dimensional lifting, especially when the object CoM is offset from the geometric center.

The main contributions of this work are threefold:

- We introduce a dual-arm box-handling framework for objects with unknown mass and CoM, integrating offline trajectory refinement, online inertial-parameter estimation, convex wrench optimization, and impedance-based execution into a closed-loop robotic manipulation pipeline.
- We formulate the 3D dual-arm contact wrench distribution problem as a second-order cone program (SOCP) under ellipsoidal friction-limit-surface constraints, explicitly capturing the coupling between tangential force and torsional moment about the contact normal.
- We introduce a safety-margin tightening of the ellipsoidal friction-limit-surface constraint, parameterized by $r_{s}$, which defines a conservative inner feasible wrench set and ensures that the commanded contact wrench remains strictly inside the friction-feasible region rather than operating on its boundary.

## 2 Methods

![Refer to caption](https://arxiv.org/html/2605.22021v1/x2.png)

Figure 3: Overview of the proposed framework. An offline trajectory refinement stage is followed by online object parameter estimation and contact wrench optimization. The optimized wrenches are executed through command synthesis and wrench correction in closed-loop interaction with the physical system.

We consider dual-arm handling of a rigid object through two frictional contacts. The object is lifted quasi-statically along a Cartesian trajectory obtained from demonstration [^18] or prior planning. While the desired object trajectory is known, the object mass and center of mass (CoM) are initially unknown. The objective is to estimate these parameters online and compute friction-feasible contact forces and torques that enable slip-free and drop-free lifting without excessive squeezing.

As summarized in Fig. 3, the proposed framework consists of three phases and a wrench execution layer:

- Phase I (offline) refines the nominal object trajectory offline to avoid undesired object–environment contact while remaining close to the reference motion.
- Phase II (online) estimates the object mass and CoM online from measured contact wrenches and robot kinematics.
- Phase III (online) computes the contact forces and torques by solving a convex optimization problem subject to force–moment equilibrium and friction constraints.

The resulting wrenches are executed through Cartesian impedance control with wrench feedback during lifting.

### 2.1 Phase I: Trajectory Refinement for Collision Avoidance

![Refer to caption](https://arxiv.org/html/2605.22021v1/figures/methods_phase1_flowchart.png)

Figure 4: Phase I trajectory refinement via black-box optimization. The objective is to refine a given object trajectory by minimizing object–environment contact while remaining close to the reference trajectory. Optimization is done using DMP-based black-box optimization and physics-based simulation.

Phase I refines the object trajectory offline to reduce undesired object–environment contact while remaining close to the reference motion. Let $\mathbf{z}_{\text{box}}^{*}(t)\in\mathbb{R}^{6}$ denote the object pose reference trajectory in local Cartesian pose coordinates. If the trajectory is obtained from demonstration [^18], an associated covariance $\Sigma_{\mathbf{z}_{\text{box}}^{*}}\in\mathbb{R}^{6\times 6}$ may also be available.

The trajectory is represented using Dynamical Movement Primitives (DMPs) [^22]. For each dimension $d=1,\ldots,6$, the DMP is parameterized by weights $\mathbf{\Theta}_{d}\in\mathbb{R}^{N}$, initialized by Locally Weighted Regression on the reference trajectory [^13]. To refine the trajectory, we define an exploration covariance over the DMP parameters:

$$
\Sigma_{\mathbf{\Theta}_{d}}=c\,\mathbf{I}_{N},
$$

where $c>0$ controls the exploration magnitude. At each iteration, $R$ candidate parameter vectors are sampled as

$$
\mathbf{\Theta}_{d,r}\sim\mathcal{N}\!\big(\mathbf{\Theta}_{d},\,\Sigma_{\mathbf{\Theta}_{d}}\big),\quad r=1,\ldots,R.
$$

Solving the sampled DMPs produces per-dimension candidate trajectories $\mathbf{z}_{\text{box},d,r}(t)$, which are stacked to form the full candidate object trajectory $\mathbf{z}_{\text{box},r}(t)$. Each candidate is simulated in Drake [^27], producing the simulated object trajectory and object–environment contact forces:

$$
\big(\mathbf{z}_{\text{box},r}^{\prime}(t),\,\mathbf{F}_{\text{env},r}(t)\big)\leftarrow\textbf{Drake}\big(\mathbf{z}_{\text{box},r}(t)\big).
$$

Each rollout is evaluated using

$$
J_{r}=\alpha J_{1,r}+(1-\alpha)J_{2,r},
$$

where $\alpha\in(0,1)$ balances tracking of the reference trajectory and avoidance of object–environment contact. The tracking and contact costs are

$$
J_{1,r}=\sum_{t}\big(\mathbf{z}^{\prime}_{\text{box},r}(t)-\mathbf{z}_{\text{box}}^{*}(t)\big)^{\top}\Sigma^{-1}_{\mathbf{z}_{\text{box}}^{*}}\big(\mathbf{z}^{\prime}_{\text{box},r}(t)-\mathbf{z}_{\text{box}}^{*}(t)\big),
$$
 
$$
J_{2,r}=\sum_{t}\lVert\mathbf{F}_{\text{env},r}(t)\rVert_{2}^{2}.
$$

The first term (5) is a Mahalanobis tracking cost, which reduces to a squared Euclidean tracking cost when covariance information is unavailable. The second term (6) penalizes object–environment contact through the squared Euclidean norm of the contact force between the object and environment $\mathbf{F}_{\text{env},r}(t)$.

The rollouts are sorted by cost, with permutation $\pi$ satisfying

$$
J_{\pi(1)}\leq J_{\pi(2)}\leq\cdots\leq J_{\pi(R)}.
$$

The DMP parameters are updated using the lowest-cost rollout,

$$
\mathbf{\Theta}_{d}\leftarrow\mathbf{\Theta}_{d,\pi(1)},\quad d=1,\ldots,6,
$$

and the exploration covariance is updated using the Cross-Entropy Method (CEM) [^26] over the best $K_{e}\leq R$ elite rollouts:

$$
\displaystyle\Sigma_{\mathbf{\Theta}_{d}}\leftarrow\frac{1}{K_{e}}\sum_{k=1}^{K_{e}}\big(\mathbf{\Theta}_{d,\pi(k)}-\mathbf{\Theta}_{d}\big)\big(\mathbf{\Theta}_{d,\pi(k)}-\mathbf{\Theta}_{d}\big)^{\top},
$$
$$
\displaystyle d=1,\ldots,6.
$$

The procedure repeats until

$$
\max_{i,j}\left(\Sigma_{\mathbf{\Theta}_{d}}\right)_{ij}<\epsilon_{\text{conv}},\quad d=1,\ldots,6,
$$

where $0<\epsilon_{\text{conv}}<c$. Upon convergence, the optimized DMP parameters define the refined object trajectory $\mathbf{z}_{\text{box}}^{\mathrm{ref}}(t)$, from which the end-effector reference trajectories $\mathbf{z}_{L}^{\mathrm{ref}}(t)$ and $\mathbf{z}_{R}^{\mathrm{ref}}(t)$ are obtained using the known grasp geometry.

### 2.2 Phase II: Estimation of Object Parameters

![Refer to caption](https://arxiv.org/html/2605.22021v1/x3.png)

(a)

![Refer to caption](https://arxiv.org/html/2605.22021v1/x5.png)

Figure 6: Object, left handle, and right handle coordinate frames. The box-fixed { B } \\{B\\} frame x l o c a, y z \\{x\_{local},y\_{local},z\_{local}\\} is defined at the geometric center, with z\_{local} aligned with the lifting direction. The center of mass (CoM) is expressed in this frame. The contact normal directions n L n\_{L} and R n\_{R} at the left and right handles are indicated. An additional weight is used to vary the mass distribution.

Throughout the proposed framework, robot-object interaction is modeled through Cartesian impedance control, where the interaction wrench is proportional to the difference between the commanded and measured end-effector poses [^5]:

$$
\mathbf{w}=\mathbf{K}(\mathbf{u}-\mathbf{z}),
$$

where $\mathbf{w}\in\mathbb{R}^{6}$ is the task-space wrench, and $\mathbf{u},\mathbf{z}\in\mathbb{\mathbb{R}}^{6}$ are the commanded pose, and end-effector pose, respectively. $\mathbf{K}\in\mathbb{R}^{6\times 6}$ is a positive definite stiffness matrix. Damping is omitted because it is handled by the low-level controller.

For each handle $i\in\{L,R\}$, the contact is represented by a point located at the center of the handle–box contact region. Its position relative to the box geometric center is denoted by $\mathbf{r}_{i}\in\mathbb{R}^{3}$. During Phase II, the measured contact wrench is written as $\hat{\mathbf{w}}_{i}=[\hat{\mathbf{f}}_{i}^{\top}\;\hat{\boldsymbol{\tau}}_{i}^{\top}]^{\top}\in\mathbb{R}^{6}$, where $\hat{(\cdot)}$ denotes a measured quantity. Here, $\hat{\mathbf{f}}_{i}$ is the measured force applied on the object and $\hat{\boldsymbol{\tau}}_{i}$ is the measured free contact moment about the contact point. Both are expressed in the object-fixed frame $\{B\}$ shown in Fig. 6.

#### 2.2.1 Lift initiation and lift-off detection

In Phase II, we estimate the object mass and CoM after lift-off. To initiate lifting, the commanded end-effector poses are ramped along predefined unit lifting directions $\boldsymbol{\ell}_{i}$, whose orientation components are zero:

$$
\mathbf{u}_{i}(k)=\mathbf{u}_{i}(0)+\alpha(k)\boldsymbol{\ell}_{i},\qquad i\in\{L,R\},
$$

where $k$ is the discrete control step, $\mathbf{u}_{i}(0)$ is the initial commanded pose, and $\alpha(k)$ is a scalar lift magnitude. The lift magnitude is initialized as $\alpha(0)=0$ and increased monotonically:

$$
\alpha(k+1)=\alpha(k)+\Delta\alpha,
$$

where $\Delta\alpha>0$ is the lift increment. This gradually increases the commanded lift until the object detaches from the ground.

Lift-off is detected from end-effector feedback. Let $\mathbf{p}_{i}(k)\in\mathbb{R}^{3}$ denote the position component of the measured end-effector pose. The vertical displacement of each handle is

$$
\Delta h_{i}(k)=\big(\mathbf{p}_{i}(k)-\mathbf{p}_{i}(0)\big)^{\top}\hat{\mathbf{z}}_{local},\qquad i\in\{L,R\}.
$$

Lift-off is declared when

$$
\min\{\Delta h_{L}(k),\Delta h_{R}(k)\}\geq h_{\mathrm{lift}},
$$

where $h_{\mathrm{lift}}$ is chosen conservatively to ensure clearance even under small object tilting.

#### 2.2.2 Mass estimation

Once lift-off is achieved, the object is assumed to be in equilibrium, and the mass is estimated from force balance along the lifting direction. We collect $M$ post-lift-off measurements and formulate the estimation as a least-squares problem. All quantities are expressed in the box-fixed frame $\{B\}$, where the contact locations and CoM are constant.

The gravity vector expressed in $\{B\}$ is defined as

$$
\mathbf{g}=-g\,\hat{\mathbf{z}}_{local},\qquad g>0,
$$

where $g$ is the gravitational acceleration magnitude.

For the $j$ -th post-lift-off measurement, with $j=1,\ldots,M$, define the total measured contact force as

$$
\hat{\mathbf{F}}_{j}=\hat{\mathbf{f}}_{L,j}+\hat{\mathbf{f}}_{R,j}.
$$

Force equilibrium gives

$$
\hat{\mathbf{F}}_{j}+m\mathbf{g}\approx\mathbf{0},
$$

where $m$ is the object mass. Accounting for measurement noise and quasi-static modeling error, this can be written as

$$
\hat{\mathbf{F}}_{j}=-\mathbf{g}m+\boldsymbol{\epsilon}_{F,j}.
$$

where $\boldsymbol{\epsilon}_{F,j}$ captures measurement noise and quasi-static modeling error. Stacking all measurements gives the linear regression model:

$$
\mathbf{y}_{F}=\boldsymbol{\Phi}_{F}m+\boldsymbol{\epsilon}_{F},
$$

where

$$
\mathbf{y}_{F}=\begin{bmatrix}\hat{\mathbf{F}}_{1}\\
\vdots\\
\hat{\mathbf{F}}_{M}\end{bmatrix}\in\mathbb{R}^{3M},\qquad\boldsymbol{\Phi}_{F}=\begin{bmatrix}-\mathbf{g}\\
\vdots\\
-\mathbf{g}\end{bmatrix}\in\mathbb{R}^{3M\times 1}.
$$

The mass estimate is obtained as

$$
\hat{m}=\boldsymbol{\Phi}_{F}^{\dagger}\mathbf{y}_{F},
$$

where $(\cdot)^{\dagger}$ denotes the Moore–Penrose pseudoinverse.

#### 2.2.3 CoM estimation

The CoM is expressed in the object-fixed frame as

$$
\mathbf{r}_{\mathrm{com}}=\begin{bmatrix}r_{x}&r_{y}&r_{z}\end{bmatrix}^{\top}.
$$

For the $j$ -th measurement, define the resultant measured contact moment about the box geometric center as

$$
\hat{\mathbf{s}}_{j}=\sum_{i\in\{L,R\}}\left(\mathbf{r}_{i}\times\hat{\mathbf{f}}_{i,j}+\hat{\boldsymbol{\tau}}_{i,j}\right),
$$

where $\mathbf{r}_{i}$ is the vector from the box geometric center to contact $i$. Moment equilibrium gives

$$
\hat{\mathbf{s}}_{j}+\mathbf{r}_{\mathrm{com}}\times\left(\hat{m}\mathbf{g}\right)\approx\mathbf{0}.
$$

Rewriting the cross product in skew-symmetric matrix form and accounting for measurement noise and quasi-static modeling error gives

$$
\hat{\mathbf{s}}_{j}=[\hat{m}\mathbf{g}]_{\times}\mathbf{r}_{\mathrm{com}}+\boldsymbol{\epsilon}_{r,j}.
$$

Stacking all measurements gives the linear regression model:

$$
\mathbf{y}_{r}=\boldsymbol{\Phi}_{r}\mathbf{r}_{\mathrm{com}}+\boldsymbol{\epsilon}_{r},
$$

where

$$
\mathbf{y}_{r}=\begin{bmatrix}\hat{\mathbf{s}}_{1}\\
\vdots\\
\hat{\mathbf{s}}_{M}\end{bmatrix}\in\mathbb{R}^{3M},\qquad\boldsymbol{\Phi}_{r}=\begin{bmatrix}[\hat{m}\mathbf{g}]_{\times}\\
\vdots\\
[\hat{m}\mathbf{g}]_{\times}\end{bmatrix}\in\mathbb{R}^{3M\times 3}.
$$

The CoM estimate is then

$$
\hat{\mathbf{r}}_{\mathrm{com}}=\boldsymbol{\Phi}_{r}^{\dagger}\mathbf{y}_{r}.
$$

The rank of $\boldsymbol{\Phi}_{r}$ determines which CoM components are observable. In the present lifting setting, gravity acts along $z_{\mathrm{local}}$, so only the in-plane CoM coordinates affect the lifting moment equilibrium. The component $r_{z}$ does not contribute to the gravitational moment during upright lifting. Therefore, the estimated in-plane coordinates $\hat{r}_{x}$ and $\hat{r}_{y}$ are used in Phase III for contact wrench optimization.

### 2.3 Phase III: Contact Force and Torque Optimization via SOCP

In Phase III, the objective is to compute optimal contact wrenches at the two robot end-effectors that support the object against gravity while respecting frictional contact constraints. The formulation is based on the quasi-static assumption and uses the online estimated object mass and CoM obtained in Phase II.

#### 2.3.1 Contact model

Let $\hat{\mathbf{n}}_{i}\in\mathbb{R}^{3}$ denote the outward unit normal at contact $i\in\{L,R\}$, see Fig. 6. The orthogonal projector onto the tangent plane is defined as

$$
\boldsymbol{\Pi}_{i}=\mathbf{I}_{3}-\hat{\mathbf{n}}_{i}\hat{\mathbf{n}}_{i}^{\top}.
$$

The signed normal force and tangential force are then

$$
f_{n,i}=\hat{\mathbf{n}}_{i}^{\top}\mathbf{f}_{i},\qquad\mathbf{f}_{t,i}=\boldsymbol{\Pi}_{i}\mathbf{f}_{i},
$$

where compression corresponds to $f_{n,i}\leq 0$. Similarly, the contact moment is decomposed into its torsional and tangential components as

$$
\tau_{n,i}=\hat{\mathbf{n}}_{i}^{\top}\boldsymbol{\tau}_{i},\qquad\boldsymbol{\tau}_{t,i}=\boldsymbol{\Pi}_{i}\boldsymbol{\tau}_{i}.
$$

Since rotational balance due to CoM offsets can be achieved by the contact forces together with the torsional moment about the contact normal, the contact model does not require bending moments about the tangent directions. Therefore, we impose

$$
\boldsymbol{\tau}_{t,i}=\boldsymbol{\Pi}_{i}\boldsymbol{\tau}_{i}=\mathbf{0}.
$$

#### 2.3.2 Equilibrium constraints

Under quasi-static equilibrium condition, the desired contact wrenches must balance the estimated gravitational wrench. Force and moment equilibrium about the box geometric center require

$$
\displaystyle\sum_{i\in\{L,R\}}\mathbf{f}_{i}+\hat{m}\mathbf{g}
$$
 
$$
\displaystyle=\mathbf{0},
$$
$$
\displaystyle\sum_{i\in\{L,R\}}\left(\mathbf{r}_{i}\times\mathbf{f}_{i}+\boldsymbol{\tau}_{i}\right)+\hat{\mathbf{r}}_{\mathrm{com}}\times\hat{m}\mathbf{g}
$$
 
$$
\displaystyle=\mathbf{0},
$$

where $\mathbf{r}_{i}$ is the vector from the box geometric center to contact $i$.

#### 2.3.3 Wrench redundancy and nullspace interpretation

The equilibrium constraints define the object-level wrench required for stable lifting, but they do not uniquely determine the individual contact wrenches. To make this redundancy explicit, define the stacked contact wrench

$$
x=\begin{bmatrix}\mathbf{w}_{L}^{\top}&\mathbf{w}_{R}^{\top}\end{bmatrix}^{\top}=\begin{bmatrix}\mathbf{f}_{L}^{\top}&\boldsymbol{\tau}_{L}^{\top}&\mathbf{f}_{R}^{\top}&\boldsymbol{\tau}_{R}^{\top}\end{bmatrix}^{\top}\in\mathbb{R}^{12}.
$$

The net wrench applied to the object about the box geometric center can be written as

$$
\mathbf{w}_{\mathrm{obj}}=\mathbf{G}x,
$$

where the grasp map is

$$
\mathbf{G}=\begin{bmatrix}\mathbf{I}_{3}&\mathbf{0}_{3}&\mathbf{I}_{3}&\mathbf{0}_{3}\\
[\mathbf{r}_{L}]_{\times}&\mathbf{I}_{3}&[\mathbf{r}_{R}]_{\times}&\mathbf{I}_{3}\end{bmatrix}\in\mathbb{R}^{6\times 12}.
$$

Here, $[\mathbf{r}_{i}]_{\times}\mathbf{f}_{i}=\mathbf{r}_{i}\times\mathbf{f}_{i}$. The equilibrium constraints (34)–(35) can therefore be written compactly as

$$
\mathbf{G}x=-\begin{bmatrix}\hat{m}\mathbf{g}\\
\hat{\mathbf{r}}_{\mathrm{com}}\times\hat{m}\mathbf{g}\end{bmatrix}.
$$

Since $\mathbf{G}$ maps a 12-dimensional contact wrench vector to a 6-dimensional object wrench, the contact wrench distribution is generally redundant at the equilibrium level. This means that there can be nonzero contact-wrench variations $x_{\mathrm{null}}$ that satisfy

$$
\mathbf{G}x_{\mathrm{null}}=\mathbf{0}.
$$

Such variations lie in the kernel, or nullspace, of $\mathbf{G}$, and therefore do not change the net wrench applied to the box. Physically, $x_{\mathrm{null}}$ represents an internal wrench: it changes how forces and torques are distributed between the two contacts without changing the object-level equilibrium. Next, the friction constraints and minimum-effort objective are introduced to select one physically feasible wrench distribution from this redundant set.

#### 2.3.4 Friction constraints

Slip at each contact is prevented using a friction limit surface [^12], which captures the coupling between tangential force and torsional moment. For contact $i\in\{L,R\}$, the limit surface is modeled using the ellipsoidal approximation [^30] as:

$$
\left(\frac{\|\mathbf{f}_{t,i}\|}{f_{t,i}^{\max}}\right)^{2}+\left(\frac{\tau_{n,i}}{\tau_{n,i}^{\max}}\right)^{2}\leq(1-r_{s})^{2},
$$

where $r_{s}\in[0,1)$ is a safety margin. More details on this safety margin can be found in A.

The size of the friction limit surface depends on the contact normal force, the friction coefficient, and the contact geometry. Since compression corresponds to $f_{n,i}\leq 0$, the compressive normal-force magnitude is $-f_{n,i}$. Hence, the maximum admissible tangential force follows from Coulomb friction, while a finite contact patch can also sustain a torsional moment about the contact normal:

$$
f_{t,i}^{\max}=\mu(-f_{n,i}),
$$
 
$$
\tau_{n,i}^{\max}=\mu(-f_{n,i})R_{\mathrm{eff},i}.
$$

where $\mu$ is the friction coefficient, and $R_{\mathrm{eff},i}$ is an effective contact radius determined by the contact geometry and pressure distribution [^12] [^30]. The derivation of $R_{\mathrm{eff},i}$ is given in B.

#### 2.3.5 Optimization problem

Using the stacked contact wrench $x$ defined in (36), the contact wrenches are selected by minimizing a weighted wrench effort. For each contact wrench $\mathbf{w}_{i}$, define the weighting matrix

$$
\mathbf{Q}_{c}=\operatorname{diag}\left(1,1,1,\frac{1}{l_{c}^{2}},\frac{1}{l_{c}^{2}},\frac{1}{l_{c}^{2}}\right),
$$

where $l_{c}>0$ is a characteristic contact length chosen from the contact patch size to account for the different units of force and moment in the weighted wrench cost. The stacked weighting matrix is then

$$
\mathbf{Q}=\operatorname{blkdiag}(\mathbf{Q}_{c},\mathbf{Q}_{c}).
$$

The contact wrenches are selected by minimizing the weighted wrench effort:

$$
\min_{x}\ x^{\top}\mathbf{Q}x=\min_{x}\left(\mathbf{w}_{L}^{\top}\mathbf{Q}_{c}\mathbf{w}_{L}+\mathbf{w}_{R}^{\top}\mathbf{Q}_{c}\mathbf{w}_{R}\right).
$$

Since $\mathbf{Q}$ is positive definite, minimizing the squared weighted effort $x^{\top}\mathbf{Q}x$ has the same minimizer as minimizing $\|\mathbf{Q}^{1/2}x\|_{2}$. Introducing an auxiliary variable $t\in\mathbb{R}$, the optimization problem is written in epigraph form as

$$
\displaystyle\min_{x,t}
$$
 
$$
\displaystyle t
$$
 
$$
\displaystyle\hat{\mathbf{n}}_{i}^{\top}\mathbf{f}_{i}\leq 0,\quad i\in\{L,R\},\quad\text{(compression)}
$$
 
$$
\displaystyle\boldsymbol{\Pi}_{i}\boldsymbol{\tau}_{i}=\mathbf{0},\quad i\in\{L,R\},\quad\text{(no bending moment)}
$$
 
$$
\displaystyle\text{equilibrium constraints }\eqref{eq:phase3_force_balance}\text{--}\eqref{eq:phase3_moment_balance},
$$
$$
\displaystyle\text{friction limit constraints }\eqref{eq:limitsurface_ellipsoid},\qquad i\in\{L,R\},
$$
$$
\displaystyle\left\|\mathbf{Q}^{1/2}x\right\|_{2}\leq t,
$$

where $\mathbf{Q}^{1/2}$ is the square-root weighting matrix.

The squared-norm epigraph constraint can be written as a second-order cone constraint, so the resulting problem is a convex second-order cone program (SOCP) and can be solved efficiently using standard interior-point methods [^3]. The optimal solution is denoted by $\mathbf{w}_{i}^{\mathrm{des}}$, $i\in\{L,R\}$, and is passed to the execution layer to control the box trajectory from Phase I in quasi-static manner.

### 2.4 Execution

Phase III provides the desired contact wrenches $\mathbf{w}_{i}^{\mathrm{des}}$, $i\in\{L,R\}$. These wrenches are executed along the end-effector references $\mathbf{z}_{i}^{\mathrm{ref}}(t)$, which are inferred from the refined object trajectory $\mathbf{z}_{\text{box}}^{\mathrm{ref}}(t)$ from Phase I using the known grasp geometry. Using the Cartesian impedance relation introduced in Fig. 5(a), the desired wrench is realized by biasing the commanded pose relative to the reference trajectory. The nominal command is defined as

$$
\mathbf{u}_{i,0}(t)=\mathbf{z}_{i}^{\mathrm{ref}}(t)+\mathbf{K}^{-1}\mathbf{w}_{i}^{\mathrm{des}},
$$

where $\mathbf{K}$ is a positive-definite Cartesian stiffness matrix.

In practice, the realized wrench may differ from the desired value due to actuator dynamics, sensor noise, and environmental uncertainty [^28]. To compensate for these effects, wrench feedback is added. Let $\mathbf{\hat{w}}_{i}(t)$ denote the measured wrench, and define the wrench error as

$$
\mathbf{e}_{i}(t)=\mathbf{\hat{w}}_{i}(t)-\mathbf{w}_{i}^{\mathrm{des}}
$$

A PID controller maps this error to a bounded corrective pose increment $\Delta\mathbf{u}_{i}(t)$, applied primarily along the squeezing direction. The final command is then given by

$$
\mathbf{u}_{i}(t)=\mathbf{u}_{i,0}(t)+\Delta\mathbf{u}_{i}(t),
$$

which is sent to the robot controller and executed under Cartesian impedance control.

## 3 Experiments and Results

### 3.1 Experiment Setup

![Refer to caption](https://arxiv.org/html/2605.22021v1/figures/experiment_setup.png)

(a)

The experimental setup is shown in Fig. 7. The system consists of two 7-DoF Kinova robotic arms, each equipped with an ATI Mini40 force/torque (F/T) sensor at the wrist. A custom handle with a friction-enhanced contact surface (coefficient of friction 0.4) is attached to each sensor. The handle contact patch has dimensions of $7\,\text{cm}\times 10\,\text{cm}$. The manipulated object is a rectangular box with a base mass of 1.7 kg. An additional mass of 0.5 kg is placed inside the box to vary the CoM location. The Cartesian impedance controller uses a diagonal stiffness matrix with translational stiffness of $1000\,\text{N/m}$ and rotational stiffness of $10\,\text{Nm/rad}$.

Two CoM configurations are considered in the experiments. In Configuration 1, the additional mass is placed with an offset in the positive $x_{local}$ and $y_{local}$ directions, as illustrated in Fig. 6. In Configuration 2, the mass is repositioned to produce a different CoM offset, resulting in a distinct mass distribution. Each configuration is repeated three times, resulting in three independent runs per configuration.

### 3.2 Phase I Result (Trajectory Refinement for Collision Avoidance)

![Refer to caption](https://arxiv.org/html/2605.22021v1/x6.png)

Figure 8: Object motion before (top) and after (bottom) Phase I trajectory refinement, evaluated in the Drake physics engine. The refined trajectory reduces object–environment contact while maintaining the intended motion.

![Refer to caption](https://arxiv.org/html/2605.22021v1/x7.png)

Figure 9: Environment contact wrench before and after Phase I refinement. After optimization, most force and torque components remain close to zero, indicating successful collision avoidance.

Phase I refinement is demonstrated for Configuration 1, where the nominal trajectory produces object–environment contact during extraction. Configuration 2 does not require refinement because its nominal trajectory is already collision-free in the simulated environment. For Configuration 1, the initial object trajectory $\mathbf{z}_{\text{box}}^{*}(t)$ was generated using [^18]. The refinement was performed in Drake [^27] with $R=50$ rollouts per iteration, $c=1000$ in (1) to set the initial exploration magnitude, $\alpha=0.2$ in (4) to balance trajectory tracking and contact avoidance, $K_{e}=5$ elite samples for the update in (9), and $\epsilon_{\text{conv}}=10^{-2}$ in (10) as the convergence threshold.

In this work, the undesired collision is primarily associated with object motion along the $y_{\mathrm{box}}$ - and $z_{\mathrm{box}}$ -position components. Thus, trajectory exploration is restricted to these components, while the remaining position and orientation components are kept fixed to the reference trajectory.

Fig. 8 shows the object motion before and after trajectory refinement. The initial trajectory produces collision with the environment, whereas the refined trajectory avoids the unintended contact while remaining close to the original motion.

The environment contact wrench in Fig. 9 represents the wrench exerted by the environment on the object, where the environment may correspond to either the ground or the shelf depending on the stage of the motion. Before refinement, the nominal trajectory produces large contact force and torque components, indicating undesired object–environment collision during extraction. These large components correspond to contact between the box and the shelf. After refinement, most force and torque components remain close to zero, confirming that the unintended contact is reduced. The nonzero $f_{y}$ and $f_{z}$ components near the end of the motion correspond to the intended box–shelf interaction during placement, where $f_{y}$ represents friction between the box and the shelf surface and $f_{z}$ represents the normal support from the shelf. Therefore, these final nonzero components are not considered collision during extraction.

The refined trajectory is then used as the reference object motion in the execution experiments in Section 3.5.

### 3.3 Phase II Result (Estimation of Object Parameters)

Table 1: Mass estimation results for two configurations.

<table><tbody><tr><th>Config</th><td>Run</td><td>Experiment</td><td>Mass (kg)</td><td>Error (%)</td></tr><tr><th rowspan="7">1</th><td>–</td><td>GT</td><td>2.20</td><td>–</td></tr><tr><td rowspan="2">Run 1</td><td>Sim</td><td>2.2016</td><td>0.07</td></tr><tr><td>Real</td><td>2.3073</td><td>4.88</td></tr><tr><td rowspan="2">Run 2</td><td>Sim</td><td>2.1998</td><td>0.01</td></tr><tr><td>Real</td><td>2.2989</td><td>4.50</td></tr><tr><td rowspan="2">Run 3</td><td>Sim</td><td>2.1993</td><td>0.03</td></tr><tr><td>Real</td><td>2.3292</td><td>5.87</td></tr><tr><th rowspan="7">2</th><td>–</td><td>GT</td><td>2.20</td><td>–</td></tr><tr><td rowspan="2">Run 1</td><td>Sim</td><td>2.2016</td><td>0.07</td></tr><tr><td>Real</td><td>2.2839</td><td>3.81</td></tr><tr><td rowspan="2">Run 2</td><td>Sim</td><td>2.1990</td><td>0.05</td></tr><tr><td>Real</td><td>2.3528</td><td>6.04</td></tr><tr><td rowspan="2">Run 3</td><td>Sim</td><td>2.2003</td><td>0.01</td></tr><tr><td>Real</td><td>2.3918</td><td>8.72</td></tr></tbody></table>

Table 2: Center-of-mass (CoM) estimation results for two configurations.

<table><tbody><tr><td>Config</td><td>Run</td><td>Experiment</td><td>CoM <math><semantics><mrow><mo>(</mo><msub><mi>r</mi> <mi>x</mi></msub><mo>,</mo><msub><mi>r</mi> <mi>y</mi></msub><mo>)</mo></mrow> <annotation>(r_{x},r_{y})</annotation></semantics></math> (mm)</td><td>Error (%)</td></tr><tr><td rowspan="7">1</td><td>–</td><td>GT</td><td><math><semantics><mrow><mo>(</mo><mn>20.50</mn><mo>,</mo><mn> 11.40</mn><mo>)</mo></mrow> <annotation>(20.50,\;11.40)</annotation></semantics></math></td><td>–</td></tr><tr><td rowspan="2">Run 1</td><td>Sim</td><td><math><semantics><mrow><mo>(</mo><mn>20.51</mn><mo>,</mo><mn> 11.44</mn><mo>)</mo></mrow> <annotation>(20.51,\;11.44)</annotation></semantics></math></td><td>0.01</td></tr><tr><td>Real</td><td><math><semantics><mrow><mo>(</mo><mn>8.58</mn><mo>,</mo><mn> 14.19</mn><mo>)</mo></mrow> <annotation>(8.58,\;14.19)</annotation></semantics></math></td><td>3.63</td></tr><tr><td rowspan="2">Run 2</td><td>Sim</td><td><math><semantics><mrow><mo>(</mo><mn>20.53</mn><mo>,</mo><mn> 11.40</mn><mo>)</mo></mrow> <annotation>(20.53,\;11.40)</annotation></semantics></math></td><td>0.01</td></tr><tr><td>Real</td><td><math><semantics><mrow><mo>(</mo><mn>7.65</mn><mo>,</mo><mn> 13.87</mn><mo>)</mo></mrow> <annotation>(7.65,\;13.87)</annotation></semantics></math></td><td>3.90</td></tr><tr><td rowspan="2">Run 3</td><td>Sim</td><td><math><semantics><mrow><mo>(</mo><mn>20.47</mn><mo>,</mo><mn> 11.39</mn><mo>)</mo></mrow> <annotation>(20.47,\;11.39)</annotation></semantics></math></td><td>0.03</td></tr><tr><td>Real</td><td><math><semantics><mrow><mo>(</mo><mn>8.21</mn><mo>,</mo><mn> 14.33</mn><mo>)</mo></mrow> <annotation>(8.21,\;14.33)</annotation></semantics></math></td><td>3.76</td></tr><tr><td rowspan="7">2</td><td>–</td><td>GT</td><td><math><semantics><mrow><mo>(</mo><mn>6.80</mn><mo>,</mo><mrow><mo>−</mo> <mn>11.40</mn></mrow><mo>)</mo></mrow> <annotation>(6.80,\;-11.40)</annotation></semantics></math></td><td>–</td></tr><tr><td rowspan="2">Run 1</td><td>Sim</td><td><math><semantics><mrow><mo>(</mo><mn>6.79</mn><mo>,</mo><mrow><mo>−</mo> <mn>11.42</mn></mrow><mo>)</mo></mrow> <annotation>(6.79,\;-11.42)</annotation></semantics></math></td><td>0.02</td></tr><tr><td>Real</td><td><math><semantics><mrow><mo>(</mo><mn>5.31</mn><mo>,</mo><mrow><mo>−</mo> <mn>16.01</mn></mrow><mo>)</mo></mrow> <annotation>(5.31,\;-16.01)</annotation></semantics></math></td><td>3.34</td></tr><tr><td rowspan="2">Run 2</td><td>Sim</td><td><math><semantics><mrow><mo>(</mo><mn>6.85</mn><mo>,</mo><mrow><mo>−</mo> <mn>11.48</mn></mrow><mo>)</mo></mrow> <annotation>(6.85,\;-11.48)</annotation></semantics></math></td><td>0.03</td></tr><tr><td>Real</td><td><math><semantics><mrow><mo>(</mo><mn>1.65</mn><mo>,</mo><mrow><mo>−</mo> <mn>16.48</mn></mrow><mo>)</mo></mrow> <annotation>(1.65,\;-16.48)</annotation></semantics></math></td><td>4.50</td></tr><tr><td rowspan="2">Run 3</td><td>Sim</td><td><math><semantics><mrow><mo>(</mo><mn>6.83</mn><mo>,</mo><mrow><mo>−</mo> <mn>11.47</mn></mrow><mo>)</mo></mrow> <annotation>(6.83,\;-11.47)</annotation></semantics></math></td><td>0.03</td></tr><tr><td>Real</td><td><math><semantics><mrow><mo>(</mo><mn>3.00</mn><mo>,</mo><mrow><mo>−</mo> <mn>18.79</mn></mrow><mo>)</mo></mrow> <annotation>(3.00,\;-18.79)</annotation></semantics></math></td><td>4.49</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2605.22021v1/x8.png)

Figure 10: Phase II CoM estimation results visualized as the inferred additional-mass location for Configuration 2 (Run 1). The additional-mass location is inferred from the estimated CoM of the combined box-additional mass system using the known box mass and additional mass. Simulation estimates closely align with the ground truth, while real-world experiment estimates show a consistent offset but remain tightly clustered, indicating low variability. The observed deviations in the real-world are on the order of a few percent relative to the object size (300 mm).

The estimated object parameters from Phase II are summarized in Table 1 and Table 2, where GT, Sim, and Real denote ground truth, simulation, and real-world experiments, respectively. Three independent runs are reported for each configuration. In simulation, small measurement noise is introduced to the contact wrench signals to reflect realistic sensing conditions. For both configurations, the simulation results closely match the ground truth across all runs, with negligible errors in both mass and CoM location. The mass error remains below $0.1\%$, and the CoM error also remains below $0.1\%$, confirming the correctness of the estimation formulation under near-ideal conditions.

In the real-world experiments, both the estimated mass and CoM exhibit consistent deviations from the ground truth across runs. The mass is overestimated by approximately $3$ – $9\%$. Despite this bias, the estimates remain consistent across repeated runs, indicating that the error is systematic rather than purely random. Similarly, the CoM estimates show an offset from the ground truth but remain tightly clustered within each configuration, indicating low run-to-run variability.

Overall, the CoM error remains within a few percent relative to the object dimensions $(300~\mathrm{mm}\times 200~\mathrm{mm})$, demonstrating that the method provides reasonably accurate and repeatable parameter estimates in practical settings. The sources of the observed systematic error are discussed further in Section 4.

![Refer to caption](https://arxiv.org/html/2605.22021v1/x9.png)

(a)

To provide a more intuitive visualization of the observed CoM offset, Fig. 10 shows the CoM estimation result for Configuration 2, Run 1, represented as the inferred additional-mass location. This visualization converts the estimated CoM of the combined box–additional-mass system into the corresponding location of the internal added mass, making the effect of the CoM estimation error easier to interpret.

### 3.4 Phase III Result (Contact Force and Torque Optimization via SOCP)

The optimized contact wrenches obtained from solving (47) are visualized in Fig. 11 for both configurations, shown for a representative run (Run 1). The results from the remaining runs exhibit similar trends and are therefore omitted for clarity. The figure evaluates the optimized wrenches from two complementary perspectives: the Coulomb friction cone in Fig. 11(a), which shows the contact force feasibility, and the friction limit surface in Fig. 11(b), which shows the coupling between tangential force and torsional moment.

As shown in Fig. 11(a), the contact force vectors are plotted in the local $x$ - $z$ plane, where the $x$ -direction corresponds to the contact normal direction and the $z$ -direction corresponds to the load-support direction. The solid gray lines denote the nominal Coulomb friction cone, while the dashed gray lines denote the contracted cone associated with the safety margin $r_{s}=0.10$. For both configurations, the optimized contact forces remain within the contracted cones, confirming that the load-supporting tangential forces are feasible for the generated squeezing forces.

The contact force distribution is physically consistent with the estimated CoM offset. In both configurations, the tangential force component at the right handle is larger than that at the left handle. This occurs because the CoM is offset along the $+x_{\mathrm{local}}$ -direction (see Table 2), requiring an asymmetric distribution of load-supporting tangential forces to satisfy moment equilibrium. At the same time, the normal force components at the two contacts act as opposing squeezing forces with equal magnitude and opposite direction, consistent with force equilibrium in the contact-normal direction.

The corresponding tangential force–torsional moment pairs are shown in Fig. 11(b). The horizontal axis represents the tangential force magnitude $\|\mathbf{f}_{t}\|$, and the vertical axis represents the torsional moment $\tau_{n}$ about the contact normal. The solid curves denote the nominal friction limit surfaces, while the dashed curves denote the contracted feasible regions with safety margin $r_{s}=0.10$. The optimized wrench samples lie inside the contracted feasible regions for both configurations, indicating that the SOCP solution satisfies the friction limit surface constraints with the prescribed safety margin.

The smaller torsional moment magnitude at the right handle is also consistent with the friction-limit-surface coupling, as defined in (41). Since the right handle carries a larger tangential force, it uses a larger portion of the available friction budget. As a result, the admissible torsional moment about the contact normal is reduced at that contact, and the optimized solution assigns a smaller $|\tau_{n}|$ to the right handle. This illustrates why tangential force and torsional moment cannot be selected independently when the friction limit surface is considered.

Across Configuration 1 and Configuration 2, the sign of the torsional moment at each handle reverses. This is consistent with the change in CoM offset along the $y_{\mathrm{local}}$ -direction: Configuration 1 has a positive $y_{\mathrm{local}}$ offset, whereas Configuration 2 has a negative $y_{\mathrm{local}}$ offset. The reversal indicates that the required torsional compensation changes direction when the object mass distribution is changed.

Differences between the simulation and real-world results are mainly due to the mass and CoM estimation errors from Phase II. Nevertheless, the optimized contact wrenches remain inside the friction-feasible regions and follow the expected force–torque distribution for each CoM configuration. The optimized wrench values are then used as the desired contact wrenches for the execution stage in Sec. 3.5.

### 3.5 Execution Result

![Refer to caption](https://arxiv.org/html/2605.22021v1/figures/fig_Phase4_trajectory_photos.png)

(a)

We evaluate the full pipeline on the dual-arm system described in Sec. 3.1. Fig. 12(a) shows representative execution sequences for the two CoM configurations, and Fig. 12(b) shows the corresponding reconstructed 3D box trajectories. As discussed in Sec. 3.2, Configuration 1 uses the refined trajectory from Phase I, whereas Configuration 2 does not require trajectory refinement. In both cases, execution uses the optimized contact wrench from Phase III based on the online CoM estimates from Phase II. Each configuration is repeated three times, and all runs show consistent behavior.

### 3.6 Ablation Study

Table 3: Ablation study evaluating the contribution of each phase in the proposed pipeline.

| Method Variant | Collision | Orientation Deviation | Wrench Quality | Notes |
| --- | --- | --- | --- | --- |
| Full pipeline | No | No | Optimal | Uses Phase I-III |
| w/o Phase I | Yes | – | – | Trajectory may collide with environment |
| w/o Phase II | No | Yes (tilted) | Suboptimal | Incorrect CoM estimate leads to wrong wrench |
| w/o Phase III | No | Yes (tilted) | Inefficient, unbalanced, excessive squeezing | No optimal wrench |

An ablation study is conducted to evaluate the contribution of each phase in the proposed pipeline, as summarized in Table 3. The full pipeline achieves collision-free manipulation while maintaining the desired object orientation by combining trajectory refinement (Phase I), online CoM estimation (Phase II), and contact wrench optimization (Phase III).

When Phase I is removed, the system relies on the nominal trajectory, which may violate environmental constraints and result in collisions in confined settings, as also can be observed in Fig. 8.

When Phase II is removed, the system relies on a nominal assumption that the center of mass is located at the geometric center of the object, causing the computed contact wrench to be inconsistent with the true mass distribution. As a result, the applied forces and torques do not properly balance the object, leading to noticeable orientation deviation during manipulation.

When Phase III is removed, no optimal contact wrench is available, and the system instead relies on a naive symmetric force assignment, where both arms apply equal normal forces, share the tangential load equally, and no contact moment is commanded. As a result, the forces and torques are not properly adapted to the object’s mass distribution. This leads to inefficient and unbalanced force application, often requiring excessive squeezing to maintain the grasp, and may still result in tilted motion due to the lack of appropriate moment compensation.

These results highlight that each phase plays a distinct and complementary role: Phase I ensures collision-free motion, Phase II provides accurate object parameter estimation, and Phase III enables physically consistent and efficient force execution.

## 4 Discussion

The proposed framework combines online CoM estimation with friction-constrained wrench optimization to address dual-arm box handling under frictional contact, where improper regulation of interaction forces can lead to slip, object drop, orientation deviation, or excessive squeezing.

##### Slip avoidance and reduced squeezing as two facets of optimization formulation

The experiments achieved slip-free and drop-free object handling under both CoM configurations without exhibiting excessive squeezing. In conventional approaches, these two objectives are often in tension: a conservatively large normal force prevents slip and drop but produces excessive squeezing, while a smaller normal force reduces squeezing at the risk of losing the grasp. The proposed framework resolves this tension by treating the friction limit surface as a hard constraint, while the cost function minimizes contact effort. Online CoM estimation in Phase II makes this minimum-effort solution tractable to compute, and the Phase III SOCP automatically selects the solution with minimum contact effort within the friction-feasible region. Slip and drop avoidance arise from the hard satisfaction of the friction constraint, whereas reduced squeezing arises from the minimization of contact effort; the two are realized jointly by the same optimization problem rather than balanced as independent objectives.

##### Influence of the cost function on the force-torque distribution

The force-torque distribution in Phase III is strongly influenced by the contact model. In the current formulation, the tangential components of the contact moment are constrained to be zero, as imposed by (33). Therefore, moment balance due to an offset CoM is achieved mainly through differences in tangential forces between the two contacts: the tangential force at one handle becomes larger than at the other, so that the object remains in equilibrium. Under this contact model, the role of $l_{c}$ in (44) is limited, as the tangential moment components are fixed to zero and the torsional moment is largely determined by the equilibrium requirement.

If the constraint in (33) were removed, tangential contact moments would also become admissible. The optimizer could then distribute the required moment balance between force asymmetry and contact torques, with the tradeoff influenced by the moment weighting through $l_{c}$. Both cases satisfy the same equilibrium constraints, but correspond to different contact models and force–torque distributions.

##### Simulation-to-Real Setup Consistency in Phase I

Phase I trajectory refinement is performed in simulation, and its effectiveness depends on how closely the simulated environment matches the real-world setup. In practice, even small differences in object geometry, contact properties, or shelf configuration can lead to residual contact during execution. In addition, the current formulation assumes that the object geometry (e.g., box dimensions) is known beforehand. In more general settings, this information would need to be obtained through perception before trajectory refinement can be applied.

##### Systematic Error in CoM Estimation in Phase II

Experimental results show a consistent bias in the estimated center of mass. These errors are attributed to sensor noise, robot kinematic inaccuracies, and, most importantly, modeling mismatch arising from the handle contact locations not being perfectly aligned with the geometric center of the box as assumed in the model [^28]. In practice, such deviations introduce moment offsets that directly affect the estimation, leading to systematic rather than random error.

Overall, the results indicate that the framework can successfully execute stable lifting with online estimation of unknown object properties, while highlighting key areas for improvement.

## 5 Conclusion

This work presented a friction-aware dual-arm box-handling framework for objects with unknown mass and center of mass. The approach combines offline trajectory refinement, online parameter estimation, and optimization-based contact wrench computation under friction constraints. The optimized wrenches are executed through impedance control on a real dual-arm robotic system.

Experiments show that the proposed method lifts objects under different CoM configurations while avoiding slip, object drop, and excessive squeezing. The results demonstrate the importance of online CoM estimation and explicit force–torque optimization for efficient frictional manipulation.

Future work will focus on improving robustness to modeling mismatch, integrating perception for object geometry estimation, and extending the framework to dynamic manipulation tasks and richer contact interactions.

## Appendix A Safety Margin for Friction Constraints

![Refer to caption](https://arxiv.org/html/2605.22021v1/x11.png)

(a)

We introduce a normalized safety margin $r_{s}\in[0,1)$ by contracting the nominal admissible friction set. Let $\mathcal{F}$ denote a nominal feasible friction set, such as the friction cone $\mathcal{C}$ or the friction limit surface $\mathcal{L}$. Let $\mathbf{w}$ denote a contact force or wrench vector.

The contracted feasible set is defined as

$$
\mathcal{F}_{r_{s}}=\mathcal{F}\ominus\mathbb{B}_{r_{s}},
$$

where $\ominus$ denotes the Pontryagin difference, also known as the Minkowski set difference [^8] [^23], and

$$
\mathbb{B}_{r_{s}}=\left\{\Delta\bar{\mathbf{w}}\mid\|\Delta\bar{\mathbf{w}}\|_{2}\leq r_{s}\right\}
$$

is a closed ball of radius $r_{s}$ in normalized force/wrench space. Here, $\Delta\bar{\mathbf{w}}$ denotes a normalized perturbation of the selected contact force or wrench $\bar{\mathbf{w}}$. As illustrated in Fig. 13, the green ball represents the perturbation set $\mathbb{B}_{r_{s}}$, and the center of the ball corresponds to the selected contact force or wrench. Equivalently,

$$
\mathcal{F}_{r_{s}}=\left\{\bar{\mathbf{w}}\mid\bar{\mathbf{w}}+\mathbb{B}_{r_{s}}\subseteq\mathcal{F}\right\}.
$$

Thus, the selected force or wrench must remain inside the nominal friction set even after any normalized perturbation of size $r_{s}$.

For the friction cone, the contraction is implemented as

$$
\|\mathbf{f}_{t}\|\leq(1-r_{s})\mu(-f_{n}),\qquad f_{n}\leq 0,
$$

and for the friction limit surface, the contraction is implemented as

$$
\left(\frac{\|\mathbf{f}_{t}\|}{-\mu f_{n}}\right)^{2}+\left(\frac{\tau_{n}}{\tau_{n}^{\max}}\right)^{2}\leq(1-r_{s})^{2},\qquad f_{n}\leq 0,
$$

where $f_{n}$ is the signed normal force, $\mathbf{f}_{t}$ is the tangential force, $\mu$ is the friction coefficient, $\tau_{n}$ is the torsional moment about the contact normal, and compression corresponds to $f_{n}\leq 0$.

When $r_{s}=0$, the nominal friction constraints are recovered. Increasing $r_{s}$ shrinks both admissible sets and therefore increases the friction safety margin.

## Appendix B Derivation of Torsional Friction Limit

### B.1 General derivation

Following [^12], consider a rigid contact with a fixed contact area $A$ under pure twist about the contact normal. Let $p(\mathbf{r})$ denote the normal pressure distribution over the patch, where $\mathbf{r}\in\mathbb{R}^{2}$ is the in-plane position vector measured from the center of rotation. Under Coulomb friction, the maximum admissible torsional moment about the normal direction can be written as

$$
\tau_{n,\max}=\mu\int_{A}p(\mathbf{r})\,\|\mathbf{r}\|\,dA,
$$

where $\mu$ is the friction coefficient. Defining the normal force magnitude as

$$
f_{n}\triangleq\int_{A}p(\mathbf{r})\,dA,
$$

we introduce the *effective contact radius*

$$
R_{\mathrm{eff}}\triangleq\frac{1}{f_{n}}\int_{A}p(\mathbf{r})\,\|\mathbf{r}\|\,dA,
$$

so that (56) becomes the compact expression

$$
\tau_{n,\max}=\mu f_{n}R_{\mathrm{eff}}.
$$

The value of $R_{\mathrm{eff}}$ depends on the contact geometry and the assumed pressure distribution $p(\mathbf{r})$.

### B.2 Evaluation for uniform pressure over a rectangular patch

Assume the contact patch is a rectangle of size $a\times b$ (in meters), centered at the origin, with a uniform pressure distribution

$$
p(\mathbf{r})=\frac{f_{n}}{ab},\qquad(x,y)\in\left[-\frac{a}{2},\frac{a}{2}\right]\times\left[-\frac{b}{2},\frac{b}{2}\right].
$$

Substituting into (58) yields

$$
R_{\mathrm{eff}}=\frac{1}{ab}\int_{-a/2}^{a/2}\int_{-b/2}^{b/2}\sqrt{x^{2}+y^{2}}\,dy\,dx.
$$

Evaluating (61) gives the closed-form expression

$$
R_{\mathrm{eff}}=\frac{\sqrt{a^{2}+b^{2}}}{6}+\frac{a^{2}}{12b}\,\mathrm{asinh}\!\left(\frac{b}{a}\right)+\frac{b^{2}}{12a}\,\mathrm{asinh}\!\left(\frac{a}{b}\right),
$$

where $\mathrm{asinh}(\cdot)$ denotes the inverse hyperbolic sine.

For the contact geometry used in this work, with $a=0.07$ m and $b=0.10$ m, (62) yields $R_{\mathrm{eff}}\approx 0.0328~\text{m}.$

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used OpenAI’s ChatGPT in order to assist with language refinement and readability of selected manuscript sections. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

[^1]: A. Bicchi and V. Kumar (2000) Robotic grasping and contact: a review. In Proceedings 2000 ICRA. Millennium conference. IEEE international conference on robotics and automation. Symposia proceedings (Cat. No. 00CH37065), Vol. 1, pp. 348–353. Cited by: §1, §1.

[^2]: J. Bohg, A. Morales, T. Asfour, and D. Kragic (2013) Data-driven grasp synthesis—a survey. IEEE Transactions on robotics 30 (2), pp. 289–309. Cited by: §1.

[^3]: S. Boyd and L. Vandenberghe (2004) Convex optimization. Cambridge university press. Cited by: §2.3.5.

[^4]: F. Caccavale, P. Chiacchio, A. Marino, and L. Villani (2008) Six-dof impedance control of dual-arm cooperative manipulators. IEEE/ASME Transactions On Mechatronics 13 (5), pp. 576–586. Cited by: §1.

[^5]: D. Campolo and F. Cardin (2025) A geometric framework for quasi-static manipulation of a network of elastically connected rigid bodies. Applied Mathematical Modelling 143, pp. 116003. Cited by: §2.2.

[^6]: H. Cao, H. Fang, W. Liu, and C. Lu (2021) Suctionnet-1billion: a large-scale benchmark for suction grasping. IEEE Robotics and Automation Letters 6 (4), pp. 8718–8725. Cited by: §1.

[^7]: N. Correll, K. E. Bekris, D. Berenson, O. Brock, A. Causo, K. Hauser, K. Okada, A. Rodriguez, J. M. Romano, and P. R. Wurman (2016) Analysis and observations from the first amazon picking challenge. IEEE Transactions on Automation Science and Engineering 15 (1), pp. 172–188. Cited by: §1, §1.

[^8]: A. Cotorruelo, I. Kolmanovsky, and E. Garone (2022) A sum-of-squares-based procedure to approximate the pontryagin difference of basic semi-algebraic sets. Automatica 135, pp. 109783. External Links: [Document](https://dx.doi.org/10.1016/j.automatica.2021.109783) Cited by: Appendix A.

[^9]: A. Dutta, E. Burdet, and M. Kaboli (2025) Predictive visuo-tactile interactive perception framework for object properties inference. IEEE Transactions on Robotics. Cited by: §1.

[^10]: S. Erhart, D. Sieber, and S. Hirche (2013) An impedance-based control architecture for multi-robot cooperative dual-arm mobile manipulation. In 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 315–322. Cited by: §1.

[^11]: M. Guo, D. V. Gealy, J. Liang, J. Mahler, A. Goncalves, S. McKinley, J. A. Ojea, and K. Goldberg (2017) Design of parallel-jaw gripper tip surfaces for robust grasping. In 2017 IEEE international conference on robotics and automation (ICRA), pp. 2831–2838. Cited by: §1.

[^12]: R. D. Howe and M. R. Cutkosky (1996) Practical force-motion models for sliding manipulation. The International Journal of Robotics Research 15 (6), pp. 557–572. Cited by: §B.1, §2.3.4, §2.3.4.

[^13]: A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal (2013) Dynamical movement primitives: learning attractor models for motor behaviors. Neural computation 25 (2), pp. 328–373. Cited by: §2.1.

[^14]: J. Kerr and B. Roth (1986) Analysis of multifingered hands. The International Journal of Robotics Research 4 (4), pp. 3–17. Cited by: §1.

[^15]: K. Kleeberger, R. Bormann, W. Kraus, and M. F. Huber (2020) A survey on learning-based robotic grasping. Current Robotics Reports 1 (4), pp. 239–249. Cited by: §1.

[^16]: L. Koutras, I. Ntoliou, and Z. Doulgeri (2023) Towards passivity based nonprehensile bimanual manipulation of large objects. In 2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids), pp. 1–8. Cited by: §1.

[^17]: D. Lee, D. V. Dimarogonas, and H. J. Kim (2026) Switching control of underactuated multichannel systems with input constraints for cooperative manipulation. IEEE Transactions on Control Systems Technology. Cited by: §1.

[^18]: Q. Y. Lee, S. R. Kulkarni, K. I. Wong, L. Yang, B. Noronha, Y. Wee, and D. Campolo (2025) Generalizing robot trajectories from single-context human demonstrations: a probabilistic approach. External Links: 2503.05619, [Link](https://arxiv.org/abs/2503.05619) Cited by: §2.1, §2, §3.2.

[^19]: S. Levine, P. Pastor, A. Krizhevsky, J. Ibarz, and D. Quillen (2018) Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International journal of robotics research 37 (4-5), pp. 421–436. Cited by: §1.

[^20]: J. Mahler, M. Matl, X. Liu, A. Li, D. Gealy, and K. Goldberg (2018) Dex-net 3.0: computing robust vacuum suction grasp targets in point clouds using a new analytic model and deep learning. In 2018 IEEE International Conference on robotics and automation (ICRA), pp. 5620–5627. Cited by: §1.

[^21]: R. M. Murray, Z. Li, and S. S. Sastry (2017) A mathematical introduction to robotic manipulation. CRC press. Cited by: §1.

[^22]: M. Saveriano, F. J. Abu-Dakka, A. Kramberger, and L. Peternel (2023) Dynamic movement primitives in robotics: a tutorial survey. The International Journal of Robotics Research 42 (13), pp. 1133–1184. Cited by: §2.1.

[^23]: R. Schneider (2013) Convex bodies: the brunn–minkowski theory. 2 edition, Encyclopedia of Mathematics and its Applications, Vol. 151, Cambridge University Press. Cited by: Appendix A.

[^24]: S. Schneider and R. H. Cannon (1989) Object impedance control for cooperative manipulation: theory and experimental results. In Proceedings, 1989 international conference on robotics and automation, pp. 1076–1083. Cited by: §1.

[^25]: C. Smith, Y. Karayiannidis, L. Nalpantidis, X. Gratal, P. Qi, D. V. Dimarogonas, and D. Kragic (2012) Dual arm manipulation—a survey. Robotics and Autonomous systems 60 (10), pp. 1340–1353. Cited by: §1.

[^26]: F. Stulp and O. Sigaud (2012) Path integral policy improvement with covariance matrix adaptation. arXiv preprint arXiv:1206.4621. Cited by: §2.1.

[^27]: R. Tedrake and the Drake Development Team (2019) Drake: model-based design and verification for robotics. External Links: [Link](https://drake.mit.edu/) Cited by: §2.1, §3.2.

[^28]: S. H. Turlapati, V. P. Nguyen, J. Gurnani, M. Z. Bin Ariffin, S. Kana, A. H. Yee Wong, B. S. Han, and D. Campolo (2024) Identification of intrinsic friction and torque ripple for a robotic joint with integrated torque sensors with application to wheel-bearing characterization. Sensors 24 (23). External Links: [Link](https://www.mdpi.com/1424-8220/24/23/7465), ISSN 1424-8220, [Document](https://dx.doi.org/10.3390/s24237465) Cited by: §2.4, §4.

[^29]: A. Wu and D. Kruse (2025) In the wild ungraspable object picking with bimanual nonprehensile manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 5669–5676. Cited by: §1.

[^30]: N. Xydas and I. Kao (1999) Modeling of contact mechanics and friction limit surfaces for soft fingers in robotics, with experimental results. The International Journal of Robotics Research 18 (9), pp. 941–950. Cited by: §2.3.4, §2.3.4.

[^31]: T. Yoshikawa and X. Zheng (1993) Coordinated dynamic hybrid position/force control for multiple robot manipulators handling one constrained object. The International journal of robotics research 12 (3), pp. 219–230. Cited by: §1.