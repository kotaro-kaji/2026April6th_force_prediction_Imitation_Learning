# Force Estimation and Latent Adaptation Prior Art

## Question

Is there prior work that overlaps with a method that learns a force estimator supervised by measured force/torque, uses estimated force or force-aware representation for imitation learning, and updates an object/material latent online from the error between estimated and measured force?

## Short Answer

There is close prior work for **force estimation from vision** and for **force-aware imitation learning**.
The closest current threat is **ForceMapping**, because it explicitly learns force-relevant visual representations with a supervised force prediction objective and uses them for visual imitation learning.

However, in the papers checked here, I did not find a complete match to the full combination:

- force estimator rather than future force predictor,
- measured force used as supervision or online feedback,
- policy does not simply consume recent measured force history,
- low-dimensional object/material latent is updated online from force-estimation error,
- the adapted latent is then used for imitation-policy conditioning.

This leaves a plausible gap if the project is framed around **force-estimation-error-driven latent adaptation**, not merely force-aware policy learning.
As of 2026-05-08, this is the current project framing.

## Strongly Related Papers

### ForceMapping

**ForceMapping: learning visual-force features from vision for soft objects manipulation** is the closest conceptual neighbor.
It learns force-aware visual representations for visual imitation learning.
The paper explicitly discusses that force can be measured by sensors or estimated from visual input through an offline-trained network, and proposes adding supervised force prediction loss to the visual representation learning objective.

Overlap:

- force is estimated from visual input,
- force supervision is used to improve representation learning,
- the learned representation is used for imitation/control,
- the motivation is force-aware manipulation without relying directly on online force sensing.

Difference:

- the focus is force-aware visual representation, especially for soft-object deformation,
- it does not appear to center on object/material latent variables,
- it does not appear to update a low-dimensional latent online using measured-vs-estimated force error,
- the main adaptation story is not hidden-physics identification at deployment.

Threat level: **High** for the "force estimator helps imitation learning" claim.
Lower threat for the "force-error-driven material latent adaptation" claim.

### Vision-Based Interaction Force Estimation

**Vision-based interaction force estimation for robot grip motion without tactile/force sensor** estimates interaction force during grasping and picking from visual and robot-side signals.
The abstract emphasizes estimating force without tactile or force/torque sensors at inference and reports estimation on seen and unseen objects.

Overlap:

- estimates interaction force without direct force sensing at inference,
- uses vision and auxiliary robot signals,
- targets robot grasping/picking.

Difference:

- primarily a force-estimation paper,
- not an imitation-learning policy-conditioning paper,
- not centered on online material-latent adaptation.

Threat level: **Medium** for force estimation, low for the full method.

### Forces for Free

**Forces for free: Vision-based contact force estimation with a compliant hand** estimates contact force by visually observing deformation of a compliant gripper.
It uses a compliant hand and vision to obtain force information as an alternative to force/torque or tactile sensors.

Overlap:

- estimates contact force from visual deformation,
- targets manipulation utility without conventional force sensors.

Difference:

- depends on a specialized compliant hand whose deformation reveals force,
- not an imitation-learning latent-adaptation framework,
- not object/material latent inference.

Threat level: **Medium** for force estimation, low for the full method.

### Use the Force, Luke

**Use the Force, Luke! Learning to Predict Physical Forces by Simulating Effects** infers contact points and physical forces from videos of humans interacting with objects.
It uses a simulator to supervise forces through their effects.

Overlap:

- infers forces from visual observations,
- learns physically meaningful force representations,
- includes generalization to novel objects.

Difference:

- focuses on human-object videos and physical force inference,
- supervision is based on simulated effects rather than measured robot force/torque during demonstrations,
- not an imitation policy with online object/material latent adaptation.

Threat level: **Medium** as a force-from-vision precedent.

### ForceMimic

**ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation** collects force-motion demonstrations and trains a force-centric imitation model with hybrid force-position execution.

Overlap:

- force-centric imitation learning,
- explicitly models time-varying wrench and pose for contact-rich manipulation.

Difference:

- it is closer to direct force imitation and hybrid force-position control,
- not primarily force estimation from vision,
- not centered on updating an object/material latent from estimation error.

Threat level: **Medium** for force-aware imitation, lower for estimator-latent adaptation.

### UMI-FT

**In-the-Wild Compliant Manipulation with UMI-FT** collects RGB, depth, pose, and finger-level wrench measurements, then trains an adaptive compliance policy that predicts position targets, grasp force, and stiffness.

Overlap:

- multimodal force-aware imitation learning,
- strong precedent for using force/torque data in contact-rich tasks such as wiping,
- directly relevant if this project uses wiping or compliant manipulation.

Difference:

- force/torque sensing is a data-collection and control signal,
- the policy predicts compliance-related control targets,
- not a force estimator with material-latent online adaptation.

Threat level: **Medium** for compliant force-aware imitation.

### AdaWorldPolicy

**AdaWorldPolicy** includes a force predictor and online adaptive learning.
Its force predictor anticipates future force/torque readings from current state and action, while AdaOL updates a small subset of network parameters using prediction errors through LoRA.

Overlap:

- force prediction,
- measured feedback is used for online adaptation,
- contact-rich manipulation and diffusion-policy-style architecture.

Difference:

- its core module is a future force predictor coupled with a world model and action expert,
- it updates network parameters rather than a deliberately interpretable low-dimensional object/material latent,
- it is not framed as replacing direct force history with a force estimator for policy conditioning.

Threat level: **High** if the project is framed as "force prediction error for online adaptation."
Lower if the project is framed as "force estimator plus latent-only material adaptation."

### IIDA

**Context is Everything: Implicit Identification for Dynamics Adaptation** infers an environment latent from context transitions `(s, a, s')` and conditions a dynamics predictor on that latent.

Overlap:

- compact latent for hidden environment/dynamics identification,
- no need for ground-truth physical parameters,
- latent-conditioned dynamics prediction.

Difference:

- not force-estimator-centered,
- not visual imitation learning,
- not specifically measured force vs estimated force as the online adaptation signal.

Threat level: **High** for the latent-identification structure, but lower for force-estimation-based imitation.

### CAVIA

**Fast Context Adaptation via Meta-Learning** introduces CAVIA, a MAML-related meta-learning method where shared network parameters are meta-trained but task-specific context parameters are adapted by gradient descent.
At test time, only the low-dimensional context parameters are updated.

Overlap:

- compact task/environment variables are adapted while shared model weights stay fixed,
- the context variables are provided as additional input to the network,
- the method gives a clean precedent for latent-only adaptation.

Difference:

- it is a general meta-learning algorithm, not a force-estimation or imitation-learning paper,
- it does not specifically use measured-vs-estimated force error,
- it does not address object/material latent learning for contact-rich manipulation.

Threat level: **High** for the generic "adapt only a compact context latent" idea, but low for the force-estimation/material-latent robotics claim.

## Most Defensible Differentiation

The safest differentiation is not "we use force" or "we estimate force."
Those are already covered by prior work.

The more defensible claim is:

- learn a force estimator supervised by measured force/torque,
- avoid giving the policy recent measured force history as an easy shortcut,
- use the mismatch between estimated and measured force to update only a compact object/material latent,
- feed the estimated force and/or adapted latent into an imitation policy,
- evaluate whether the latent captures hidden material or object physics under controlled object variation.

## Sources

- ForceMapping: https://www.tandfonline.com/doi/full/10.1080/01691864.2025.2483929
- Vision-based interaction force estimation for robot grip motion without tactile/force sensor: https://doi.org/10.1016/j.eswa.2022.118441
- Forces for free: Vision-based contact force estimation with a compliant hand: https://pubmed.ncbi.nlm.nih.gov/40561044/
- Use the Force, Luke!: https://doi.org/10.1109/CVPR42600.2020.00030
- ForceMimic: https://forcemimic.github.io/
- UMI-FT: https://umi-ft.github.io/
- AdaWorldPolicy: https://adaworldpolicy.github.io/
- IIDA local note: `wiki/iida_context_is_everything_implicit_identification.md`
- CAVIA local note: `wiki/cavia_fast_context_adaptation_meta_learning.md`
