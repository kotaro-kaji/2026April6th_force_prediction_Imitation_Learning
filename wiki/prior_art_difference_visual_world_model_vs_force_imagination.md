# Prior-Art Difference: Visual World Models vs Force Imagination

## Core Claim

The main difference from visual world-model-based imitation learning is not merely that the proposed method predicts force.

The sharper difference is that this project targets **hidden internal physical properties whose task-relevant effects are not always sufficiently exposed as visual state changes**.
Visual world models can adapt when the relevant physical error becomes visible in predicted images, but they are less clearly sufficient when success depends on contact force, compliance, friction, deformation, or material response that is only weakly or unreliably visible.

## Existing Direction in Prior Work

A common recent structure is:

- share an encoder between a world model and an imitation policy,
- attach separate prediction and action heads,
- predict future camera images or equivalent visual observations,
- use prediction error to update the shared encoder or policy-related representation online.

This direction already makes visual prediction and imitation learning mutually useful.
If the world prediction error captures the reason an action failed, online updating can implicitly correct a representation used by the policy.

## Why PushT Alone Is Not Enough

PushT-like pushing tasks are a useful controlled benchmark, but they may not be enough to establish the unique value of force imagination.

In pushing, many hidden physical differences eventually appear as visual motion differences.
For example, if an object rotates differently because the center of mass is different, predicting the future image already forces the model to infer something related to the center of mass.
If an object barely moves under compliant pushing, image prediction error can implicitly push the representation toward friction or mass-like information.

Therefore, a visual world model may be able to solve a hidden-physics PushT variant in principle.
Even if no prior paper has explicitly evaluated internal physical-property variation as the main subject, a reviewer could reasonably argue that the problem may already be solvable by existing visual prediction and online adaptation methods.

The safe interpretation is:

- PushT can test whether force imagination helps in a controlled simulation.
- PushT alone should not be the sole evidence for the claim that force prediction is necessary for hidden-physics imitation.
- The claim should avoid saying that image-prediction world models cannot solve PushT unless an empirical baseline confirms that.

## Why Wiping Is the Stronger Main Task

Wiping with visually similar sponges of different softness is a stronger task for this research question.

In wiping, the relevant hidden property is not only object motion.
Sponge softness changes contact area, deformation, required normal force, friction, and dirt-removal efficiency.
These variables can determine task success even when the visual observation does not reliably show the sponge's deformation or the true contact force.

This makes the force-imagination claim more defensible:

- the important state is contact-rich and material-dependent,
- the causal variable is not fully visible from ordinary camera views,
- success requires choosing actions based on the object's internal material response,
- force prediction provides a more direct learning signal than image prediction.

## Recommended Experimental Story

The strongest experimental story is a two-stage one.

First, use hidden-physics PushT in simulation as a controlled starting point.
This evaluates whether the method can exploit force imagination and internal physical-property representations in a clean, reproducible setting.

Second, use wiping with sponges of different softness as the main evidence for the paper's unique contribution.
This tests the regime where visual world-model prediction is less likely to contain the full task-relevant signal, and where force imagination is more naturally tied to the missing hidden property.

The best result would be:

- high success on PushT variants with different internal physical properties,
- high success on wiping with sponges of different softness,
- ablations showing that visual prediction alone is weaker, especially on wiping,
- evidence that the learned representation separates or tracks hidden physical/material differences.

## Positioning Against Prior Work

The contribution should not be framed as simply "handling object changes" or "online updating a world model," because prior work already addresses those topics.

A clearer positioning is:

> Prior visual world-model methods adapt policies by predicting observable future states, but they have not isolated settings where the decisive variation lies in visually ambiguous internal material properties. This project studies whether force imagination can provide the missing supervision for learning and adapting an internal physical-property representation for imitation learning.

This framing separates the work from visual OOD, camera-shift robustness, and ordinary object-swap adaptation.
The central question becomes whether force imagination is useful when the hidden property matters more than visible appearance or visible object motion.

## Source

- Raw user note: [2026-05-21_prior_art_difference_world_model_vs_force_imagination.md](../raw/2026-05-21_prior_art_difference_world_model_vs_force_imagination.md)
- Related pages:
  - [Proposed Method: Force-Imagination-Based Online-Correctable Imitation Learning](proposed_method_force_imagination_online_correctable_imitation.md)
  - [AdaWorldPolicy vs Same-Visual Different-Physics Gap](adaworldpolicy_vs_same_visual_different_physics_gap.md)
  - [Task Candidates for Hidden-Physics Imitation Learning](task_candidates_for_hidden_com_material_latent.md)
