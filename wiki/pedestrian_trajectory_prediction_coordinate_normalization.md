# Pedestrian Trajectory Prediction Coordinate Normalization

## Source

Zamboni et al., “Pedestrian Trajectory Prediction with Convolutional Neural Networks,” Pattern Recognition 121, 2022, arXiv:2010.05796.

Local source:

- [Pedestrian Trajectory Prediction with Convolutional Neural Networks](<../raw/Pedestrian Trajectory Prediction with Convolutional Neural Networks.pdf>)

## Why It Matters Here

This paper provides a controlled comparison that is unusually close to the current Absolute-versus-Residual question. It trains both convolutional and recurrent trajectory predictors with four coordinate representations:

1. raw absolute scene coordinates;
2. coordinates relative to the first observed position;
3. coordinates relative to the latest observed position;
4. consecutive relative displacements, interpreted as velocities.

The third representation is equivalent to direct multi-horizon residual prediction from the latest observation:

```text
target[k] = position[t+k] - position[t]
prediction_absolute[k] = position[t] + prediction_relative[k]
```

It is not recursive integration of consecutive predicted velocities.

## Reported Results

The paper evaluates 8 observed positions and 12 future positions on the ETH/UCY benchmark. The table below records the average ADE/FDE across five scenes.

| Model | Raw absolute | First-observation origin | Latest-observation origin | Consecutive relative displacement |
|---|---:|---:|---:|---:|
| Conv1D | 2.684 / 3.288 | 0.542 / 1.084 | **0.533** / 1.116 | 0.550 / 1.109 |
| LSTM | 3.302 / 4.152 | 0.559 / 1.136 | **0.535 / 1.111** | 0.558 / 1.149 |

The authors select the latest-observation origin because it gives the lowest average displacement error for both architectures. They interpret the latest observation as the most informative reference point for the future trajectory.

For Conv1D, the first-observation representation has a slightly lower average final displacement error than the latest-observation representation. The strongest supported claim is therefore about average trajectory error, not universal superiority on every metric.

## Correct Interpretation

This paper supports using the latest observation as the origin of a deterministic future-state head. It is more directly relevant than merely observing that some dynamics papers predict displacement.

However, the very large failure of raw absolute coordinates should not be transferred numerically to the robot project:

- each pedestrian scene uses an arbitrary global origin;
- raw coordinate ranges therefore differ greatly across scenes;
- the robot workspace may use a consistent normalized coordinate frame;
- pedestrian trajectories do not include action conditioning, force, wrench, `z_phys`, or image latents.

Thus, the paper supports the inductive bias and coordinate choice, but it does not predict the size of the gain in the robot world model.

## Relation to the Proposed Interface

The paper itself transforms both inputs and targets into the selected relative coordinate system. The current project can preserve a cleaner external contract:

```text
teacher target: absolute future state
internal target: latest-observation-relative residual
returned value: reconstructed absolute future state
```

For a fixed latest observation and MSE, this gives the same residual-learning objective while avoiding a second public dataset and checkpoint semantic.

The paper also reinforces the recommendation to predict every future horizon relative to the same latest observation rather than accumulating one-step residuals.

## Limitations for `z_phys`

The paper evaluates trajectory error only. It does not test whether a hidden physical representation is used. A latest-observation skip can still permit the persistence solution when motion is small, so the robot project still requires:

- a latest-observation persistence baseline;
- action and `z_phys` shuffle ablations;
- same-appearance/different-physics evaluation;
- results separated by motion and contact regime.
