# 2026-04-19 Note: force predictor inspired by facial motion prediction paper

## Context

This note records a project direction inspired by the paper `202430182_en.pdf`.

## Idea

- The paper uses a prediction module for future facial motion and feeds that prediction into imitation learning.
- By analogy, the plan here is to build a force state transition model, or more generally a force predictor.
- The intended role of this predictor is to model how force-related state evolves over time and to use that predictive signal inside the downstream policy or control framework.

## Interpretation

- The paper is not directly about hidden physical properties, but it provides a useful structural reference:
  predict a task-relevant future variable first, then feed that prediction into imitation learning.
- In this project, the analogous target variable would be force or force-related state rather than facial landmarks.

## Status

- This is currently a planning note, not yet a finalized implementation design.
