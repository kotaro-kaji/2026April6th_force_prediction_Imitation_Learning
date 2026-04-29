# AdaWorldPolicy Force Preprocessing

## Question

Does AdaWorldPolicy apply any explicit filtering to force signals, such as low-pass filtering, moving average smoothing, or FFT-based preprocessing?

## Short Answer

I did **not** find explicit evidence that AdaWorldPolicy applies low-pass filtering, moving-average smoothing, Butterworth filtering, or FFT/Fourier transforms to force data.

What the paper **does** explicitly describe is **quantile-based normalization** for force data in the real-world setup.
With the local raw paper text now available, this is stated more concretely as using the **1st and 99th percentiles**.

## What The Paper Explicitly Says

In the supplementary implementation details, the authors state that force data needs different preprocessing from images or relative actions because force signals do not have clear theoretical bounds and can contain large spikes or variance.

They then say that standard normalization such as min-max scaling based on absolute extremes can be unstable or compress the distribution too much.

Their reported solution is:

- use **quantile-based normalization**
- scale force data using statistical percentiles
- specifically use the **1st percentile** and **99th percentile**
- use this to reduce the effect of outliers

This is the only explicit force preprocessing statement I found in the paper text.

## What Became Clear From The Local Raw Text

The local raw source gives the exact sentence:

- force measurements can have large variance or spikes
- min-max normalization based on absolute extremes can be unstable
- they therefore use **quantile-based normalization with the 1st and 99th percentiles**

So the new concrete detail is the percentile pair itself:

- lower bound: `1st percentile`
- upper bound: `99th percentile`

## What Is Still Not Explicit

Even with the added raw text, I still did **not** find an explicit statement that they:

- clip force values to the percentile range before scaling
- clamp normalized values after scaling

So the paper now clearly specifies the percentile pair, but still does **not** clearly specify whether clipping is performed.

## What I Did Not Find

I did **not** find explicit mentions of:

- low-pass filtering
- Butterworth filtering
- moving-average filtering
- FFT or Fourier transforms
- spectrogram conversion

## Interpretation

So for AdaWorldPolicy, the force pipeline appears to be:

- force-torque readings are used as an input modality
- a force predictor models future force readings
- preprocessing is described at the level of **robust normalization**, not temporal filtering

That makes AdaWorldPolicy different from papers such as ACP or Adaptive Wiping, which explicitly mention moving-average filtering or Butterworth low-pass filtering.

## Sources

- AdaWorldPolicy arXiv HTML: https://arxiv.org/html/2602.20057v1
- Local raw source:
  - `raw/AdaWorldPolicy World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation.md`
  - Section `7.2 Data Processing`
  - explicit statement that force data uses quantile-based normalization with the `1st` and `99th` percentiles
- ArXiv HTML:
  - https://arxiv.org/html/2602.20057v1

## Date

Checked against the arXiv HTML version on 2026-04-28.
