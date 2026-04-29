# AdaWorldPolicy vs Same-Visual Different-Physics Gap

## Question

Does AdaWorldPolicy already cover the core research question of adapting to objects that look the same but have different physical properties, using force prediction error?

## Short Answer

Not cleanly.

AdaWorldPolicy is mainly positioned and evaluated as a method for handling **visual OOD shifts** and broader deployment shifts.
It does use force input and a force predictor, but it does **not** cleanly isolate the question:

- same visual appearance
- different hidden physical properties
- adaptation driven by force prediction error alone

This leaves room for a more controlled study focused on hidden physics under matched appearance.

More importantly, there is a stronger potential weakness for the same-looking hidden-physics setting:

- if multiple objects share the same visual appearance
- but differ only in hidden physical properties
- the model does not appear to maintain a long-term explicit object-physics identity
- and may therefore need to keep re-identifying the object from only the most recent force observations

In such cases, the learning problem itself can become ambiguous.

## Why This Matters

If object appearance changes together with physics, then improved performance can come from visual recognition of the changed object, rather than from true inference of hidden physical properties.

For this project, the sharper question is:

- when objects look the same,
- but internal mass, center of mass, stiffness, friction, or compliance differ,
- how far can adaptation go using force-related prediction error?

That question is not the main emphasis of AdaWorldPolicy’s experiments.

There is also a structural reason this matters.

In the real-world setting, AdaWorldPolicy uses only a very short observation history, and in the reported configuration the history length is `1`.
That means that when appearance is matched across objects, the model may have to infer "which physical object this is" repeatedly from immediate force feedback, instead of carrying a persistent hidden-physics representation over a longer horizon.

For tasks such as manipulating a box whose inside is not visible, this can create confusion during learning:

- visually identical observations correspond to different latent physics
- but the model has no explicit long-term object-identity or material latent
- and only a very short force history is available to disambiguate them

This is a stronger weakness than simply saying that the paper focuses mainly on visual OOD.

## Practical Implication

This means AdaWorldPolicy is relevant as a strong reference for:

- unified world-model and policy learning
- force-aware prediction
- online adaptation at deployment

But it does **not** remove the value of a project that is explicitly designed around:

- same-appearance objects
- hidden physical variation
- persistent identification of hidden physics over time
- a controlled evaluation of adaptation from force prediction error

## Research Value For This Project

So the research value here remains:

- isolate hidden-physics adaptation from visual OOD adaptation
- test whether force prediction error alone can identify same-looking but physically different objects
- test whether a persistent latent or longer-horizon memory is needed to avoid confusion across visually identical but physically different objects
- make the evaluation cleaner than broad mixed-shift settings

## Source

- User assessment on 2026-04-28:
  - AdaWorldPolicy mainly studies how to handle visual OOD shifts, and its experiments are centered on that.
  - Its exploration of how far one can adapt to same-looking objects with different physical properties using force prediction error alone is still insufficient.
  - A stronger weakness is that when training includes visually identical objects that differ only in hidden physical properties, the model lacks a long-term explicit signal for which physical object it is currently handling.
  - In that case, because it must keep identifying the object from only the most recent one-step or short-horizon force observations, learning may become confused.
  - This is especially relevant for tasks such as manipulating boxes whose internal state is not visually observable.
  - Therefore there is still research value in a project that isolates this question more directly.

## Original User Wording

For this note, the original wording from the user is likely easier for the user themself to understand than a cleaned-up summary, so it is preserved below.

> もっと強い弱点は、学習時に視覚的な見た目が同じで、物性のみが異なるオブジェクトに対するデモデータがあるとき、world modelにおいて、どの物性のオブジェクトを扱っているかの長期的な入力がないため（正確には直近1 stepないし、数ステップのforceのみから識別し続ける必要があるため）、すごいworkだけど、純粋に見た目が同じで物性だけが異なるオブジェクトの学習の混乱を招く可能性がある。それは例えば、中身が見えない箱に対するタスクなどだ。
