# 2026-04-24 Note: FiLM parameterization ideas for force predictor and diffusion policy

## Background

これまで「物性 latent を PB のような明示表現として持ち、それを policy に入力する」という整理で考えていたが、
物性をベクトルそのものとして持つよりも、FiLM の modulation parameter として持たせる方が、最近の設計としては自然かもしれないという発想が出てきた。

ここでの関心は、force predictor と Diffusion Policy (DP) の両方に対して、
hidden な物性をどのように注入するのがよいか、という点にある。

## Idea 1

DP と force predictor は最終ヘッドだけを分け、
その前段の Transformer encoder 部分は共有する。
そして、物性は共有 encoder 内で効く FiLM parameter として持たせる。

イメージとしては、

- 共有 Transformer encoder が observation/history を表現化する
- 物性に対応する FiLM が encoder の中間特徴を modulation する
- その上に DP 用ヘッドと force predictor 用ヘッドを載せる

という構造である。

この案の良さは、force prediction に必要な hidden physics の表現と、
policy に必要な表現を強く結びつけられる可能性がある点にある。
一方で、表現共有が強すぎると、force prediction に最適な表現と policy に最適な表現が衝突する可能性もある。

## Idea 2

Transformer encoder 部分は共有しない。
つまり、

- force predictor 用の Transformer encoder
- DP 用の Transformer encoder

をそれぞれ別に持つ。

ただし両者に同じ構造の FiLM 層を設け、その FiLM 層の parameter を共有する。

この案では、
encoder 自体は各タスクに合わせて別々に最適化できる一方で、
「hidden な物性がどういう modulation として効くべきか」という部分だけを共有資産として扱う。

## Main comparison point

比較の本質は、

- 物性を「表現空間そのもの」として共有したいのか
- 物性を「各ネットワークの特徴をどう変調するか」という作用の形で共有したいのか

の違いにある。

Idea 1 は前者に近く、Idea 2 は後者に近い。

## Current intuition

今の直感としては、研究の主眼が
「force prediction を通じて hidden physics を学び、それが policy 改善につながる」
ことの実証にあるなら、まずは Idea 1 の方が筋がよい可能性がある。

理由は、force predictor 側で学ばれた hidden physics に関する表現が、
本当に DP 側でも使われていると言いやすいためである。

一方で、DP と force prediction で必要な表現がかなり異なるなら、
Idea 2 のように encoder を分けて FiLM だけ共有した方が、最適化は安定するかもしれない。

## Questions to revisit later

- force predictor と DP で入力系列の統計や必要な時系列表現はどれくらい近いか
- encoder を共有したときに negative transfer がどれくらい起きるか
- FiLM をどこに入れるのがよいか
- FiLM parameter 自体を test-time adaptation 可能にするか
- FiLM 共有だけで「hidden physics を共有している」と十分に主張できるか
