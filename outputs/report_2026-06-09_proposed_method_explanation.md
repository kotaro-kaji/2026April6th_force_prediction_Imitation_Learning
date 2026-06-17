# 2026-06-09 提案手法・研究説明

## 概要

本レポートは、現在の wiki に蓄積されている研究メモをもとに、提案手法と研究の位置付けを整理したものです。

中心となるページは以下です。

- `wiki/proposed_method_force_imagination_online_correctable_imitation.md`
- `wiki/research_theme_1_force_estimation_material_imitation.md`
- `wiki/preventing_hidden_context_latent_ignored.md`
- `wiki/prior_art_difference_visual_world_model_vs_force_imagination.md`
- `wiki/task_search_beyond_sponge_wiping_for_force_imagination.md`

## 研究の中心

本研究は、接触リッチな模倣学習において、見た目からは分からない物体・材料の物性を、力の推定誤差を使って同定し、その同定結果を方策に反映する手法である。

単に「力を入力に使う模倣学習」ではない。より正確には、測定された力・トルクを教師信号として force imagination model を学習し、実行時には推定力と測定力のズレから低次元の物性 latent だけをオンライン更新し、その latent または推定力で imitation policy を条件付ける研究である。

ここで重要なのは、方策に直近の実測 force history をそのまま与えて解かせるのではなく、力のズレを使って「この物体・材料はどのような hidden physics を持つか」を表す compact latent を更新する点である。

## 問題意識

通常の vision-based imitation learning は、見えている状態から専門家動作を再現する。しかし、接触リッチ操作では、同じ見た目でも必要な動作が変わることがある。

例としては以下がある。

- 同じ外形だが重心が違う物体
- 同じ見た目だが摩擦が違う物体
- 同じスポンジに見えるが硬さ・柔らかさが違う道具
- 同じブラシに見えるが毛の剛性が違う道具
- 同じ機構に見えるが内部抵抗が違う対象

このような場合、画像だけを見ても「どの物性の対象を扱っているか」は分からない。一方で、接触したときの力応答にはその物性が強く現れる。本研究は、この力応答を hidden physics identification の信号として使う立場を取る。

## 提案手法の構成

手法は大きく 3 つのモジュールに分かれる。

### 1. Force imagination model

入力は、視覚、proprioception、行動、物性 latent `d_i` などである。出力は現在の相互作用力、または wrench である。デモ収集時には実測 force/torque があるため、それを教師信号として「この状態・行動・latent ならどのような力が生じるべきか」を学習する。

### 2. Imitation policy

入力は通常の観測に加えて、推定された力、または更新済みの latent `d_i` である。出力は行動、action chunk、あるいは Diffusion Policy のような軌道分布である。役割は、単なる軌道再生ではなく、hidden physics に応じて行動を変えることである。

### 3. Online latent correction

実行時に force/torque センサが使える場合、実測力 `F_measured` と推定力 `F_imagined` の誤差を使って、モデル全体ではなく latent `d_i` だけを更新する。

更新式としては以下の形になる。

```text
d_i <- d_i - alpha * grad_{d_i} L(F_measured, F_imagined(o, a, d_i))
```

ここで force estimator と policy の重みは固定する。CAVIA 的に、オンライン適応は低次元 context variable だけに制限する、という発想である。

## 学習目的

最小構成では、損失は次のようになる。

```text
L = L_policy + lambda_F L_force + lambda_reg L_latent
```

`L_policy` は模倣学習の損失、`L_force` は推定力と実測力の誤差、`L_latent` は latent を安定かつコンパクトに保つ正則化である。

重要なのは、`L_force` が単なる補助損失ではないことである。この損失は、実行時に latent を修正するための信号にもなる。つまり force imagination は「表現学習の補助タスク」ではなく、「オンラインで物性 latent を合わせ込むための物理的な誤差信号」である。

## 普通の force-aware imitation との違い

既存研究には、力を使う模倣学習、力を予測するモデル、compliance-aware policy、wrench を模倣する手法がすでにある。したがって、「力を使うから新しい」という主張は弱い。

本研究の差分は、次の組み合わせにある。

- force/torque はデモ時の教師信号として使う
- 方策に実測 force history を直接与えてショートカットさせない
- 物体・材料の hidden physics を低次元 latent として持つ
- 実行時には force 推定誤差で latent だけを更新する
- 更新された latent または imagined force が方策を条件付ける

このため、研究の本質は force-aware policy ではなく、force-estimation-error-driven hidden-physics latent adaptation である。

## なぜ latent が必要か

同じ見た目の対象が複数あり、それぞれ物性だけが違う場合、直近 1 step や数 step の force だけから毎回識別し続けるのは曖昧になる。特に AdaWorldPolicy のような world-model + online adaptation 系の手法は強力だが、同じ見た目・違う物性という設定では、長期的に「今扱っている対象の物性」を保持する明示的な表現がないと混乱する可能性がある。

そのため、本手法では persistent な object/material latent を持たせる。これは「この瞬間の force を読む」だけでなく、「この対象はこういう物性を持つ」という同定結果を時間方向に保持する役割を持つ。

## タスク設計

研究を成立させるには、タスク選定が重要である。単に接触リッチならよいわけではない。条件は以下である。

- 見た目は同じ、またはほぼ同じ
- hidden physics が異なる
- 正しい行動がその hidden physics に依存する
- その違いが force には強く現れる
- 画像の未来予測だけでは十分に取り出しにくい
- 軌道再生だけでは失敗しやすい

現在の整理では、PushT は制御された simulation benchmark としては有用だが、主張の中心にはやや弱い。質量・摩擦・重心の違いは、物体の回転や滑りとして画像上に現れるため、visual world model でも latent を学習できる可能性があるからである。

より強い主実験候補は、スポンジ拭きやブラシ掃きのような surface interaction task である。たとえば、見た目が似たスポンジでも硬さが違えば、接触面積、法線力、摩擦、汚れ除去効率が変わる。しかし、その内部圧力分布や接触品質は通常のカメラ画像から安定して読めるとは限らない。この場合、force imagination の方が hidden variable とよく対応する。

## 推奨される実験ストーリー

強い実験構成は二段階である。

第一に、simulation で same-appearance hidden-physics PushT や長尺物体の unknown CoM grasp-and-lift を使い、手法が controlled setting で latent adaptation を使えることを示す。

第二に、実機でスポンジ拭き、またはブラシ剛性の違う coffee-bean sweeping のようなタスクを行い、画像だけでは接触品質を十分に捉えにくい場面で force imagination が効くことを示す。

重要な ablation は以下である。

- imagined force なし
- imagined force はあるが online latent correction なし
- latent ではなくネットワーク全体をオンライン更新
- 実測 force history を方策に直接入力
- latent を固定、ゼロ化、ランダム化
- visual world model baseline
- latent 次元や context 長の比較

さらに、推定 latent が物性ごとにクラスタリングするか、latent を変えたときに force estimate や policy action が意味のある変化をするかを見ると、表現としての説得力が増す。

## 論文での主張の仕方

避けるべき主張は以下である。

```text
既存の visual world model では hidden physics は解けない。
```

これは強すぎる。PushT のように hidden physics が画像上の運動差として現れる場合、visual world model でも解ける可能性がある。

より安全で鋭い主張は以下である。

```text
既存の visual world model は主に可視状態の予測誤差を通じて適応する。
一方で本研究は、接触品質・圧力・摩擦・コンプライアンスのように、
タスクに重要だが通常画像には十分現れにくい hidden material property に対し、
force estimation error を使って compact latent を適応する。
```

要するに、本研究は「力を使った模倣学習」ではなく、力の推定誤差を使って、見えない物性をオンライン同定し、その同定結果で模倣方策を修正する研究である。この点を前面に出すと、ACP、UMI-FT、ForceMimic、AdaWorldPolicy、DyWA などの強い先行研究との差分が明確になる。

