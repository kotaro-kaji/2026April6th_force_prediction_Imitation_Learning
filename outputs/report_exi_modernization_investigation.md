# EXI-Net 現代化の実装調査

timestamp: 2026-04-21 13:45 JST
operation: query
changed files: `outputs/report_exi_modernization_investigation.md`
result summary: `raw/dynamical-metalearning` が、EXI-Net を出発点とした予測器・物性 latent の現代化にとって有力な実装参考になるかを整理した。

## あなたがやろうとしていること

今回の中心は「Liさんの face keypoint predictor を使うか」ではない。
本当の文脈は次の通り。

- 予測器と物性表現 latent の元の発想は EXI-Net から来ている
- ただし EXI-Net は実装テンプレートとしてはやや古く感じている
- そこで、根本の考え方は残しつつ、より現代的な構成に差し替えたい
- 興味の中心は policy ではなく、予測器と hidden-physics / material latent にある

つまり今探しているのは、次を満たす現代的な実装のヒント。

1. hidden dynamics や material 情報を使う予測器
2. デプロイ時に適応可能な latent 表現
3. EXI-Net より今風で、実装もしやすい設計

## 結論

`raw/dynamical-metalearning` は、Liさんの face keypoint prediction よりも、あなたの研究にとってかなり良い実装参考になる。
ただし、そのまま最終形として使うべきものではない。

一番自然な読み方はこう。

- Liさんの仕事は「制御前に役立つものを予測して入れる」という構造的参考
- `dynamical-metalearning` は hidden dynamics の扱いという意味でかなり近い
- ただし、あなたの理想形はそのさらに先にあるハイブリッド

## なぜ Liさんの face prediction は主テンプレートではないのか

Liさんの論文は次のことをやっている。

- MediaPipe で 468 個の顔 landmark を抽出する
- RGB と landmark を使って短期先の顔運動を予測する
- 予測した landmark を ACT 系の模倣学習 policy に入力する

この論文が役に立つのは、次の形がきれいだから。

1. task-relevant な未来変数を予測する
2. それを policy の入力に入れる

この構造自体は force prediction にも応用可能。
ただし、そこにある hidden variable は顔運動であって、物体の物理特性ではない。
hidden dynamics identification の方法とは言いにくいし、object/material latent を中心にもしていない。

なので、あなたの主問題に対する実装テンプレートとしては弱い。

## なぜ dynamical-metalearning の方が近く見えるのか

追加した submodule は system identification と forward dynamics prediction を中心にしている。
重要なのは次の点。

- 軌道の context から予測器を学習している
- mass, center of mass, inertia などの hidden physical parameters を明示的にランダム化している
- 最近の履歴から dynamics を読む sequence model を使っている
- 変化した dynamics への汎化を評価している

この時点で、顔 landmark prediction よりもあなたの hidden-physics 設定にかなり近い。

特に transformer 系の経路は重要で、

- 過去の観測と制御の context trajectory を埋め込み
- context を encode し
- 将来の制御に対する将来出力を予測する

という構成になっている。

言い換えると、これはすでに

- 「履歴を見れば今どんな環境か分かる」
- 「その inferred environment 情報を使って未来の挙動を予測する」

という設計空間に入っている。
ここはあなたの狙いにかなり近い。

## それでも dynamical-metalearning がそのまま目標ではない理由

最大のズレは、研究の見せ方と method の明確さにある。

あなたがやりたい話は、

- predictor を学習する
- それを compact な hidden-physics 表現で条件付けする
- デプロイ時には predictor 本体は凍結する
- hidden-physics 表現だけを online に適応させる

というもの。

一方 `dynamical-metalearning` は、その explicit latent を method の主役として前面には出していない。
むしろ context window 全体による implicit な system identification に寄っている。
hidden physics は sequence model の内部表現に溶け込んでおり、`d_i` や `z_env` のような compact な per-object latent として明示されていない。

これはあなたの研究では重要な違いになる。
なぜなら explicit latent にしておく利点が少なくとも 4 つあるから。

1. method の説明がしやすい
2. EXI-Net の系譜に素直につながる
3. デプロイ時に何を適応させるかが明確
4. unseen material, mass, CoM, compliance の話が整理しやすい

なので `dynamical-metalearning` は工学的な参考としてはかなり近いが、論文主張として一番きれいな形ではまだない。

## EXI-Net との関係でどう解釈するのが自然か

一番自然な位置づけは次の通り。

- EXI-Net は概念的な出発点
- `dynamical-metalearning` は現代的な system identification の参考
- あなたの method は EXI-Net の「デプロイ時に latent 側だけ適応させる」論理を残すべき
- ただし latent inference のやり方は現代化すべき

要するに、

- EXI-Net: latent を比較的明示的に持ち、最適化で合わせる
- dynamical-metalearning: hidden dynamics を context から暗黙に読む
- あなたの最適案: 履歴から compact latent を推定し、その latent だけを online に適応する

## 推奨

「今すぐ実装のヒントとしてどの repo を見るべきか」という問いへの答えは、

`raw/dynamical-metalearning`

でよい。

ただし「研究貢献として何を実際に作るべきか」という問いへの答えは、

EXI-Net の deployment logic と dynamical-metalearning の history-based inference を組み合わせたハイブリッド

になる。

より具体的には、強い方向性は次の通り。

1. observation-action-force history に対する短い history encoder を使う
2. そこから explicit な environment / material latent `z_env` を出す
3. `z_env` を force predictor あるいは residual dynamics predictor に入れる
4. デプロイ時には predictor weights を凍結する
5. online では `z_env` だけを更新する

この更新方法は、

- 毎ステップ encoder で直接推定する
- latent optimization で微調整する
- encoder 初期化 + 少しだけ latent refinement する

のいずれでもよい。

## dynamical-metalearning から借りるとよい部分

借りる価値が高いのは次。

- context-conditioned dynamics prediction という考え方
- hidden physical parameters をランダム化した学習設定
- static な object ID table ではなく短い履歴窓を使うこと
- history encoder としての Transformer 的な使い方

逆に、そのままは借りにくい部分は次。

- hidden dynamics を context encoder の内部に暗黙表現として閉じ込める end-to-end な見せ方
- compact latent を method の主役として露出していない点
- predictor と latent だけ欲しい場合には repo 全体がやや重い点

## 実装の当面のすすめ方

最初の強いプロトタイプとしては、次がよい。

- 入力: image features, current force, action, short recent history
- history encoder: 小さめの Transformer encoder
- latent head: compact な `z_env` を出す
- predictor head: next force, future force sequence, または residual next state を予測する
- deployment: network weights は固定し、`z_env` だけを更新する

これなら、

- 現代的な実装になる
- EXI-Net の発想と素直につながる
- pure implicit context conditioning より latent の話が明快になる
- submodule 全体をそのまま持ってくるよりシンプルで defend しやすい

## 最終判断

あなたの直感は方向としては合っていた。
追加した submodule は「Liさんがたぶん使っていたもの」ではないが、Liさんの face predictor よりは、あなたの研究のヒントとしてかなり有用。

なので答えは、

- 「Liさんの実装を使う」

でもなく、

- 「dynamical-metalearning をそのまま使う」

でもない。

正確には、

- 「dynamical-metalearning を、より良い現代的ヒントとして使う。ただし method の中心は、履歴から推定されて online に適応できる explicit な EXI-Net 系 latent に置く」

である。
