## [2026-04-28 10:26] query | 過去 wrench を入力に入れるべきか

### 結論

研究の主目的が **正確な次時刻 wrench 予測** ではなく、**rich な物性表現の獲得** であるなら、
少なくとも主たる物性表現学習タスクにおいては、`t` 時刻の wrench をそのまま入力に入れない方がよい。

より正確に言うと、

- `wrench_t` を入力に入れた 1-step 予測は、予測問題としては自然
- しかし、物性表現学習のための補助タスクとしては近道が多い
- そのため、主たる表現学習では `wrench_t` なしの estimator / identifier 的な設定の方が筋がよい

というのが現時点での意見である。

### 理由

現在の設定では、`measured_eef_wrench_moving_average_t` が state に入り、
target は `measured_eef_wrench_moving_average_{t+1}` である。
このときモデルは、

- 現在 wrench から次 wrench への滑らかな外挿
- admittance 系の局所応答の補間
- moving average 済み信号の時系列的慣性

だけでもかなり loss を下げられる可能性が高い。

つまり、loss を下げるために hidden physics を強く使う必然がない。
CoM や剛性や摩擦が効いていないわけではないが、
それらを明示的に表現しなくても性能が出てしまう危険がある。

この意味で、`wrench_t -> wrench_{t+1}` は
「物性表現を学ばせるタスク」というより
「wrench の自己回帰に近いタスク」になりやすい。

### では過去 wrench は完全に不要か

完全に不要とは思わない。
ただし、**役割を分けるべき** だと思う。

- 物性表現を作る主タスク:
  過去 wrench は入れない、もしくは非常に制限する
- 制御の安定化や最終精度を上げる補助入力:
  過去 wrench を別枝で使う

つまり、`wrench` は「便利だから入れる」のではなく、
「何のために入れるか」を分離すべきである。

### 一番避けたい設計

一番避けたいのは、

- 入力に `wrench_t` を入れる
- 出力は `wrench_{t+1}`
- horizon は 1 step
- その中間表現を物性表現と解釈する

という設計である。

この場合、中間表現が rich な hidden physics を持っているのか、
単に局所時系列予測に都合のよい表現なのかがかなり曖昧になる。

### ただし単発 estimator だけでも弱い

一方で、`wrench_t` を消せばそれで十分とも思わない。
単時刻の

- `image_t`
- `eef_pose_t`
- `command_t`

だけから `wrench_t` や `wrench_{t+1}` を当てる設定は、
hidden physics を推定するには情報が足りない可能性がある。

物性は、単発観測というより、
**短い相互作用履歴の中で初めて見える** ことが多いからである。

したがって本当に欲しいのは、
`predictor` と `estimator` の二択というより、
**interaction history から hidden physics を推定する identifier**
に近い。

### 現時点での推奨

主たる物性表現学習タスクとしては、次の順がよい。

1. `wrench_t` は入力から外す。
2. 代わりに `image, pose, action` の短い履歴を入れる。
3. target は `wrench_t` 単発より、`wrench_{t:t+H}` あるいは `wrench_{t+1:t+H}` の方がよい。
4. 評価は wrench 誤差だけでなく、その中間表現を下流 policy に入れたときの改善で見る。

### 妥協案

もし過去 wrench を完全に捨てるのが不安なら、次のような妥協案がある。

- 主 branch:
  `image, pose, action` 履歴から latent `z` を作る
- 補助 branch:
  過去 wrench を使って最終予測だけ少し助ける

このとき重要なのは、
**下流 policy に渡すのは過去 wrench 依存の最終 head ではなく、主 branch の latent にする**
ことである。

### 短い結論

「過去 wrench を入れるべきか」という問いへの答えは、

- 予測精度を上げたいなら、入れるのは自然
- 物性表現を学びたいなら、主タスクには入れない方がよい

である。

したがって、研究の主眼が物性表現なら、
**過去 wrench は主入力から外し、短い相互作用履歴ベースの estimator / identifier に寄せるべき**
というのが現時点での私の意見である。

### 参照したローカル実装

- `/home/kotaro/my_projects/robotics/RoboManipBaselines/unstructured/commands_RMB2.sh`
- `/home/kotaro/my_projects/robotics/RoboManipBaselines/robo_manip_baselines/policy/wrench_predictor/WrenchPredictorDataset.py`
- `/home/kotaro/my_projects/robotics/RoboManipBaselines/robo_manip_baselines/policy/wrench_predictor/WrenchPredictorPolicy.py`
- `/home/kotaro/my_projects/robotics/RoboManipBaselines/robo_manip_baselines/policy/wrench_predictor/detr/models/wrenchpredictor_core.py`
