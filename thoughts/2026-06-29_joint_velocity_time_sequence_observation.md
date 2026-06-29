# Joint Velocity and Time-Sequence Observation

User original note on 2026-06-29.

joint velocity は time sequence 観測があれば不要かもしれない

提案手法では「持ち直し」が発生する。
一見すると、持ち直しの有無から「今アームが対象を持ちに行っているのか / 離れているのか」の情報が必要そうに見える。

ただ、観測に time sequence が入っているなら、joint position の履歴から速度情報は暗黙に復元できる。
つまり、joint velocity は入れても入れなくても大きくは変わらない可能性がある。

一方で、ACT のような手法では current observation への依存が強く、joint velocity が必要になる可能性がある。

タグ:
#robotics #observation #joint_velocity #time_sequence #ACT #提案手法
