# Transformer 実装教科書

timestamp: 2026-04-21 14:20 JST
operation: query
changed files: `outputs/report_transformer_textbook_for_implementation.md`
result summary: Transformer が何かを最短で理解し、`raw/dynamical-metalearning` を読み始めて自分の実装に繋げるための実装志向の教科書を作成した。

## この教科書のゴール

この教科書の目的は、Transformer の歴史や全派生モデルを網羅することではない。
目的は次の 1 点だけである。

「これを読めば、Transformer をある程度ブラックボックスとして扱いつつ、`raw/dynamical-metalearning` を読み始め、自分の `history -> z_env -> predictor` 実装にブーストをかけられる」

つまり、研究者向けの完全講義ではなく、実装者向けの最短ルートである。

## 最初に結論

今のあなたに必要なのは、Transformer を厳密に全部理解することではない。
必要なのは次の理解である。

- Transformer encoder は「履歴列を読んで、各時刻の文脈化された表現を返す部品」
- 入力は token 列
- 出力は token ごとの hidden state 列
- その hidden state から履歴全体の要約 `z_env` を作って predictor に渡せばよい

この API 的理解で、かなりのところまで実装できる。

## 高レベル像

まず、Transformer 全体の元々の姿を 1 枚で見る。
これは encoder-decoder の古典形だが、あなたが今すぐ必要なのは主に encoder 側だけである。

![Transformer 全体像](https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)

画像出典:
[The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

この図で今覚えるべきことは 2 つだけ。

1. Transformer には encoder と decoder がある
2. あなたの用途では、まず encoder が主役

理由は、あなたがやりたいことが「履歴から hidden physics に効く表現を抜く」ことであって、「次の単語を1個ずつ生成する」ことではないからである。

## あなたの用途に引き直した Transformer

あなたの研究文脈では、Transformer を次のように読み替えればよい。

- 入力 token: 各時刻の observation, action, force などをまとめた特徴
- encoder: その時系列全体を読んで文脈化する
- pooling / special token / final state: 履歴全体の要約を作る
- latent head: `z_env` を出す
- predictor head: `z_env` を使って force や next state を予測する

式より先に、この対応づけを腹落ちさせるのが重要。

## encoder は何をしているのか

encoder の 1 ブロックは実はかなり単純で、基本的には

- self-attention
- feed-forward network

の組み合わせでできている。

![Encoder ブロック](https://jalammar.github.io/images/t/Transformer_encoder.png)

画像出典:
[The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

ここで大事なのは、encoder は「履歴を読んで、各 token を文脈込みの token に変える装置」だということ。

RNN 的に言えば、履歴情報を埋め込んだ hidden state を各時刻ごとに返す装置に近い。
ただし RNN と違って、左から順番に処理するのではなく、時刻同士が直接見に行ける。

## self-attention の役割

あなたはすでに self-attention の文脈に入っているので、ここでは実装者向けにだけ整理する。

![QKV の図](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

画像出典:
[The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

self-attention をあなた向けに言うと、

- ある時刻の token が
- 他のどの時刻を見るべきかを決めて
- 必要な情報を集める

という仕組みである。

あなたの問題に置き換えると、

- 「今の force 予測のために、どの過去の接触や action が効くのか」
- 「この履歴の中のどこが物性を示しているのか」

を読むための部品になる。

なので、self-attention は「魔法の知能」ではなく、「履歴のどこを見るかを学習する仕組み」と思えばよい。

## 実装者として最低限知るべき API

Transformer を実装で使うには、次の 5 点だけは押さえた方がよい。

### 0. `B, T, D` は何か

この教科書では tensor の shape を次のように書く。

- `B`: batch size
- `T`: time steps または sequence length
- `D`: feature dimension または embedding dimension

たとえば `(B, T, D)` は、

- `B` 個の系列があり
- 各系列に `T` 個の時刻があり
- 各時刻が `D` 次元の特徴ベクトルで表されている

という意味である。

あなたの文脈だと、`D` はたとえば

- observation feature
- action
- force
- image embedding

などを結合した token の次元になる。

### 1. 入力 shape

PyTorch では典型的に次のどちらかになる。

- `(seq, batch, feature)`
- `(batch, seq, feature)` if `batch_first=True`

今のあなたは `batch_first=True` で統一してよい。
つまり基本は

`(B, T, D)`

で考えればよい。

### 2. 出力 shape

encoder は通常、各 token に対する hidden state を返す。
つまり入力が `(B, T, D)` なら、出力もだいたい `(B, T, D)`。

大事なのは「履歴全体の要約が自動で1個返るわけではない」ということ。
要約 `z_env` は自分で取り出す必要がある。

### 3. 履歴要約の作り方

`z_env` の作り方は主に 3 つある。

1. 最終時刻の hidden state を使う  
2. 全時刻平均 pooling を使う  
3. 学習可能な special token を先頭に入れて、その出力を使う

最初の実装なら 1 か 2 で十分。

### 4. mask

mask は「どこを見てよいか」を制限する。

- encoder で過去も未来も全部見てよいなら causal mask は不要
- 未来を見てはいけない予測なら causal mask が必要
- padding 部分を無視したいなら key padding mask が必要

あなたの「履歴から latent を推定する」用途では、まずは履歴窓の全時刻を見てよいので、causal mask なしから始めてよいことが多い。

### 5. encoder だけで足りるか

今のあなたは、ほぼ encoder だけで足りる。
decoder が必要になるのは、生成を 1 ステップずつ行うときや、元の machine translation 形に近いとき。

force predictor や latent inference なら、まず encoder だけで考えてよい。

## 最短で理解するための公式資料

まずはブログや二次資料より、PyTorch 公式 docs を土台にした方がよい。
実装との接続が最も強いからである。

### `MultiheadAttention`

ここでは

- `query / key / value`
- 入出力 shape
- `attn_mask`
- `key_padding_mask`

を確認する。

公式:
https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

### `TransformerEncoderLayer`

ここが最重要。
1 ブロックの中身を知るページである。

公式:
https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html

### `TransformerEncoder`

これは `TransformerEncoderLayer` を複数積んだもの。

公式:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

## 論文は読むべきか

読むなら後でよい。
今すぐ最優先ではない。

読むならこれだけで十分。

`Attention Is All You Need`
https://arxiv.org/abs/1706.03762

この論文を読む目的は、

- Transformer の元の姿を知る
- encoder/decoder/self-attention の関係を確認する

くらいでよい。

最初から数式を全部追う必要はない。
あなたの今の目標は実装なので、論文の完全理解より docs とコードの対応付けの方が重要。

## 人間が書いた、しかもかなり良い補助資料

もし「人間が書いた説明のほうが頭に入りやすい」なら、これはかなり良い。

### The Illustrated Transformer

https://jalammar.github.io/illustrated-transformer/

Transformer の高レベル像をつかむには非常に強い。
図が良いので、最初の直観形成に向いている。
ただし、実装 API の理解は PyTorch 公式 docs で補うべき。

### The Annotated Transformer

https://nlp.seas.harvard.edu/annotated-transformer/

これは「論文と実装の橋渡し」として強い。
Jay Alammar よりコード寄りで、元論文より実装寄り。
PyTorch で line-by-line に近い理解をしたいなら非常に有用。

今のあなたなら、

1. Illustrated Transformer で図解の直観を掴む
2. PyTorch docs で API を確認する
3. Annotated Transformer で橋渡しする

の順がよい。

## あなたの用途に最も重要な 1 行

Transformer encoder は、

「履歴 token 列を入れると、各時刻について文脈化された表現列を返す関数」

として理解してよい。

つまり API 的には、

`H = Encoder(X)`

で、

- `X`: `(B, T, D_in)` を linear などで `(B, T, D_model)` に写した token 列
- `H`: `(B, T, D_model)` の文脈化済み token 列

となる。

その後は自分で

- `H[:, -1, :]`
- `H.mean(dim=1)`
- special token の出力

のどれかを選んで `z_env` を作ればよい。

## あなた向けの最小設計

あなたの研究文脈で一番自然な最小設計は次。

### 入力

各時刻で

- image feature
- current force
- action
- 必要なら proprioception

を連結し、token にする。

### 履歴 encoder

小さな `TransformerEncoder` を使う。

### latent head

encoder の出力から履歴要約を作り、MLP で `z_env` を出す。

### predictor head

`z_env` と current observation を使って、

- next force
- future force sequence
- residual next state

のどれかを予測する。

### deployment

predictor と encoder 本体の重みを凍結し、
必要なら `z_env` だけを online に更新する。

## `dynamical-metalearning` をどう読めばよいか

この submodule を今読むときの視点は重要である。
全部を理解しようとすると重い。
見るべきところだけ見る。

### まず見るべきファイル

- `raw/dynamical-metalearning/sys_identification/architectures/transformer/transformer_sim.py`
- `raw/dynamical-metalearning/sys_identification/train.py`
- `raw/dynamical-metalearning/sys_identification/datasets.py`

### 何を見るか

1. 入力が何か  
`y`, `u`, `u_new` など、何を token 化しているか

2. encoder が何を返しているか  
文脈表現をどう future prediction に使っているか

3. shape がどう流れるか  
どの tensor が `(batch, time, feature)` なのか

4. hidden dynamics をどう間接的に扱っているか  
履歴を使って future trajectory を当てているなら、それは implicit sys-id と見なせる

### 今は気にしなくてよいこと

- repo 全体の完全理解
- diffusion 系の枝
- training script の細かい実験設定
- 論文中の全評価指標

まずは transformer 経路だけでよい。

## この教科書を読んだ直後の行動計画

### Step 1

次の 3 つを読む。

- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html)
- [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)

### Step 2

次に次を流し読む。

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

### Step 3

その後、手元の submodule の transformer 実装を読む。

- [transformer_sim.py](/home/kotaro/my_projects/my_life_log/LLM_knowledge_spaces/2026April6th_force_prediction_Imitation_Learning/raw/dynamical-metalearning/sys_identification/architectures/transformer/transformer_sim.py:1)

### Step 4

最後に自分の最小実装を書く。

`history -> encoder -> z_env -> predictor`

これで理解が急に定着する。

## どこまで分かれば十分か

次の問いに答えられれば、実装を始めてよい。

1. token とは何か
2. encoder には何 shape の tensor を入れるか
3. encoder の出力は何 shape か
4. どの hidden state から `z_env` を作るか
5. predictor に何を渡すか
6. 未来 mask が必要か不要か

これに答えられるなら、Transformer の数式を全部覚えていなくても十分に進める。

## 最終メッセージ

Transformer を完全理解してから実装する必要はない。
あなたの今のゴールでは、

- Transformer を encoder API として理解し
- 履歴から文脈表現を出し
- そこから `z_env` を作り
- predictor に流す

という線が見えていれば十分である。

この理解があれば、`raw/dynamical-metalearning` を読む意味もかなり明確になる。
見るべきものは「Transformer の神秘」ではなく、「履歴列をどう token 化して、どう予測に繋いでいるか」だからである。

## 参考文献

1. Vaswani et al., *Attention Is All You Need*  
https://arxiv.org/abs/1706.03762

2. PyTorch `MultiheadAttention`  
https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

3. PyTorch `TransformerEncoderLayer`  
https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html

4. PyTorch `TransformerEncoder`  
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

5. Jay Alammar, *The Illustrated Transformer*  
https://jalammar.github.io/illustrated-transformer/

6. Harvard NLP, *The Annotated Transformer*  
https://nlp.seas.harvard.edu/annotated-transformer/

7. `raw/dynamical-metalearning` submodule  
[README.md](/home/kotaro/my_projects/my_life_log/LLM_knowledge_spaces/2026April6th_force_prediction_Imitation_Learning/raw/dynamical-metalearning/README.md:1)
