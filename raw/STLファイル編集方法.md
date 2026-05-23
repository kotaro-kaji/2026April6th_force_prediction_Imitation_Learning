stlファイルを編集したいんだけど、方法ないかな。初心者。

---

**ChatGPT**

あります。初心者なら、 **STLを「粘土みたいに直接いじる」のか、「寸法を決めてCADっぽく直す」のか** で方法を分けるのが一番わかりやすいです。

## 結論

一番おすすめはこの順です。

| 目的 | おすすめソフト | 難易度 | 向いている編集 |
| --- | --- | --- | --- |
| ちょっと削る・穴を開ける・分割する | **PrusaSlicer / Bambu Studio / Cura** | 低 | 3Dプリント前の簡単加工 |
| 形をなめらかに直す・へこみを直す | **Meshmixer** | 低〜中 | メッシュ編集、修復、切断 |
| 寸法を指定して正確に直す | **Fusion 360** | 中 | CAD的な編集、穴径変更、面取り |
| 本格的に形状を作り替える | **Blender** | 中〜高 | 自由変形、有機的形状、彫刻 |
| ブラウザで簡単に加工 | **Tinkercad** | 低 | 穴あけ、結合、単純形状追加 |

## 初心者におすすめの選び方

### 1\. 3Dプリント用に少し直したいだけ

たとえば、

- サイズを変えたい
- 一部を切り落としたい
- モデルを分割したい
- サポートや向きを調整したい
- 簡単な穴を追加したい

この程度なら、まずは **スライサーソフト** で十分です。

使っているプリンタがあるなら、

- Bambu Lab系 → **Bambu Studio**
- Prusa系 → **PrusaSlicer**
- 汎用 → **Cura**

が無難です。

STLを読み込んで、拡大縮小、回転、分割、カットなどができます。

---

### 2\. STLそのものを直接修正したい

初心者なら **Meshmixer** がわかりやすいです。

できること：

- STLの穴埋め
- 不要部分の削除
- モデルの切断
- 表面をならす
- 厚みをつける
- メッシュ修復
- 複数STLの合体

ただし、Meshmixerはやや古いソフトなので、将来的にはBlenderやFusion 360に移行する可能性があります。それでも「STLをちょっと直す」用途では今でも使いやすいです。

---

### 3\. 寸法をきっちり変えたい

たとえば、

- 穴径を 3.0 mm から 3.5 mm にしたい
- 部品の幅を 20 mm にしたい
- ネジ穴の位置を正確に変えたい
- 平面に正確な突起を追加したい

こういう場合は **Fusion 360** が向いています。

ただし注意点があります。STLは普通のCADデータではなく、三角形の集合です。そのためFusion 360で編集するときは、

1. STLを読み込む
2. メッシュをBRepやソリッドに変換する
3. CAD機能で編集する

という流れになります。

STLの三角形が多すぎると変換が重くなったり失敗したりします。

---

### 4\. 自由に形を変えたい

たとえば、

- フィギュアを改造したい
- 曲面を押し出したい
- 表面を彫刻みたいに変えたい
- 複雑な形をなめらかに加工したい

この場合は **Blender** が向いています。

ただし、Blenderは最初の操作が少し難しいです。機能は強いですが、初心者が「穴径を正確に3.5 mmにしたい」みたいな用途で使うと逆に大変です。

---

## まず試すならこれ

初心者なら、次の順で試すのが現実的です。

1. **スライサーでできないか確認**
2. 無理なら **Meshmixer**
3. 寸法精度が必要なら **Fusion 360**
4. 形状を大きく作り替えるなら **Blender**

## よくある編集別のおすすめ

| やりたいこと | 方法 |
| --- | --- |
| サイズ変更 | スライサー、Tinkercad、Fusion 360 |
| 一部を切る | PrusaSlicer、Bambu Studio、Meshmixer |
| 穴を開ける | Tinkercad、Fusion 360、Meshmixer |
| 欠け・穴を修復 | Meshmixer、Blender、Windows 3D Builder |
| 複数STLを合体 | Tinkercad、Meshmixer、Blender |
| 寸法を正確に編集 | Fusion 360 |
| フィギュア・曲面編集 | Blender |
| STLを軽くする | Meshmixer、Blender |

## 最初のおすすめルート

**Windowsなら、まずは「3D Builder」か「Meshmixer」からで十分です。**

特に、単純な修正なら Windows標準系の **3D Builder** でもできます。  
もう少し編集したいなら **Meshmixer** 。  
寸法を正確にしたいなら **Fusion 360** 。

STL編集は「CAD編集」と違って、元データが三角形の集合なので、きれいに編集できないことがあります。可能なら、元の **STEP / F3D / SolidWorks / Fusion形式** などがある場合は、STLではなくそちらを編集する方が圧倒的に楽です。

---

Bambuu だけど、うーん。一部を切り落としたいだけ。でも切り落としって一部の方向だけ？なんか直方体作って直方体の中のやつ切り落としとかってできる？どうやる？

---

**ChatGPT**

Bambu Studioでできる範囲を前提に説明します。ポイントは「平面でスパッと切る」機能と、「直方体を当てて削る」機能は別物です。

---

いいね。えっと Bambu StudioでSTLを開く モデルを右クリック Add Negative Part / 負のパーツを追加 を選ぶ Cube / 立方体 を選ぶ 4までいった。でも高さ方向でcubeを移動できない。平面方向しかない

---

**ChatGPT**

それはたぶん、 **Cube本体を普通にドラッグしている** ので、Bambu Studioが「ベッド上の平面移動」に制限している状態です。

高さ方向に動かすには、次のどちらかを使います。

## 方法A：左のObjectsからNegative Cubeを選んで、Z座標を数値入力

一番確実です。

1. 左側の **Objects / オブジェクト一覧** を開く
2. 元STLの下にある **Negative Cube** または **Cube** を選択  
	※元モデルではなく、追加した負の立方体を選ぶ
3. 画面上部の **Move / 移動** を押す、または `M` キー
4. 右側または上部に出る座標欄で **Z** の値を変える

Bambu公式Wikiでも、Move機能は上部ツールバーまたは `M` キーから使うと説明されています。

たとえば、CubeのZを上げたいなら、

```
Z = 10 mm
Z = 20 mm
Z = 35 mm
```

みたいに数値で入れます。

---

## 方法B：矢印ハンドルの「青い矢印」をドラッグする

MoveツールをONにすると、赤・緑・青の矢印が出ます。

```
赤 = X方向
緑 = Y方向
青 = Z方向、高さ方向
```

**青い矢印をつかんでドラッグ** すると、高さ方向に動かせます。

ただし、見えている角度によっては青い矢印が押しにくいです。  
その場合は、マウスで視点を少し回してから、青矢印をつかむとやりやすいです。

---

## よくある詰まりどころ

### 1\. 元モデルを選んでいる

Negative Cubeではなく、元のSTLを選んでいると、Cubeだけ動きません。

左のObjects一覧で、元モデルの下にある **Cube / Negative Cube** を選んでください。

---

### 2\. Assembly全体を選んでいる

元モデル＋Negative Cube全体を選んでいると、一緒に動いてしまいます。

この場合も、左のObjects一覧から **Cubeだけ** を選びます。  
Bambu Studioでは、結合・グループ内の個別パーツを選ぶには、左側のObjects一覧から選ぶのが確実です。

---

### 3\. ドラッグ移動ではZが動かない

これは正常です。  
ベッド面上で直接ドラッグすると、基本的にXY平面移動になります。

高さ方向は、

```
Moveツール → 青矢印
```

または

```
座標欄のZを数値入力
```

で動かします。

---

## 今回やるべき操作

今の状態からなら、これで進めてください。

1. 左のObjectsで **Negative Cube** をクリック
2. `M` キーを押す
3. 青い矢印をドラッグ、または **Z座標** を入力
4. Cubeを切りたい高さに合わせる
5. **Slice Plate**
6. Previewで削れているか確認

削る直方体の位置合わせは、ドラッグよりも **X/Y/Zの数値入力** のほうが失敗しにくいです。

---

お。いけそう。あとでかさ変えたい

---

**ChatGPT**

Cubeの大きさ変更は、 **Negative Cubeだけを選択して Scale / 拡大縮小** でできます。

## 手順

1. 左の **Objects** で **Negative Cube** を選択  
	※元のSTLではなく、Cubeだけ
2. 上部ツールバーの **Scale / 拡大縮小** を押す  
	または `S` キー
3. 寸法を変える  
	画面右側または上部に、だいたい次のような項目が出ます。
```
X: 幅
Y: 奥行き
Z: 高さ
```
4. **Uniform scale / 比率固定** がONならOFFにする  
	これがONだと、X/Y/Zが全部同じ倍率で変わります。
5. X/Y/Zを個別に入力する  
	例：
```
X = 20 mm
Y = 10 mm
Z = 30 mm
```

これで、 **横長・縦長・薄い板状のNegative Cube** にできます。

## マウスで変える場合

Scaleツールを押すと、Cubeの周りにハンドルが出ます。

- 角のハンドル：全体を拡大縮小
- 面方向のハンドル：一方向に拡大縮小
- 数値入力：一番正確

初心者なら、 **数値入力** のほうが安全です。

## 注意点

Negative Cubeを大きくすると、当然その範囲に入ったモデル部分が全部削れます。  
変更後は必ず、

```
Slice Plate → Preview
```

で、削れ方を確認してください。

今回の用途では、

```
Move / M → 位置調整
Scale / S → 大きさ調整
Slice Plate → 結果確認
```

の3つだけ覚えれば十分です。

---

よし。最高。あった。次は実際に切り落としからの保存までやりたい

---

**ChatGPT**

ここからは「印刷時だけ削る」ではなく、形状データとして確定させる操作です。Bambu Studio内でできる場合はMesh Booleanを使い、見つからない場合は別ソフトに逃がすのが確実です。

---

Mesh Boolean選ぶとこうなる

---

**ChatGPT**

これは **元STL `right_finger.stl` が自己交差していて、Booleanできない** というエラーです。Cube側ではなく、 **切られる側のSTLが壊れ気味** です。

表示内容の意味はこれです。

```
self intersection
= STLの三角形メッシュが自分自身にめり込んでいる / 交差している
```

## まずBambu Studio内で試すこと

### 1\. 元モデルを修復する

1. `right_finger.stl` を選択
2. 右クリック
3. **Fix Model / モデルを修復** を選ぶ
4. 修復が終わったら、もう一度 **Mesh Boolean**

これで通ることがあります。

---

## Bambu Studioで無理なら、一番ラクな逃げ道

この場合、 **Microsoft 3D Builderで修復 → Bambu Studioに戻す** が初心者には一番簡単です。Windowsなら特に楽です。

### 手順

1. **3D Builder** を開く
2. `right_finger.stl` を読み込む
3. 「1つ以上のオブジェクトが無効です。修復しますか？」のような表示が出たら **修復**
4. 修復後に **名前を付けて保存**
5. 修復済みSTLをBambu Studioで開く
6. もう一度、CubeとMesh Boolean

3D BuilderはSTL修復がかなり簡単です。Boolean前の前処理として使えます。

---

## それでもダメなら、Blenderで切る方が早い

Bambu StudioのBooleanは、STLが少し壊れていると失敗しやすいです。  
今回のように自己交差がある場合、 **Bambu Studioで頑張るより、BlenderでBooleanまたは切断してからSTL保存** の方が安定します。

ただし、初心者向けの順番としてはまずこれです。

```
Bambu StudioのFix Model
↓
ダメなら3D Builderで修復
↓
修復済みSTLをBambu StudioでBoolean
↓
それでもダメならBlender
```

## 今やるべきこと

まずはこれだけ試してください。

```
right_finger.stlを右クリック
→ Fix Model / モデルを修復
→ もう一度 Mesh Boolean
```

それで同じエラーが出るなら、 **3D Builderで修復してから戻す** のが次の最短ルートです。

---

Fix Modelなんてないな。右クリックしても

---

**ChatGPT**

Bambu Studioの版や環境によって、右クリックに **Fix Model** が出ないことがあります。ここで詰まるなら、Bambu内修復にこだわらず、 **Windowsの3D Builderで修復してから戻す** のが最短です。

## 一番簡単な手順：3D Builderで修復

1. Windowsで **3D Builder** を起動  
	入っていなければ Microsoft Store から入れます。
2. `right_finger.stl` を開く
3. 読み込んだ直後に、モデルが壊れている場合はだいたい  
	**「1つ以上のオブジェクトが無効です。修復しますか？」**  
	のような表示が出ます。
4. **修復** を押す
5. 修復後に保存
	```
	名前を付けて保存 → STL
	```
6. 修復済みSTLをBambu Studioに読み込む
7. もう一度
	```
	普通のCubeを配置
	→ 元モデルとCubeを選択
	→ Mesh Boolean
	→ Difference
	→ Export as STL
	```

## 3D Builderで直した後の注意

Bambu Studioに戻すときは、 **Negative Cubeではなく普通のCube** を使った方が安定します。

```
Add Part → Cube
```

で普通のCubeを追加して、それを「削るための箱」として使います。

## 代替：Bambu Studio内で修復できる場所を探す場合

右クリックにない場合でも、メニュー側にあることがあります。

探す場所：

```
上部メニュー
→ Repair / 修復
```

または

```
モデル選択時の警告アイコン
→ Repair
```

ただ、見つからないなら探し回るより、3D Builderに渡した方が早いです。

## 今回の最短ルート

```
3D Builderで right_finger.stl を開く
→ 自動修復
→ STLで保存
→ Bambu Studioに戻す
→ 普通のCubeでMesh Boolean
```

このエラーは操作ミスというより、 **元STLのメッシュ品質の問題** です。Bambu Studio単体で無理に直そうとすると時間を食いやすいです。

---

ubuntuです

---

**ChatGPT**

Ubuntuなら、 **3D Builder案は使えない** ので、次のどちらかが現実的です。

## 最短おすすめ：Blenderで修復して切る

Ubuntuなら **Blender** が一番確実です。

インストール：

```bash
sudo apt update
sudo apt install blender
```

または Snap版：

```bash
sudo snap install blender --classic
```

## Blenderで「直方体で切り落としてSTL保存」する手順

### 1\. STLを読み込む

Blenderを開いて、

```
File → Import → STL
```

で `right_finger.stl` を読み込みます。

最初にある立方体が邪魔なら、選択して `X` → Delete で消します。

---

### 2\. カッター用のCubeを作る

```
Shift + A
→ Mesh
→ Cube
```

で立方体を追加します。

このCubeを、切り落としたい部分に置きます。

操作：

```
G = 移動
S = 拡大縮小
R = 回転
```

軸指定もできます。

```
G → Z = 高さ方向に移動
S → X = X方向だけ拡大縮小
S → Y = Y方向だけ拡大縮小
S → Z = Z方向だけ拡大縮小
```

例：

```
G → Z → 10
```

でZ方向に10 mm移動します。

---

### 3\. Booleanで削る

1. `right_finger.stl` のモデルを選択
2. 右側のレンチアイコン **Modifiers** を開く
3. **Add Modifier**
4. **Boolean**
5. Operationを **Difference**
6. Objectに、さっき作った **Cube** を指定
7. Solverを **Exact** にする
8. **Apply**

これで、

```
right_finger − Cube
```

が実行されます。

---

### 4\. Cubeを消す

BooleanをApplyしたあと、Cubeはもう不要です。

Cubeを選択して、

```
X → Delete
```

---

### 5\. STLとして保存

削れたモデルを選んだ状態で、

```
File → Export → STL
```

このとき、できれば

```
Selection Only
```

をONにして、削れたモデルだけを書き出します。

---

## BlenderでもBooleanに失敗する場合

元STLが自己交差しているので、先に修復します。

Blenderでモデルを選択して：

```
Tab
→ Edit Mode
→ A で全選択
→ Mesh
→ Clean Up
→ Merge by Distance
```

次に：

```
Mesh
→ Clean Up
→ Delete Loose
```

さらに：

```
Shift + N
```

で法線を再計算します。

その後、もう一度Booleanを試します。

---

## コマンドで修復したい場合

Ubuntuなら **MeshLab** も使えます。

```bash
sudo apt install meshlab
```

MeshLabで開いて、修復系フィルタを使う方法もありますが、初心者にはBlenderより少し分かりにくいです。

より自動寄りなら **admesh** もあります。

```bash
sudo apt install admesh
admesh --repair right_finger.stl --write-binary-stl right_finger_repaired.stl
```

そのあと、 `right_finger_repaired.stl` をBambu Studioに戻してMesh Booleanを試します。

## 今回のおすすめ

まずはこれです。

```bash
sudo apt install admesh
admesh --repair right_finger.stl --write-binary-stl right_finger_repaired.stl
```

それでできた `right_finger_repaired.stl` をBambu Studioに読み込み直して、普通のCubeでMesh Booleanを試すのが一番早いです。

ダメならBlenderで直接切り落としてSTL保存、が次の手です。