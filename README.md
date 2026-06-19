# OjiTopology — Blender Quad Remesher Add-on

選択したオブジェクトを **四角形ポリゴン主体のきれいなエッジフロー** へ自動
リトポロジー（quad remesh）する Blender アドオン。**目標頂点数を指定** して
ワンクリックで実行できる。

![panel](docs/panel.png)

## できること

- 目標頂点数を指定したクリーンな四角形リトポロジー
- N パネル（サイドバー）の UI から大きなボタン一発で実行
- シャープ／境界の保持、スムーズシェード、X/Y/Z 対称
- 2 つのエンジンを切り替え可能

## インストール

1. Blender を開く
2. `Edit > Preferences > Add-ons`
3. `Install...`（4.2 以降は `Install from Disk...`）から `ojitopology.py` を選択
4. 一覧の **OjiTopology Quad Remesher** にチェックを入れて有効化

> Blender 3.0 以降で動作。Field エンジンは Blender 同梱の numpy を使用。

## 使い方

1. メッシュオブジェクトを選択（オブジェクトモード）
2. 3D ビューポートで `N` キー → **OjiTopology** タブ
3. **目標頂点数** を入力し、必要ならオプションを設定
4. **Quad Remesh** ボタンを押す
5. 完了後、ヘッダに `頂点数 / 面数 / quad率` が表示される

## エンジンとアルゴリズム（研究背景）

中核アルゴリズムは以下の論文を踏まえて Blender で使える形に落とし込んでいる。

### 1. QuadriFlow エンジン（既定・堅牢）

- Huang et al., *“QuadriFlow: A Scalable and Robust Method for
  Quadrangulation”* (SGP 2018) のネイティブ実装
  （`bpy.ops.object.quadriflow_remesh`）をラップ。
- 「論文 → Blender への落とし込み」が公式に完了しているコアで、高品質で速い。
- 目標頂点数を、閉じた純 quad メッシュのオイラー公式
  `V − E + F = 2`, `E = 2F` ⇒ `F ≈ V` を用いて **目標面数**へ変換して渡す。

### 2. Field-Aligned エンジン（実験・自作）

Jakob et al., *“Instant Field-Aligned Meshes”* (SIGGRAPH Asia 2015) の
cross field の考え方を Python で独自実装したもの。

処理の流れ：

1. **一様な三角土台** — 表面積 `A` と目標頂点数 `N` から目標辺長
   `L ≈ √(A / N)` を求め、Voxel Remesh で一様な三角メッシュを作る。
2. **cross field の構築** — 各頂点に 4-RoSy（90°対称）の方向場を持たせる。
   - Taubin の曲率テンソル法（ICCV 1995）で主曲率（最小曲率）方向を推定し初期化。
   - 隣接頂点の場を接平面へ射影し 90°対称マッチングして平均する局所平滑化
     （Instant Meshes の中核操作）を反復。
3. **場に沿った quad 化** — 隣接三角形ペアが作る四角形について、エッジ流れ方向が
   cross field にどれだけ整列するか・平面性・正方性をスコアリングし、貪欲に
   選んで共有エッジを溶解し quad へ統合する（quad-dominant 化）。

> Field エンジンは研究目的の実装で、品質・速度は QuadriFlow に劣る。
> 実用では QuadriFlow を推奨。

## 参考文献・ソース

- Instant Field-Aligned Meshes — https://igl.ethz.ch/projects/instant-meshes/
- Instant Meshes (実装) — https://github.com/wjakob/instant-meshes
- QuadriFlow — https://www.researchgate.net/publication/326463904
- Taubin, *Estimating the tensor of curvature of a surface* (ICCV 1995)
- Blender Python API `quadriflow_remesh` —
  https://docs.blender.org/api/current/bpy.ops.object.html

## ライセンス

GPL-3.0-or-later（Blender アドオンの慣例に準拠）。
