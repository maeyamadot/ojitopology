# OjiTopology — Blender Quad Remesher Add-on (v2)

選択したオブジェクトを **四角形ポリゴン主体のきれいなエッジフロー** へ自動
リトポロジー（quad remesh）する Blender アドオン。**目標頂点数を指定** して
ワンクリックで実行できる。ZBrush の ZRemesher のように、マテリアル境界や
頂点グループでユーザーが「ポリグループ」を指定し、エッジフローを誘導できる。

## v2 での変更点（v1 からの反省）

v1 ではアルゴリズムの中身が見えないネイティブの `quadriflow_remesh` を
ラップしたエンジンと、独自実装の「Field-Aligned」エンジンの 2 つを用意して
いたが、後者は **頂点位置を実際に field に沿った格子へ動かすステップ
（position field）が無く**、元の三角分割を field でただ選んで dissolve
していただけだったため、見た目上 quad に見えても整列していなかった。

v2 では:

- ネイティブのラップ（QuadriFlow エンジン）を**完全に削除**し、独自実装の
  みで構成する。
- 欠落していた **position field** のステップを実装し、頂点位置そのものを
  cross field 上の格子点へ歩み寄らせてからクラスタリングし、本当に
  field に整列した土台メッシュの上で quad 抽出するようにした。
- ポリグループ（マテリアル境界／頂点グループ／シャープマーク／角度／境界）
  をエッジフローのガイド・固定境界として扱えるようにした。

## できること

- 目標頂点数を指定したクリーンな四角形リトポロジー
- N パネル（サイドバー）の UI から大きなボタン一発で実行
- ポリグループ指定によるエッジフロー誘導（自動検出 or 頂点グループ）
- シャープ／境界／マテリアル境界の保持、スムーズシェード、X/Y/Z 対称

## インストール

1. Blender を開く
2. `Edit > Preferences > Add-ons`
3. `Install...`（4.2 以降は `Install from Disk...`）から `ojitopology.py` を選択
4. 一覧の **OjiTopology Quad Remesher** にチェックを入れて有効化

> Blender 3.0 以降で動作。Blender 同梱の numpy を使用する。

## 使い方

1. メッシュオブジェクトを選択（オブジェクトモード）
2. 3D ビューポートで `N` キー → **OjiTopology** タブ
3. **目標頂点数** を入力
4. 必要なら **ポリグループ / 特徴線** の設定でエッジフローを誘導:
   - 「角度で自動検出」: 鋭いエッジ（角など）を自動でエッジフローのガイドに
   - 「頂点グループを使用」: 事前にウェイトペイントで頂点グループを作っておけば
     その境界（自分で指定するポリグループ）に沿ってエッジフローが揃う
5. **Quad Remesh** ボタンを押す
6. 完了後、ヘッダに `頂点数 / 面数 / quad率` が表示される

## アルゴリズム（研究背景）

ZRemesher のような「ポリグループに沿った整列済み quad メッシュ」を目標に、
中核アルゴリズムは以下の研究を踏まえて Python で独自実装している
（Blender ネイティブの実装には依存しない）。

### 1. Orientation field（方向場 / 4-RoSy cross field）

Jakob et al., *“Instant Field-Aligned Meshes”* (SIGGRAPH Asia 2015) の
考え方に基づき、各頂点に「エッジが流れるべき方向」を持たせる場を構築する。

- Taubin, *“Estimating the tensor of curvature of a surface from a
  polyhedral approximation”* (ICCV 1995) の曲率テンソル法で主曲率（最小曲率）
  方向を推定し初期化。曲率の小さい方向にエッジを流すことで自然な流れになる。
- 隣接頂点の場を接平面へ射影し 90°対称（4-RoSy）マッチングして平均する
  局所平滑化（Instant Meshes の中核操作）を、ランダムな走査順で
  Gauss-Seidel 的に反復する（論文が指摘する通り、ランダム順の方が収束が速い）。
- **特徴線（ポリグループ境界）上の頂点は、その線の接線方向に固定**する。
  これにより、マテリアル境界や頂点グループの輪郭、シャープエッジに沿って
  エッジフローが揃う（ZRemesher の Polygroups と同じ役割）。

### 2. Position field（位置場）— v1 で欠落していた核心部分

Orientation field だけでは「どの辺をどう繋ぐか」しか決まらず、頂点位置は
元のままなので整列した見た目にならない。Instant Meshes 論文 Sec.4 の
position field を簡略化して実装し、**各頂点を自分の局所座標系
（orientation field とその直交方向が張る基底）上の格子点へ歩み寄らせる**
反復解法を行う。収束後、近い格子点に集まった頂点を Union-Find でクラスタ
リングし、それぞれを新しい 1 頂点として採用する。特徴線上の頂点は固定し、
特徴エッジは異なるクラスタとして保持されるため、ポリグループの輪郭が
リトポロジー後も消えない。

### 3. 密度の正規化（isotropic remeshing）

Position field の反復解法は、入力メッシュの密度と目標密度の比が大きいと
局所的には自己整合するが大域的には間違った格子へ収束する
「エイリアシング」を起こすことを実験で確認した（`/tmp` でのオフライン検証）。
Blender 標準の Voxel Remesh は密度正規化に使えるが、境界・マテリアル・
頂点グループの情報を破棄してしまうため、代わりに **bmesh ネイティブの
isotropic remeshing**（Botsch & Kobbelt, 2004 の辺分割／収縮による等方化）
を実装し、特徴エッジを保護しながら密度だけを目標解像度へ正規化する。

### 4. Quad 抽出

密度正規化 → orientation field → position field → クラスタリングを経て
「field に整列した土台メッシュ」ができた上で、隣接する三角形ペアを
field への整列度・正方性・平面性でスコアリングし、貪欲に選んで共有辺を
溶解して quad 化する。

### 5. 対称仕上げ

X/Y/Z 対称を指定した場合は、Bisect Plane で片側を切り落としてから
Mirror モディファイアを適用し、厳密な対称形を保証する。

## 既知の限界

- Position field の反復解法・クラスタリングは合成データ（平面格子・クレース
  付きメッシュ）でオフライン検証済みだが、実際の Blender 上での bmesh 操作
  （isotropic remeshing・quad dissolve・Mirror 適用）は環境上実行できず、
  実機での見た目の確認はまだ行っていない。複雑な形状や非常に薄い形状では
  追加の調整が必要な場合がある。
- マテリアル割り当てや UV は現状出力メッシュへ引き継がれない。

## 参考文献・ソース

- Instant Field-Aligned Meshes — https://igl.ethz.ch/projects/instant-meshes/
- Instant Meshes (実装) — https://github.com/wjakob/instant-meshes
- QuadriFlow（参考、本アドオンでは現在使用していない） —
  https://www.researchgate.net/publication/326463904
- Taubin, *Estimating the tensor of curvature of a surface* (ICCV 1995)
- Botsch & Kobbelt, *A Remeshing Approach to Multiresolution Modeling*
  (SGP 2004) — isotropic remeshing（辺分割/収縮による密度正規化）

## ライセンス

GPL-3.0-or-later（Blender アドオンの慣例に準拠）。
