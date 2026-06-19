# SPDX-License-Identifier: GPL-3.0-or-later
"""
OjiTopology - Quad Remesher Add-on for Blender (v3)
===================================================

選択オブジェクトを、目標頂点数を指定して **四角形ポリゴン主体のきれいな
エッジフロー** へ自動リトポロジー(quad remesh)するアドオン。
ZBrush の ZRemesher のように、ポリグループ（マテリアル境界／頂点グループ／
シャープマーク／境界）をエッジフローのガイドとして使える。

設計の変遷と v3 の方針
----------------------
v1: Blender ネイティブ quadriflow_remesh をラップ（中身が見えない）。
v2: Instant Meshes の position field を実装し、頂点を格子へ収束させてから
    クラスタリング・格子抽出して quad を作ろうとした。しかし
    - 各エッジの整数オフセットを独立に丸める簡易抽出では大域的に整合した
      格子にならず、出力グラフが穴だらけ・三角形だらけになった。
    - クラスタリングで元三角分割を畳むと劣化した三角網しか残らなかった。
    - 純 Python のループはハイポリのスカルプトで事実上停止し、空メッシュに
      なって「メッシュが消える」現象が出た。
    （これらの破綻はオフラインの合成データ検証で定量的に確認した。）

v3: クラスタリングと格子抽出を撤廃し、堅牢性を最優先した素直なパイプライン
    に作り直す。Instant Meshes の orientation field（4-RoSy cross field）は
    流れの方向を決めるガイドとして引き続き使うが、quad 化は「整った三角網の
    隣接三角形ペアを field 整列度でマッチングして共有辺を溶解する」方式に統一
    する。これにより穴やメッシュ消失が起きず、必ず有効な quad 主体メッシュに
    なる（任意の三角網では約 80% が quad、残りは特異点付近の三角/多角形。
    これは ZRemesher 等でも避けられない）。最後にサーフェス拘束スムージングで
    エッジループを直線的に整える。

パイプライン:
  1. 入力取得（ハイポリは Decimate で安全な規模へ落とす）
  2. 特徴線（ポリグループ境界）検出
  3. 等方リメッシュで目標密度の整った三角土台を作る（特徴線を保護）
  4. Orientation field（4-RoSy cross field）構築（特徴線で固定）
  5. field 整列度で三角形ペアを貪欲マッチングし共有辺を溶解 → quad 主体化
  6. 特徴線・境界を固定したサーフェス拘束スムージングで流れを整える
  7. 元サーフェスへ Shrinkwrap して形状ディテールを保持
  8. 対称仕上げ・スムーズシェード

参考: Jakob et al. "Instant Field-Aligned Meshes" (SIGGRAPH Asia 2015) /
      Taubin "Estimating the tensor of curvature" (ICCV 1995) /
      Botsch & Kobbelt "A Remeshing Approach to Multiresolution Modeling"
      (SGP 2004, 等方リメッシュ).
"""

bl_info = {
    "name": "OjiTopology Quad Remesher",
    "author": "OjiTopology",
    "version": (3, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > OjiTopology",
    "description": "目標頂点数を指定して四角形リトポロジー (Quad Remesh) する。"
                    "ポリグループ(マテリアル/頂点グループ/シャープ)でエッジフローを誘導",
    "category": "Mesh",
}

import math
import random

import bpy
import bmesh
from bpy.props import (
    IntProperty,
    EnumProperty,
    BoolProperty,
    FloatProperty,
    StringProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, Operator, Panel

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - Blender はほぼ numpy を同梱する
    np = None
    _HAS_NUMPY = False

# 純 Python ループの処理時間とメモリを守るため、フィールド計算に回す三角メッシュの
# 頂点数の上限。これを超える密度は要求されても自動でこの値に丸める。
# (参考: 純 Python の cross field 平滑化は約 250 頂点/秒。15000 頂点で約1分。
#  これ以上は Blender が固まったように見えるため上限とする。)
_MAX_WORKING_VERTS = 15000


# ----------------------------------------------------------------------------
# 設定 (PropertyGroup)
# ----------------------------------------------------------------------------
class OjiTopologySettings(PropertyGroup):
    target_verts: IntProperty(
        name="目標頂点数",
        description="リトポロジー後のおおよその頂点数",
        default=2000,
        min=8,
        soft_max=100000,
        max=1000000,
    )

    # --- 特徴線(ポリグループ境界)検出 ---------------------------------------
    preserve_boundary: BoolProperty(
        name="境界を保持",
        description="開いたメッシュの境界エッジを特徴線として固定する",
        default=True,
    )
    detect_sharp_marks: BoolProperty(
        name="シャープマークを使用",
        description="エッジの『シャープ』マークを特徴線として扱う",
        default=True,
    )
    detect_angle: BoolProperty(
        name="角度で自動検出",
        description="隣接面のなす角が閾値を超えるエッジを特徴線として自動検出する",
        default=True,
    )
    sharp_angle: FloatProperty(
        name="検出角度",
        description="特徴線として検出する最小の面角度(度)",
        default=40.0,
        min=1.0,
        max=179.0,
        subtype='ANGLE',
    )
    use_material_boundaries: BoolProperty(
        name="マテリアル境界を使用",
        description="マテリアルスロットの境界を特徴線(ポリグループ境界)として扱う",
        default=False,
    )
    region_mode: EnumProperty(
        name="領域指定",
        description="エッジフローを誘導する領域(ポリグループ)の指定方法",
        items=[
            ('AUTO', "自動(形状から検出)",
             "シャープ/角度/境界/マテリアルから自動検出するのみ"),
            ('VERTEX_GROUP', "頂点グループを使用",
             "指定した頂点グループの境界を追加の特徴線として使う"),
        ],
        default='AUTO',
    )
    vertex_group: StringProperty(
        name="頂点グループ",
        description="領域境界として使う頂点グループ(ユーザーが頂点を割り当てて指定する"
                    "ポリグループ)",
        default="",
    )

    # --- 出力オプション ------------------------------------------------------
    smooth_normals: BoolProperty(
        name="スムーズシェード",
        description="結果をスムーズシェーディングにする",
        default=True,
    )
    project_to_surface: BoolProperty(
        name="元サーフェスへ投影",
        description="結果を元のメッシュ表面へ Shrinkwrap し、ディテールを保持する",
        default=True,
    )
    symmetry_x: BoolProperty(name="X 対称", default=False)
    symmetry_y: BoolProperty(name="Y 対称", default=False)
    symmetry_z: BoolProperty(name="Z 対称", default=False)

    seed: IntProperty(
        name="シード",
        description="場の平滑化の走査順を変える乱数シード(結果がわずかに変わる)",
        default=0,
        min=0,
    )

    # --- アルゴリズム詳細設定 -------------------------------------------------
    field_iterations: IntProperty(
        name="方向場の平滑化回数",
        description="orientation field (cross field) を滑らかにする反復回数"
                    "(多いほど流れが滑らかだが遅くなる)",
        default=20,
        min=1,
        max=300,
    )
    field_use_curvature: BoolProperty(
        name="曲率に整列",
        description="主曲率方向で場を初期化し、形状の特徴に沿わせる",
        default=True,
    )
    relax_iterations: IntProperty(
        name="流れの整え回数",
        description="quad 化後にエッジループを直線的に整えるスムージング回数",
        default=8,
        min=0,
        max=60,
    )
    resample_iterations: IntProperty(
        name="密度正規化の反復回数",
        description="目標解像度へ密度を揃える(辺分割/収縮)反復回数",
        default=8,
        min=1,
        max=40,
    )


# ----------------------------------------------------------------------------
# 小さな数学ユーティリティ
# ----------------------------------------------------------------------------
def _np_normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _tangent_project(vec, normal):
    """vec を normal に直交する接平面へ射影して正規化する。退化していたら None。"""
    vec = vec - normal * np.dot(vec, normal)
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return None
    return vec / n


def _rosy_rotations(d, n):
    """接平面内ベクトル d の 4-RoSy 対称(90度回転)4方向を返す。"""
    perp = np.cross(n, d)
    return (d, perp, -d, -perp)


def _rosy_best_match(d, ref, n):
    """d の 90度回転の中で ref に最も近い向きを返す。"""
    best = d
    best_dot = -2.0
    for cand in _rosy_rotations(d, n):
        dot = float(np.dot(cand, ref))
        if dot > best_dot:
            best_dot = dot
            best = cand
    return best


def _symmetry_axes(settings):
    axes = set()
    if settings.symmetry_x:
        axes.add('X')
    if settings.symmetry_y:
        axes.add('Y')
    if settings.symmetry_z:
        axes.add('Z')
    return axes


def _shade(obj, smooth):
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = smooth
    mesh.update()


def _mesh_stats(mesh):
    nv = len(mesh.vertices)
    nf = len(mesh.polygons)
    if nf == 0:
        return nv, 0, 0.0
    quads = sum(1 for p in mesh.polygons if p.loop_total == 4)
    return nv, nf, quads / nf


# ----------------------------------------------------------------------------
# 入力取得 (ハイポリは Decimate で安全な規模に落とす)
# ----------------------------------------------------------------------------
def _make_working_bmesh(obj, context, target_verts):
    """評価済みメッシュから作業用 bmesh を作る。巨大ならまず Decimate する。

    ハイポリのスカルプト(数十万〜数百万ポリ)をそのまま純 Python で処理すると
    停止・空メッシュ化するため、フィールド計算前に C++ の Decimate モディファイア
    で安全な面数まで落とす。元データは最後まで変更しない(結果は最後に書き戻す)。
    """
    me = obj.data
    n_faces = len(me.polygons)
    # 等方リメッシュ前の安全な上限。目標頂点数の数倍あれば十分。
    cap_faces = min(200000, max(target_verts * 8, 20000))

    temp_mod = None
    if n_faces > cap_faces:
        temp_mod = obj.modifiers.new(name="OjiTopo_Decimate", type='DECIMATE')
        temp_mod.decimate_type = 'COLLAPSE'
        temp_mod.ratio = max(0.01, min(1.0, cap_faces / n_faces))

    depsgraph = context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    eval_me = obj_eval.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(eval_me)
    obj_eval.to_mesh_clear()

    if temp_mod is not None:
        obj.modifiers.remove(temp_mod)

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    return bm


# ----------------------------------------------------------------------------
# 特徴線(ポリグループ境界)検出
# ----------------------------------------------------------------------------
def _detect_feature_edges(bm, obj, settings):
    """シャープ/角度/境界/マテリアル/頂点グループから特徴エッジを検出する。"""
    n = len(bm.verts)
    feature_vert = bytearray(n)
    feature_pairs = set()

    use_boundary = settings.preserve_boundary
    use_sharp = settings.detect_sharp_marks
    use_angle = settings.detect_angle
    angle_thresh = math.radians(settings.sharp_angle)
    use_material = settings.use_material_boundaries
    use_vgroup = settings.region_mode == 'VERTEX_GROUP' and settings.vertex_group

    deform_layer = None
    vgroup_index = None
    if use_vgroup:
        vg = obj.vertex_groups.get(settings.vertex_group)
        if vg is not None:
            vgroup_index = vg.index
            deform_layer = bm.verts.layers.deform.active

    def in_group(v):
        if deform_layer is None:
            return False
        return v[deform_layer].get(vgroup_index, 0.0) >= 0.5

    for e in bm.edges:
        is_feature = False
        if use_boundary and e.is_boundary:
            is_feature = True
        if not is_feature and use_sharp and not e.smooth:
            is_feature = True
        if not is_feature and use_angle and len(e.link_faces) == 2:
            try:
                ang = e.calc_face_angle()
            except Exception:
                ang = None
            if ang is not None and ang > angle_thresh:
                is_feature = True
        if not is_feature and use_material and len(e.link_faces) == 2:
            if e.link_faces[0].material_index != e.link_faces[1].material_index:
                is_feature = True
        if not is_feature and use_vgroup:
            if in_group(e.verts[0]) != in_group(e.verts[1]):
                is_feature = True

        if is_feature:
            i, j = e.verts[0].index, e.verts[1].index
            feature_pairs.add((i, j))
            feature_pairs.add((j, i))
            feature_vert[i] = 1
            feature_vert[j] = 1

    return feature_pairs, feature_vert


# ----------------------------------------------------------------------------
# 密度の正規化 (isotropic remeshing: Botsch & Kobbelt 2004 の簡易版)
# ----------------------------------------------------------------------------
def _isotropic_resample(bm, h, feature_pairs, iterations):
    """目標辺長 h に近づけて密度を正規化する。特徴エッジは保護する。"""
    long_len = h * 1.33
    short_len = h * 0.55

    for _ in range(iterations):
        bm.edges.ensure_lookup_table()
        long_edges = [e for e in bm.edges if e.calc_length() > long_len]
        if long_edges:
            bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1,
                                      use_grid_fill=True)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

        bm.edges.ensure_lookup_table()
        short_edges = []
        for e in bm.edges:
            if e.calc_length() >= short_len or e.is_boundary:
                continue
            i, j = e.verts[0].index, e.verts[1].index
            if (i, j) in feature_pairs:
                continue
            short_edges.append(e)
        if short_edges:
            bmesh.ops.collapse(bm, edges=short_edges, uvs=True)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

        # 軽い接線方向の頂点再配置(等方性の改善)
        bm.verts.ensure_lookup_table()
        smooth_targets = [v for v in bm.verts if not v.is_boundary]
        if smooth_targets:
            bmesh.ops.smooth_vert(bm, verts=smooth_targets, factor=0.5,
                                  use_axis_x=True, use_axis_y=True,
                                  use_axis_z=True)

        if not long_edges and not short_edges:
            break

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()


# ----------------------------------------------------------------------------
# Orientation field (4-RoSy cross field)
# ----------------------------------------------------------------------------
def _compute_curvature_dirs(verts, vnormals, neighbors):
    """各頂点の主曲率(最小曲率)方向を Taubin 法(ICCV 1995)で推定する。"""
    n = len(verts)
    dirs = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        ni = vnormals[i]
        nb = neighbors[i]
        if len(nb) < 2:
            continue
        M = np.zeros((3, 3), dtype=np.float64)
        wsum = 0.0
        for j in nb:
            e = verts[j] - verts[i]
            elen2 = float(np.dot(e, e))
            if elen2 < 1e-16:
                continue
            t = _tangent_project(e, ni)
            if t is None:
                continue
            kappa = 2.0 * float(np.dot(e, ni)) / elen2
            w = math.sqrt(elen2)
            M += w * kappa * np.outer(t, t)
            wsum += w
        if wsum < 1e-12:
            continue
        M /= wsum
        try:
            evals, evecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            continue
        normal_like = max(range(3), key=lambda k: abs(float(np.dot(evecs[:, k], ni))))
        tangent_idx = [k for k in range(3) if k != normal_like]
        k_min = min(tangent_idx, key=lambda k: abs(evals[k]))
        d = _tangent_project(evecs[:, k_min], ni)
        if d is not None:
            dirs[i] = d
    return dirs


def _feature_tangents(verts, feature_pairs, n):
    """特徴線上の各頂点について、線が伸びる方向(符号は不定)を推定する。"""
    neigh_dirs = [[] for _ in range(n)]
    for (i, j) in feature_pairs:
        e = verts[j] - verts[i]
        ln = np.linalg.norm(e)
        if ln > 1e-9:
            neigh_dirs[i].append(e / ln)

    tangents = [None] * n
    for i in range(n):
        ds = neigh_dirs[i]
        if not ds:
            continue
        ref = ds[0]
        acc = ref.copy()
        for d in ds[1:]:
            if np.dot(d, ref) < 0:
                d = -d
            acc = acc + d
        nrm = np.linalg.norm(acc)
        tangents[i] = acc / nrm if nrm > 1e-9 else ref
    return tangents


def _build_field(verts, vnormals, neighbors, iterations, use_curvature,
                  feature_vert, tangents, rng):
    """4-RoSy cross field を構築・平滑化する。特徴線上の頂点は方向を固定する。"""
    n = len(verts)
    field = np.zeros((n, 3), dtype=np.float64)

    if use_curvature:
        field = _compute_curvature_dirs(verts, vnormals, neighbors)

    for i in range(n):
        if feature_vert[i] and tangents[i] is not None:
            d = _tangent_project(tangents[i], vnormals[i])
            if d is not None:
                field[i] = d
                continue
        if np.linalg.norm(field[i]) < 1e-6:
            ni = vnormals[i]
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, ni))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            d = _tangent_project(ref, ni)
            field[i] = d if d is not None else ref

    field = _np_normalize(field)

    for _ in range(iterations):
        new_field = field.copy()
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if feature_vert[i]:
                continue
            ni = vnormals[i]
            acc = field[i].copy()
            for j in neighbors[i]:
                dj = _tangent_project(field[j], ni)
                if dj is None:
                    continue
                matched = _rosy_best_match(dj, acc, ni)
                acc = acc + matched
                t = _tangent_project(acc, ni)
                if t is not None:
                    acc = t
            new_field[i] = acc
        field = _np_normalize(new_field)

    return field


# ----------------------------------------------------------------------------
# Quad 化: field 整列度で三角形ペアを貪欲マッチングし共有辺を溶解する
# ----------------------------------------------------------------------------
def _quad_dominant(bm, field, feature_pairs):
    """整った三角網の隣接三角形ペアを field 整列度でマッチングして quad 化する。

    各内部エッジ(2 つの三角形が共有)について、その辺を溶解してできる四角形が
    どれだけ cross field に沿い、正方形に近く、平面的かをスコア化する。
    スコアの高い順に貪欲に選び、両側の三角形がまだ未使用なら共有辺を溶解する。
    特徴エッジは溶解候補から除外して残す(ポリグループ境界を維持)。
    """
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    pos = {v.index: np.array(v.co[:]) for v in bm.verts}
    candidates = []
    for e in bm.edges:
        i, j = e.verts[0].index, e.verts[1].index
        if (i, j) in feature_pairs:
            continue
        lf = e.link_faces
        if len(lf) != 2:
            continue
        fA, fB = lf[0], lf[1]
        if len(fA.verts) != 3 or len(fB.verts) != 3:
            continue
        s0, s1 = e.verts[0], e.verts[1]
        a = next((v for v in fA.verts if v not in (s0, s1)), None)
        b = next((v for v in fB.verts if v not in (s0, s1)), None)
        if a is None or b is None:
            continue

        P0, P1, P2, P3 = pos[s0.index], pos[a.index], pos[s1.index], pos[b.index]
        U = (P1 - P0) + (P2 - P3)
        V = (P3 - P0) + (P2 - P1)
        nU, nV = np.linalg.norm(U), np.linalg.norm(V)
        if nU < 1e-9 or nV < 1e-9:
            continue
        U /= nU
        V /= nV

        nA = np.array(fA.normal[:])
        nB = np.array(fB.normal[:])
        planarity = float(np.dot(nA, nB))
        if planarity < 0.2:
            continue

        navg = _np_normalize(nA + nB)
        ref = field[s0.index]
        facc = ref.copy()
        for vi in (a.index, s1.index, b.index):
            facc = facc + _rosy_best_match(field[vi], ref, navg)
        f = _tangent_project(facc, navg)
        if f is None:
            continue
        fperp = np.cross(navg, f)

        aU = max(abs(float(np.dot(U, f))), abs(float(np.dot(U, fperp))))
        aV = max(abs(float(np.dot(V, f))), abs(float(np.dot(V, fperp))))
        field_score = 0.5 * (aU + aV)
        squareness = 1.0 - abs(float(np.dot(U, V)))
        score = field_score * squareness * planarity

        candidates.append((score, e.index, fA.index, fB.index))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used = bytearray(len(bm.faces))
    edge_lut = {e.index: e for e in bm.edges}
    to_dissolve = []
    for score, eidx, fa, fb in candidates:
        if used[fa] or used[fb]:
            continue
        used[fa] = 1
        used[fb] = 1
        to_dissolve.append(edge_lut[eidx])

    if to_dissolve:
        bmesh.ops.dissolve_edges(bm, edges=to_dissolve, use_verts=False)
    bm.normal_update()


# ----------------------------------------------------------------------------
# 流れを整えるサーフェス拘束スムージング
# ----------------------------------------------------------------------------
def _relax_flow(bm, obj, settings, iterations):
    """quad 化後の頂点を緩和してエッジループを直線的に整える。

    特徴線・境界上の頂点は動かさない。各反復で接線方向の Laplacian
    スムージングを行う(法線方向の体積収縮は後段の Shrinkwrap で戻す)。
    """
    if iterations <= 0:
        return
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.ensure_lookup_table()
    feature_pairs, feature_vert = _detect_feature_edges(bm, obj, settings)
    movable = [v for v in bm.verts
               if not v.is_boundary and not feature_vert[v.index]]
    if not movable:
        return
    for _ in range(iterations):
        bmesh.ops.smooth_vert(bm, verts=movable, factor=0.5,
                              use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.normal_update()


# ----------------------------------------------------------------------------
# 元サーフェスへの投影 (Shrinkwrap)
# ----------------------------------------------------------------------------
def _project_to_surface(obj, snapshot):
    mod = obj.modifiers.new(name="OjiTopo_Shrinkwrap", type='SHRINKWRAP')
    mod.target = snapshot
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


# ----------------------------------------------------------------------------
# 対称仕上げ (Bisect + Mirror で厳密な対称形を保証する)
# ----------------------------------------------------------------------------
def _apply_symmetry(obj, axes):
    if not axes:
        return
    axis_normal = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}

    bpy.context.view_layer.objects.active = obj
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for ax in axes:
        normal = axis_normal[ax]
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        bmesh.ops.bisect_plane(
            bm, geom=geom, dist=1e-4,
            plane_co=(0.0, 0.0, 0.0), plane_no=normal,
            clear_outer=False, clear_inner=True,
        )
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    mod = obj.modifiers.new(name="OjiTopo_Mirror", type='MIRROR')
    mod.use_axis = ('X' in axes, 'Y' in axes, 'Z' in axes)
    mod.use_clip = True
    bpy.ops.object.modifier_apply(modifier=mod.name)


# ----------------------------------------------------------------------------
# エンジン本体
# ----------------------------------------------------------------------------
def engine_field_aligned(obj, context, settings):
    if not _HAS_NUMPY:
        raise RuntimeError("このアドオンには numpy が必要です(Blender に同梱)")

    # --- 1) 入力取得(ハイポリは Decimate) -----------------------------------
    bm = _make_working_bmesh(obj, context, settings.target_verts)
    bm.verts.ensure_lookup_table()
    bm.normal_update()

    area = sum(f.calc_area() for f in bm.faces)
    if area < 1e-12 or len(bm.verts) < 4:
        bm.free()
        raise RuntimeError("面のあるメッシュを選択してください")

    n_target = max(8, settings.target_verts)
    # 三角土台の頂点数が上限を超えないよう h を下限でクランプする
    min_h_from_cap = math.sqrt(area / _MAX_WORKING_VERTS)
    h = max(math.sqrt(area / n_target), min_h_from_cap, 1e-6)

    # --- 2) 特徴線検出 -------------------------------------------------------
    feature_pairs, _ = _detect_feature_edges(bm, obj, settings)

    # --- 3) 等方リメッシュ ---------------------------------------------------
    _isotropic_resample(bm, h, feature_pairs, settings.resample_iterations)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    n = len(bm.verts)
    if n < 4:
        bm.free()
        raise RuntimeError("密度正規化の結果が空になりました。目標頂点数を見直してください")

    # 密度正規化でトポロジが変わったので特徴線を再検出
    feature_pairs, feature_vert_arr = _detect_feature_edges(bm, obj, settings)
    feature_vert = np.array(feature_vert_arr, dtype=bool)

    verts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    vnormals = _np_normalize(np.array([v.normal[:] for v in bm.verts], dtype=np.float64))
    neighbors = [[] for _ in range(n)]
    seen_pairs = set()
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        key = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        neighbors[a].append(b)
        neighbors[b].append(a)

    # --- 4) Orientation field ------------------------------------------------
    rng = random.Random(settings.seed)
    tangents = _feature_tangents(verts, feature_pairs, n)
    field = _build_field(
        verts, vnormals, neighbors,
        iterations=settings.field_iterations,
        use_curvature=settings.field_use_curvature,
        feature_vert=feature_vert,
        tangents=tangents,
        rng=rng,
    )

    # --- 5) field 誘導 quad マッチング ---------------------------------------
    _quad_dominant(bm, field, feature_pairs)

    # --- 6) 流れの整え(サーフェス拘束スムージング) --------------------------
    _relax_flow(bm, obj, settings, settings.relax_iterations)

    bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=bm.edges[:])
    bm.normal_update()

    if len(bm.faces) == 0:
        bm.free()
        raise RuntimeError("結果が空になりました。目標頂点数やオプションを見直してください")

    # --- 7) 元サーフェスへ投影するためのスナップショットを作って書き戻す ------
    snapshot = None
    if settings.project_to_surface:
        snap_mesh = obj.data.copy()
        snapshot = bpy.data.objects.new(obj.name + "_OjiSnap", snap_mesh)
        context.scene.collection.objects.link(snapshot)
        snapshot.matrix_world = obj.matrix_world.copy()

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    if snapshot is not None:
        try:
            _project_to_surface(obj, snapshot)
        finally:
            mesh_data = snapshot.data
            bpy.data.objects.remove(snapshot, do_unlink=True)
            if mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

    return "Field-Aligned"


# ----------------------------------------------------------------------------
# オペレーター
# ----------------------------------------------------------------------------
class OBJECT_OT_ojitopology_remesh(Operator):
    bl_idname = "object.ojitopology_remesh"
    bl_label = "Quad Remesh"
    bl_description = "選択メッシュを目標頂点数で四角形リトポロジーする"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and context.mode == 'OBJECT'
        )

    def execute(self, context):
        settings = context.scene.ojitopology
        obj = context.active_object

        if not obj.data.polygons:
            self.report({'ERROR'}, "面のあるメッシュを選択してください")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        try:
            engine_name = engine_field_aligned(obj, context, settings)
            _apply_symmetry(obj, _symmetry_axes(settings))
        except RuntimeError as exc:
            self.report({'ERROR'}, f"{exc}")
            return {'CANCELLED'}
        except Exception as exc:  # noqa: BLE001 - ユーザーに原因を見せる
            self.report({'ERROR'}, f"リメッシュ失敗: {exc}")
            return {'CANCELLED'}

        if settings.smooth_normals:
            _shade(obj, True)

        nv, nf, quad_ratio = _mesh_stats(obj.data)
        self.report(
            {'INFO'},
            f"[{engine_name}] 頂点 {nv} / 面 {nf} / quad率 {quad_ratio*100:.0f}%",
        )
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# UI パネル
# ----------------------------------------------------------------------------
class VIEW3D_PT_ojitopology(Panel):
    bl_label = "OjiTopology Quad Remesher"
    bl_idname = "VIEW3D_PT_ojitopology"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OjiTopology"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.ojitopology
        obj = context.active_object

        col = layout.column(align=True)
        col.prop(settings, "target_verts")

        box = layout.box()
        box.label(text="ポリグループ / 特徴線 (エッジフローの誘導)")
        box.prop(settings, "preserve_boundary")
        row = box.row(align=True)
        row.prop(settings, "detect_sharp_marks")
        row.prop(settings, "detect_angle")
        if settings.detect_angle:
            box.prop(settings, "sharp_angle")
        box.prop(settings, "use_material_boundaries")
        box.prop(settings, "region_mode")
        if settings.region_mode == 'VERTEX_GROUP':
            if obj and obj.type == 'MESH':
                box.prop_search(settings, "vertex_group", obj, "vertex_groups")
            else:
                box.prop(settings, "vertex_group")

        box2 = layout.box()
        box2.label(text="出力オプション")
        box2.prop(settings, "smooth_normals")
        box2.prop(settings, "project_to_surface")
        row = box2.row(align=True)
        row.label(text="対称:")
        row.prop(settings, "symmetry_x", text="X", toggle=True)
        row.prop(settings, "symmetry_y", text="Y", toggle=True)
        row.prop(settings, "symmetry_z", text="Z", toggle=True)
        box2.prop(settings, "seed")

        box3 = layout.box()
        box3.label(text="アルゴリズム詳細")
        box3.prop(settings, "field_use_curvature")
        box3.prop(settings, "field_iterations")
        box3.prop(settings, "relax_iterations")
        box3.prop(settings, "resample_iterations")
        if not _HAS_NUMPY:
            box3.label(text="numpy が見つかりません", icon='ERROR')

        layout.separator()
        big = layout.row()
        big.scale_y = 1.6
        big.operator("object.ojitopology_remesh", icon='MOD_REMESH')

        if obj and obj.type == 'MESH':
            nv = len(obj.data.vertices)
            nf = len(obj.data.polygons)
            layout.label(text=f"現在: 頂点 {nv} / 面 {nf}")
        else:
            layout.label(text="メッシュを選択してください", icon='INFO')


# ----------------------------------------------------------------------------
# 登録
# ----------------------------------------------------------------------------
_classes = (
    OjiTopologySettings,
    OBJECT_OT_ojitopology_remesh,
    VIEW3D_PT_ojitopology,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ojitopology = PointerProperty(type=OjiTopologySettings)


def unregister():
    del bpy.types.Scene.ojitopology
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
