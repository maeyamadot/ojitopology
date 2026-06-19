# SPDX-License-Identifier: GPL-3.0-or-later
"""
OjiTopology - Quad Remesher Add-on for Blender
==============================================

選択オブジェクトを、目標頂点数を指定して **四角形ポリゴン主体のきれいな
エッジフロー** へ自動リトポロジー(quad remesh)するアドオン。
ZBrush の ZRemesher のように、ポリグループ（マテリアル境界／頂点グループ／
シャープマーク／境界）をエッジフローのガイドとして使える。

研究・実装の背景（v2）
----------------------
v1 では Blender ネイティブの `quadriflow_remesh` をラップしていたが、それは
「ボタンを押すだけ」でアルゴリズムの中身が見えないブラックボックスであり、
また実験用の独自エンジンも cross field の計算結果を使わず元の三角分割を
そのままなぞるだけで、実際には整列していなかった。これらを踏まえ v2 では
ネイティブ実装への依存をやめ、論文を踏まえた独自実装のみで構成する。

中核アルゴリズムは Jakob et al., "Instant Field-Aligned Meshes"
(SIGGRAPH Asia 2015) の 2 段階フィールド法を踏まえる:

1. Orientation field (4-RoSy cross field)
   各頂点に「エッジが流れるべき方向」を持たせる場。Taubin の曲率テンソル法
   (ICCV 1995) で主曲率方向から初期化し、隣接頂点との 90度対称(4-RoSy)
   マッチングによる局所平滑化(Gauss-Seidel, ランダム順走査)で滑らかにする。
   特徴線(シャープ/境界/マテリアル境界/頂点グループ境界)上の頂点は、その
   特徴線のタンジェント方向に固定し、ZRemesher の Polygroups と同様に
   エッジフローのガイドとして機能させる。

2. Position field
   v1 最大の欠陥はここが無かったこと。各頂点を、自分自身の orientation
   field が張る局所座標系上の格子点へ「歩み寄らせる」反復解法
   (Instant Meshes 論文 Sec.4 の position field)。収束後、近い格子点に
   集まった頂点同士を Union-Find でクラスタリングし、それぞれのクラスタを
   新しい 1 頂点として採用する。これにより実際に **頂点位置そのものが
   field に沿った格子へ移動する**ため、辺の流れが本当に整列する。

3. Quad 抽出
   上記でできた「格子に整列した三角メッシュ」の上で、隣接する三角形ペアを
   field への整列度・正方性・平面性でスコアリングし、貪欲に選んで共有辺を
   溶解して quad 化する。元の三角分割を直接 quad 化する v1 と異なり、
   ここでは整列済みの土台メッシュを使うため意味のある quad 化になる。

前処理として、入力密度と目標密度の比が大きい場合に position field の
反復解法がエイリアシング(局所的に自己整合するが大域的には間違った格子へ
収束する現象)を起こすことを実験で確認した。Blender 標準の Voxel Remesh は
密度正規化に使えるが、境界・マテリアル・頂点グループなどのデータを破棄して
しまうため、その代わりに **bmesh ネイティブの isotropic remeshing**
(Botsch & Kobbelt 2004 の辺分割/収縮による等方化)を実装し、特徴線を保護
しながら密度だけを正規化する。
"""

bl_info = {
    "name": "OjiTopology Quad Remesher",
    "author": "OjiTopology",
    "version": (2, 0, 0),
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


# ----------------------------------------------------------------------------
# 設定 (PropertyGroup)
# ----------------------------------------------------------------------------
class OjiTopologySettings(PropertyGroup):
    target_verts: IntProperty(
        name="目標頂点数",
        description="リトポロジー後のおおよその頂点数",
        default=2000,
        min=8,
        soft_max=200000,
        max=10000000,
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
        description="orientation field (cross field) を滑らかにする反復回数",
        default=20,
        min=1,
        max=300,
    )
    field_use_curvature: BoolProperty(
        name="曲率に整列",
        description="主曲率方向で場を初期化し、形状の特徴に沿わせる",
        default=True,
    )
    position_iterations: IntProperty(
        name="位置場の反復回数",
        description="position field (頂点を格子点へ歩み寄らせる解法) の反復回数",
        default=30,
        min=1,
        max=300,
    )
    cluster_tolerance: FloatProperty(
        name="クラスタリング許容度",
        description="この比率(目標辺長に対する割合)より近い頂点を1つの格子点に統合する",
        default=0.35,
        min=0.1,
        max=0.9,
    )
    resample_iterations: IntProperty(
        name="密度正規化の反復回数",
        description="位置場を解く前に密度を目標解像度へ正規化する"
                    "(辺の分割/収縮)反復回数",
        default=6,
        min=1,
        max=30,
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


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


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
# 特徴線(ポリグループ境界)検出
# ----------------------------------------------------------------------------
def _detect_feature_edges(bm, obj, settings):
    """シャープ/角度/境界/マテリアル/頂点グループから特徴エッジを検出する。

    戻り値: (feature_pairs, feature_vert)
      feature_pairs: {(i, j), (j, i), ...} 特徴エッジの両端頂点インデックス組
      feature_vert : bytearray(len(bm.verts)) 各頂点が特徴線上かどうか
    """
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
            deform_layer = bm.verts.layers.deform.verify()

    def in_group(v):
        if deform_layer is None:
            return False
        w = v[deform_layer].get(vgroup_index, 0.0)
        return w >= 0.5

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
    """目標辺長 h に近づけて密度を正規化する。特徴エッジは保護する。

    長い辺を分割し、短い辺(特徴線でないもの)を収縮することを繰り返す。
    Position field の反復解法は、入力密度と目標密度の比が大きいと
    エイリアシング(局所的には整合するが大域的に間違った格子へ収束する現象)
    を起こすため、その前段で密度を揃える目的の処理。
    """
    long_len = h * 1.35
    short_len = h * 0.55

    for _ in range(iterations):
        bm.edges.ensure_lookup_table()
        long_edges = [e for e in bm.edges if e.calc_length() > long_len]
        if long_edges:
            bmesh.ops.subdivide_edges(
                bm, edges=long_edges, cuts=1, use_grid_fill=True,
            )
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

        bm.edges.ensure_lookup_table()
        short_edges = []
        for e in bm.edges:
            if e.calc_length() >= short_len:
                continue
            if e.is_boundary:
                continue
            i, j = e.verts[0].index, e.verts[1].index
            if (i, j) in feature_pairs:
                continue
            short_edges.append(e)
        if short_edges:
            bmesh.ops.collapse(bm, edges=short_edges, uvs=True)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

        if not long_edges and not short_edges:
            break

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()


# ----------------------------------------------------------------------------
# 1) Orientation field (4-RoSy cross field)
# ----------------------------------------------------------------------------
def _compute_curvature_dirs(verts, vnormals, neighbors):
    """各頂点の主曲率(最小曲率)方向を Taubin 法で推定する。

    Taubin, "Estimating the tensor of curvature of a surface from a
    polyhedral approximation" (ICCV 1995) に基づく簡易版。
    """
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
        if nrm > 1e-9:
            tangents[i] = acc / nrm
        else:
            tangents[i] = ref
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
# 2) Position field — v1 で欠落していた核心部分
# ----------------------------------------------------------------------------
def _build_position_field(verts, vnormals, field, neighbors, feature_vert,
                           h, iterations, rng):
    """各頂点を、自身の局所座標系上の格子点へ歩み寄らせる反復解法。

    Instant Meshes 論文 Sec.4 の position field を簡略化したもの。
    各頂点 i の局所直交基底 (field[i], binormal[i]) を作り、隣接頂点 j との
    オフセットをその基底上で h の整数倍へスナップした目標位置へ向けて
    Gauss-Seidel 的に更新する。特徴線上の頂点は固定する。
    """
    n = len(verts)
    binormal = np.cross(vnormals, field)
    p = verts.copy()

    for _ in range(iterations):
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if feature_vert[i]:
                continue
            oi = field[i]
            bi = binormal[i]
            ni = vnormals[i]
            acc = np.zeros(3)
            w = 0
            for j in neighbors[i]:
                d = p[j] - p[i]
                u = float(np.dot(d, oi))
                v = float(np.dot(d, bi))
                ru = round(u / h) * h
                rv = round(v / h) * h
                target = p[j] - (ru * oi + rv * bi)
                acc += target
                w += 1
            if w == 0:
                continue
            newp = acc / w
            disp = newp - verts[i]
            disp = disp - ni * float(np.dot(disp, ni))  # 接平面内のみ許可
            p[i] = verts[i] + disp

    return p


# ----------------------------------------------------------------------------
# クラスタリング (格子点へ収束した頂点群を1つの新頂点へ統合)
# ----------------------------------------------------------------------------
def _cluster_positions(p, vnormals, edge_list, feature_pairs, tol, feature_vert):
    n = len(p)
    uf = _UnionFind(n)
    for (i, j) in edge_list:
        if feature_vert[i] or feature_vert[j]:
            continue
        if (i, j) in feature_pairs:
            continue
        if np.linalg.norm(p[i] - p[j]) < tol:
            uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    remap = {}
    cluster_id = [0] * n
    members = []
    for i, r in enumerate(roots):
        if r not in remap:
            remap[r] = len(members)
            members.append([])
        cid = remap[r]
        cluster_id[i] = cid
        members[cid].append(i)

    new_positions = np.array([np.mean(p[idxs], axis=0) for idxs in members])
    new_normals = np.zeros((len(members), 3))
    for cid, idxs in enumerate(members):
        m = np.mean(vnormals[idxs], axis=0)
        nrm = np.linalg.norm(m)
        new_normals[cid] = m / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])

    return cluster_id, new_positions, new_normals, members


def _build_base_faces(triangles, cluster_id):
    """クラスタリング後の頂点で、元の三角形位相を引き継いだ土台メッシュを作る。"""
    seen = set()
    faces = []
    for (a, b, c) in triangles:
        ca, cb, cc = cluster_id[a], cluster_id[b], cluster_id[c]
        if ca == cb or cb == cc or ca == cc:
            continue
        key = frozenset((ca, cb, cc))
        if key in seen:
            continue
        seen.add(key)
        faces.append((ca, cb, cc))
    return faces


def _cluster_field(field, members, new_normals):
    """クラスタごとの代表 field ベクトル(メンバーの場を RoSy マッチして平均)。"""
    cf = np.zeros((len(members), 3))
    for cid, idxs in enumerate(members):
        ni = new_normals[cid]
        ref = _tangent_project(field[idxs[0]], ni)
        if ref is None:
            ref = np.array([1.0, 0.0, 0.0])
        acc = ref.copy()
        for vi in idxs[1:]:
            dj = _tangent_project(field[vi], ni)
            if dj is None:
                continue
            acc = acc + _rosy_best_match(dj, acc, ni)
        t = _tangent_project(acc, ni)
        cf[cid] = t if t is not None else ref
    return cf


# ----------------------------------------------------------------------------
# 3) Quad 抽出 — 整列済みの土台メッシュ上で三角形ペアを field 基準でマージ
# ----------------------------------------------------------------------------
def _quad_merge(bm, positions, field, locked_pairs):
    bm.normal_update()
    candidates = []
    for e in bm.edges:
        i, j = e.verts[0].index, e.verts[1].index
        if frozenset((i, j)) in locked_pairs:
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

        P0, P1, P2, P3 = (
            positions[s0.index], positions[a.index],
            positions[s1.index], positions[b.index],
        )
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


# ----------------------------------------------------------------------------
# エンジン本体
# ----------------------------------------------------------------------------
def engine_field_aligned(obj, settings):
    if not _HAS_NUMPY:
        raise RuntimeError("このエンジンには numpy が必要です")

    mesh = obj.data

    src_bm = bmesh.new()
    src_bm.from_mesh(mesh)
    bmesh.ops.triangulate(src_bm, faces=src_bm.faces[:])
    src_bm.normal_update()

    area = sum(f.calc_area() for f in src_bm.faces)
    if area < 1e-12 or len(src_bm.verts) < 4:
        src_bm.free()
        raise RuntimeError("面のあるメッシュを選択してください")

    n_target = max(8, settings.target_verts)
    h = math.sqrt(area / n_target)
    h = max(h, 1e-6)

    # --- 特徴線検出 (密度正規化の前。短辺収縮で保護するため) -------------------
    feature_pairs, _ = _detect_feature_edges(src_bm, obj, settings)

    # --- 密度正規化 (isotropic remeshing) -----------------------------------
    _isotropic_resample(src_bm, h, feature_pairs, settings.resample_iterations)
    src_bm.verts.ensure_lookup_table()
    src_bm.faces.ensure_lookup_table()
    src_bm.verts.index_update()

    n = len(src_bm.verts)
    if n < 4:
        src_bm.free()
        raise RuntimeError("密度正規化の結果が空になりました。目標頂点数を見直してください")

    # 密度正規化でトポロジが変わったため、特徴線を現在の状態で再検出する
    feature_pairs, feature_vert_arr = _detect_feature_edges(src_bm, obj, settings)
    feature_vert = np.array(feature_vert_arr, dtype=bool)

    verts = np.array([v.co[:] for v in src_bm.verts], dtype=np.float64)
    vnormals = _np_normalize(np.array([v.normal[:] for v in src_bm.verts], dtype=np.float64))
    neighbors = [[] for _ in range(n)]
    edge_list = []
    for e in src_bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        neighbors[a].append(b)
        neighbors[b].append(a)
        edge_list.append((a, b))
    triangles = [tuple(v.index for v in f.verts) for f in src_bm.faces if len(f.verts) == 3]
    src_bm.free()

    rng = random.Random(settings.seed)
    tangents = _feature_tangents(verts, feature_pairs, n)

    # --- 1) Orientation field -------------------------------------------------
    field = _build_field(
        verts, vnormals, neighbors,
        iterations=settings.field_iterations,
        use_curvature=settings.field_use_curvature,
        feature_vert=feature_vert,
        tangents=tangents,
        rng=rng,
    )

    # --- 2) Position field + クラスタリング (目標頂点数に近づくよう h を調整) --
    members = None
    new_positions = new_normals = base_faces = None
    cur_h = h
    for attempt in range(4):
        p = _build_position_field(
            verts, vnormals, field, neighbors, feature_vert,
            cur_h, settings.position_iterations, rng,
        )
        tol = cur_h * settings.cluster_tolerance
        cluster_id, cand_positions, cand_normals, cand_members = _cluster_positions(
            p, vnormals, edge_list, feature_pairs, tol, feature_vert,
        )
        actual = len(cand_positions)
        if actual < 4:
            cur_h *= 0.6
            continue
        ratio = actual / n_target
        new_positions, new_normals, members = cand_positions, cand_normals, cand_members
        base_faces = _build_base_faces(triangles, cluster_id)
        if 0.4 <= ratio <= 2.5 or attempt == 3:
            break
        cur_h *= math.sqrt(ratio)

    if new_positions is None or len(new_positions) < 4:
        raise RuntimeError("リメッシュに失敗しました。目標頂点数を見直してください")

    locked_pairs = {frozenset((cluster_id[i], cluster_id[j])) for (i, j) in feature_pairs}
    cf = _cluster_field(field, members, new_normals)

    # --- 3) Quad 抽出 ----------------------------------------------------------
    new_bm = bmesh.new()
    bm_verts = [new_bm.verts.new(tuple(co)) for co in new_positions]
    new_bm.verts.ensure_lookup_table()

    made = set()
    for (a, b, c) in base_faces:
        key = frozenset((a, b, c))
        if key in made:
            continue
        try:
            new_bm.faces.new((bm_verts[a], bm_verts[b], bm_verts[c]))
            made.add(key)
        except ValueError:
            continue

    new_bm.normal_update()
    _quad_merge(new_bm, new_positions, cf, locked_pairs)

    bmesh.ops.dissolve_degenerate(new_bm, dist=1e-6, edges=new_bm.edges[:])
    new_bm.normal_update()

    new_bm.to_mesh(obj.data)
    new_bm.free()
    obj.data.update()

    return "Field-Aligned"


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

    mod = obj.modifiers.new(name="OjiMirror", type='MIRROR')
    mod.use_axis = ('X' in axes, 'Y' in axes, 'Z' in axes)
    mod.use_clip = True
    bpy.ops.object.modifier_apply(modifier=mod.name)


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
            engine_name = engine_field_aligned(obj, settings)
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
        box3.prop(settings, "position_iterations")
        box3.prop(settings, "cluster_tolerance")
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
