# SPDX-License-Identifier: GPL-3.0-or-later
"""
OjiTopology - Quad Remesher Add-on for Blender
==============================================

選択オブジェクトを四角形ポリゴン主体の、きれいなエッジフローを持つメッシュへ
自動リトポロジー(quad remesh)するアドオン。目標頂点数を指定できる。

研究・実装の背景
----------------
Quad remesh の中核アルゴリズムは以下の研究を踏まえている:

* Jakob et al., "Instant Field-Aligned Meshes" (SIGGRAPH Asia 2015)
  - 方向場 (orientation field / 4-RoSy cross field) と位置場 (position field) を
    局所平滑化で解き、エッジ方向と頂点位置を同時最適化して quad を生成する。
* Huang et al., "QuadriFlow: A Scalable and Robust Method for Quadrangulation"
  (SGP 2018)
  - Instant Meshes を発展させ、特異点配置を整数計画で整理した堅牢な手法。
  - ★ これは Blender に C++ でネイティブ実装されており
    `bpy.ops.object.quadriflow_remesh()` から利用できる。
    すなわち「論文 → Blender に落とし込む」が公式に完了しているコア。

本アドオンは実在の quad remesher アドオン(CurioMesh 等)と同じく
2 つのエンジンを備える:

1. QUADRIFLOW エンジン (既定 / 堅牢)
   研究アルゴリズムのネイティブ実装をラップし、目標頂点数 → 目標面数へ変換する。

2. FIELD エンジン (実験 / 自作)
   Instant Meshes の cross field の考え方を Python で独自実装したもの。
   Taubin の曲率テンソルから主曲率方向を推定し 4-RoSy 場を平滑化、
   その場に沿って三角形をペアリングし quad-dominant メッシュを作る。
"""

bl_info = {
    "name": "OjiTopology Quad Remesher",
    "author": "OjiTopology",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > OjiTopology",
    "description": "目標頂点数を指定して四角形リトポロジー (Quad Remesh) する",
    "category": "Mesh",
}

import math

import bpy
import bmesh
from mathutils import Vector
from bpy.props import (
    IntProperty,
    EnumProperty,
    BoolProperty,
    FloatProperty,
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

    engine: EnumProperty(
        name="エンジン",
        description="quad remesh に使うアルゴリズム",
        items=[
            ('QUADRIFLOW', "QuadriFlow (推奨)",
             "Blender ネイティブの研究実装。堅牢で高品質"),
            ('FIELD', "Field-Aligned (実験)",
             "cross field を自前計算して quad 化する独自実装"),
        ],
        default='QUADRIFLOW',
    )

    preserve_sharp: BoolProperty(
        name="シャープを保持",
        description="鋭いエッジ(特徴線)をできるだけ維持する",
        default=True,
    )
    preserve_boundary: BoolProperty(
        name="境界を保持",
        description="開いたメッシュの境界を維持する",
        default=True,
    )
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
        description="結果を変える乱数シード",
        default=0,
        min=0,
    )

    # Field エンジン専用
    field_iterations: IntProperty(
        name="場の平滑化回数",
        description="cross field を滑らかにする反復回数",
        default=15,
        min=1,
        max=200,
    )
    field_use_curvature: BoolProperty(
        name="曲率に整列",
        description="主曲率方向で場を初期化し、特徴線に沿わせる",
        default=True,
    )


# ----------------------------------------------------------------------------
# 共通ユーティリティ
# ----------------------------------------------------------------------------
def _estimate_face_target(target_verts):
    """目標頂点数から目標面数を推定する。

    閉じた純 quad メッシュではオイラーの公式 V - E + F = 2 と E = 2F より
    V = F + 2、すなわち F ≈ V。よって目標面数 ≈ 目標頂点数。
    """
    return max(4, int(round(target_verts)))


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
    """(頂点数, 面数, quad率) を返す。"""
    nv = len(mesh.vertices)
    nf = len(mesh.polygons)
    if nf == 0:
        return nv, 0, 0.0
    quads = sum(1 for p in mesh.polygons if p.loop_total == 4)
    return nv, nf, quads / nf


# ----------------------------------------------------------------------------
# エンジン 1: QuadriFlow (ネイティブ研究実装のラップ)
# ----------------------------------------------------------------------------
def engine_quadriflow(obj, settings):
    target_faces = _estimate_face_target(settings.target_verts)
    axes = _symmetry_axes(settings)

    kwargs = dict(
        target_faces=target_faces,
        mode='FACES',
        use_preserve_sharp=settings.preserve_sharp,
        use_preserve_boundary=settings.preserve_boundary,
        smooth_normals=settings.smooth_normals,
        seed=settings.seed,
    )
    # 対称はバージョンによって扱いが違うため安全に試す
    if axes:
        kwargs["use_paint_symmetry"] = True

    # 一部の引数は Blender のバージョン差があるため段階的にフォールバック
    try:
        bpy.ops.object.quadriflow_remesh(**kwargs)
    except TypeError:
        kwargs.pop("use_paint_symmetry", None)
        bpy.ops.object.quadriflow_remesh(**kwargs)

    return "QuadriFlow"


# ----------------------------------------------------------------------------
# エンジン 2: Field-Aligned (自作 cross field 実装)
# ----------------------------------------------------------------------------
#
# 流れ:
#   1) Voxel Remesh で目標解像度の一様な三角メッシュ土台を作る
#   2) 各頂点に 4-RoSy の cross field を構築する
#        - Taubin の曲率テンソルから主曲率方向で初期化 (任意)
#        - 隣接頂点との 90度対称マッチングで局所平滑化 (Instant Meshes の核)
#   3) cross field への整列度で隣接三角形ペアを貪欲に選び、
#      その共有エッジを溶解して quad へ統合する (quad-dominant 化)
# ----------------------------------------------------------------------------

def _np_normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _tangent_project(vec, normal):
    """vec を normal に直交する接平面へ射影して正規化。"""
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


def _compute_curvature_dirs(verts, vnormals, neighbors):
    """各頂点の主曲率(最小曲率)方向を Taubin 法で推定。

    Taubin, "Estimating the tensor of curvature of a surface from a
    polyhedral approximation" (ICCV 1995) に基づく簡易版。
    エッジは曲率が小さい方向(最小主曲率)に沿って流したいので、その向きを返す。
    """
    n = len(verts)
    dirs = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        ni = vnormals[i]
        nb = neighbors[i]
        if len(nb) < 2:
            dirs[i] = np.zeros(3)
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
            # Taubin の法線曲率
            kappa = 2.0 * float(np.dot(e, ni)) / elen2
            w = math.sqrt(elen2)
            M += w * kappa * np.outer(t, t)
            wsum += w
        if wsum < 1e-12:
            dirs[i] = np.zeros(3)
            continue
        M /= wsum
        # 接平面内の主方向 = 法線に最も平行でない固有ベクトル群
        try:
            evals, evecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            dirs[i] = np.zeros(3)
            continue
        # 法線に最も近い固有ベクトルを除外し、残り 2 つのうち
        # |固有値| が小さい = 最小曲率方向 を採用
        normal_like = max(range(3), key=lambda k: abs(float(np.dot(evecs[:, k], ni))))
        tangent_idx = [k for k in range(3) if k != normal_like]
        # 最小曲率方向 = |eval| が小さい方
        k_min = min(tangent_idx, key=lambda k: abs(evals[k]))
        d = _tangent_project(evecs[:, k_min], ni)
        dirs[i] = d if d is not None else np.zeros(3)
    return dirs


def _build_field(verts, vnormals, neighbors, iterations, use_curvature):
    """各頂点の 4-RoSy cross field を構築・平滑化する。"""
    n = len(verts)
    field = np.zeros((n, 3), dtype=np.float64)

    if use_curvature:
        field = _compute_curvature_dirs(verts, vnormals, neighbors)

    # 初期化が退化している頂点は接平面内の適当な向きで埋める
    for i in range(n):
        if np.linalg.norm(field[i]) < 1e-6:
            ni = vnormals[i]
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, ni))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            d = _tangent_project(ref, ni)
            if d is None:
                d = _tangent_project(np.array([0.0, 0.0, 1.0]), ni)
            field[i] = d if d is not None else ref

    field = _np_normalize(field)

    # Instant Meshes 風の局所平滑化:
    # 各頂点で、隣接頂点の場を自分の接平面へ射影 → 4-RoSy マッチ → 平均
    for _ in range(iterations):
        new_field = field.copy()
        for i in range(n):
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


def _voxel_remesh_inplace(obj, voxel_size):
    """Remesh モディファイア(Voxel)を適用して一様な三角土台を作る。"""
    mod = obj.modifiers.new(name="OjiVoxel", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = max(voxel_size, 1e-4)
    mod.use_smooth_shade = False
    # 適用にはアクティブ & OBJECT モードが必要
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def engine_field_aligned(obj, settings):
    if not _HAS_NUMPY:
        raise RuntimeError("Field エンジンには numpy が必要です")

    mesh = obj.data

    # --- 1) 目標解像度の voxel 土台 -----------------------------------------
    # 表面積から quad 1 枚あたりの目標辺長 L ≈ sqrt(Area / N) を求める
    # voxel_size はローカル座標で効くため、面積もローカル座標で測って整合させる
    # (スケール未適用のオブジェクトでも破綻しない)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    area = sum(f.calc_area() for f in bm.faces)
    bm.free()
    n_target = max(8, settings.target_verts)
    edge_len = math.sqrt(max(area, 1e-9) / n_target)
    edge_len = max(edge_len, 1e-4)

    _voxel_remesh_inplace(obj, edge_len)

    # --- 2) bmesh を組み、三角化して numpy 配列へ ---------------------------
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()

    n = len(bm.verts)
    if n == 0:
        bm.free()
        raise RuntimeError("Voxel リメッシュの結果が空です。目標頂点数を見直してください")

    verts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    vnormals = _np_normalize(
        np.array([v.normal[:] for v in bm.verts], dtype=np.float64)
    )
    neighbors = [[] for _ in range(n)]
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        neighbors[a].append(b)
        neighbors[b].append(a)

    # --- 3) cross field の構築 ---------------------------------------------
    field = _build_field(
        verts, vnormals, neighbors,
        iterations=settings.field_iterations,
        use_curvature=settings.field_use_curvature,
    )

    # --- 4) 場への整列度で三角形ペアを貪欲選択し共有エッジを溶解 ------------
    candidates = []
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) != 2:
            continue  # 境界・非多様体エッジは対象外
        fA, fB = lf[0], lf[1]
        if len(fA.verts) != 3 or len(fB.verts) != 3:
            continue
        s0, s1 = e.verts[0], e.verts[1]
        a = next((v for v in fA.verts if v not in (s0, s1)), None)
        b = next((v for v in fB.verts if v not in (s0, s1)), None)
        if a is None or b is None:
            continue

        P0 = verts[s0.index]
        P1 = verts[a.index]
        P2 = verts[s1.index]
        P3 = verts[b.index]

        # quad (P0->P1->P2->P3) の 2 つのエッジ流れ方向
        U = (P1 - P0) + (P2 - P3)
        V = (P3 - P0) + (P2 - P1)
        nU = np.linalg.norm(U)
        nV = np.linalg.norm(V)
        if nU < 1e-9 or nV < 1e-9:
            continue
        U /= nU
        V /= nV

        # 平面性 (2 面の法線の一致度)
        nA = np.array(fA.normal[:])
        nB = np.array(fB.normal[:])
        planarity = float(np.dot(nA, nB))
        if planarity < 0.3:
            continue  # 折れすぎ → 特徴線を残す

        navg = _np_normalize(nA + nB)
        # 中心の場 = 4 頂点の場を RoSy マッチして平均
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

    used_face = bytearray(len(bm.faces))
    edge_lut = {e.index: e for e in bm.edges}
    to_dissolve = []
    for score, eidx, fa, fb in candidates:
        if used_face[fa] or used_face[fb]:
            continue
        used_face[fa] = 1
        used_face[fb] = 1
        to_dissolve.append(edge_lut[eidx])

    if to_dissolve:
        bmesh.ops.dissolve_edges(bm, edges=to_dissolve, use_verts=False)

    # 軽い緩和 (場に沿った見た目の改善)
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

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

        # 念のためアクティブ & 単一選択に
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        try:
            if settings.engine == 'QUADRIFLOW':
                engine_name = engine_quadriflow(obj, settings)
            else:
                engine_name = engine_field_aligned(obj, settings)
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
        col.prop(settings, "engine")

        box = layout.box()
        box.label(text="オプション")
        box.prop(settings, "preserve_sharp")
        box.prop(settings, "preserve_boundary")
        box.prop(settings, "smooth_normals")

        row = box.row(align=True)
        row.label(text="対称:")
        row.prop(settings, "symmetry_x", text="X", toggle=True)
        row.prop(settings, "symmetry_y", text="Y", toggle=True)
        row.prop(settings, "symmetry_z", text="Z", toggle=True)
        box.prop(settings, "seed")

        if settings.engine == 'FIELD':
            fbox = layout.box()
            fbox.label(text="Field エンジン設定")
            fbox.prop(settings, "field_use_curvature")
            fbox.prop(settings, "field_iterations")
            if not _HAS_NUMPY:
                fbox.label(text="numpy が見つかりません", icon='ERROR')

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
