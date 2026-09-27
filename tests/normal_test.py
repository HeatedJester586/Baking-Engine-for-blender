# SPDX-License-Identifier: GPL-3.0-or-later
"""
Normal-map ground truth test for PBR Atlas Baker.

Cycles bakes the real shading normal (object space) of the original mesh and
of the atlas copy; they must match. Covers a normal texture wired straight
into the BSDF's Normal input (Blender uses its colour as a world-space
direction) and a normal texture going through a Normal Map node, on a rotated
cube whose faces point every way.

Run with either:
    blender -b --factory-startup --python tests/normal_test.py
    python tests/normal_test.py          # with the `bpy` module from PyPI
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pbr_atlas_baker"))
import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
import baker  # noqa: E402
import smoke_test  # noqa: E402
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
rng = np.random.default_rng(3)
def smooth_img(name, base, amp, non_color):
    y, x = np.mgrid[0:64, 0:64] / 64.0
    px = np.zeros((64, 64, 4)); px[..., 3] = 1
    for c in range(3):
        px[..., c] = base[c] + amp * np.sin(6.28 * (x * (c + 1) + y * (2 - c)))
    return smoke_test.make_image(name, np.clip(np.rint(px * 255), 0, 255).astype(np.uint8), non_color)
# A: normal texture wired straight into Normal, left on sRGB (like the guitar)
img_a = smooth_img("unwired_Normal", (0.5, 0.5, 0.9), 0.15, False)
mat_a, tree, bsdf = smoke_test.new_material("unwired")
tree.links.new(smoke_test.image_node(tree, img_a).outputs["Color"], bsdf.inputs["Normal"])
# B: proper Normal Map node
img_b = smooth_img("proper_Normal", (0.5, 0.5, 0.9), 0.2, True)
mat_b, tree, bsdf = smoke_test.new_material("proper")
nm = tree.nodes.new('ShaderNodeNormalMap')
tree.links.new(smoke_test.image_node(tree, img_b).outputs["Color"], nm.inputs["Color"])
tree.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
# cube: every face gets the full 0..1 UV square, rotated differently per face
mesh = bpy.data.meshes.new("cube"); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=2)
uvl = bm.loops.layers.uv.new("UVMap")
for i, f in enumerate(bm.faces):
    f.material_index = i % 2
    for k, loop in enumerate(f.loops):
        cu, cv = (i % 3) / 3.0, (i // 3) / 2.0   # own tile per face, no overlap
        u, v = [(0, 0), (1, 0), (1, 1), (0, 1)][(k + i) % 4]
        loop[uvl].uv = (cu + 0.02 + u * 0.29, cv + 0.02 + v * 0.46)
bm.to_mesh(mesh); bm.free()
obj = bpy.data.objects.new("cube", mesh); bpy.context.scene.collection.objects.link(obj)
mesh.materials.append(mat_a); mesh.materials.append(mat_b)
obj.rotation_euler = (0.4, 0.2, 0.7)   # not axis-aligned: world != object space
bpy.context.view_layer.update()
r = baker.build_atlas(obj, padding=4, hide_source=False, log=lambda m: None)
atlas = r["object"]

scene = bpy.context.scene; scene.render.engine = 'CYCLES'; scene.cycles.samples = 1
def bake(o, uv_name):
    img = bpy.data.images.new("bake_" + o.name, 256, 256, float_buffer=True)
    for slot in o.material_slots:
        n = slot.material.node_tree.nodes.new('ShaderNodeTexImage'); n.image = img
        slot.material.node_tree.nodes.active = n
    for x in bpy.context.selected_objects: x.select_set(False)
    o.hide_set(False); o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.bake(type='NORMAL', normal_space='OBJECT', uv_layer=uv_name, margin=0)
    a = np.empty(256 * 256 * 4, np.float32); img.pixels.foreach_get(a)
    return a.reshape(256, 256, 4)[..., :3] * 2 - 1
src = bake(obj, "UVMap"); dst = bake(atlas, baker.ATLAS_UV_NAME)
old = baker._uv_array(obj.data, "UVMap"); new = baker._uv_array(atlas.data, baker.ATLAS_UV_NAME)
errs = {0: [], 1: []}
for f in obj.data.polygons:
    L = list(f.loop_indices)
    for w in ([.3, .3, .2, .2], [.1, .4, .3, .2], [.25, .15, .35, .25], [.4, .2, .1, .3]):
        w = np.array(w); a = src[tuple(np.floor(w @ old[L] * 256).astype(int)[::-1])]
        b = dst[tuple(np.floor(w @ new[L] * 256).astype(int)[::-1])]
        cos = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
        errs[f.material_index].append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
ok = True
for index, label in ((0, "straight into Normal"), (1, "through a Normal Map node")):
    mean, worst = np.mean(errs[index]), np.max(errs[index])
    good = mean < 3.0 and worst < 6.0
    ok &= good
    print(("PASS  " if good else "FAIL  ")
          + f"normal texture {label}: atlas vs Blender mean {mean:.2f} deg, max {worst:.2f} deg")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
if not ok:
    sys.exit(1)
