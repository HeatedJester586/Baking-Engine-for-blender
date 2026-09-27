# SPDX-License-Identifier: GPL-3.0-or-later
"""
Headless smoke test for PBR Atlas Baker.

Builds a small multi-material mesh that exercises the tricky cases (ORM
channel packing, metal, tiling through a Mapping node, rotated and mirrored
normal-mapped islands, floating geometry, an empty slot), bakes it with the
GPU rasteriser and with Cycles, and checks that the two methods agree.
Cycles is treated as the reference.

Run with either:
    blender -b --factory-startup --python tests/smoke_test.py
    python tests/smoke_test.py          # with the `bpy` module from PyPI
"""

import os
import sys
import tempfile

import bpy
import bmesh
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pbr_atlas_baker"))
import baker  # noqa: E402

RES = 256
TMP = tempfile.mkdtemp(prefix="pbr_atlas_test_")


def init_gpu_headless():
    """In background mode the GPU module only works once something has
    created a GPU context; a tiny EEVEE render does exactly that."""
    if not bpy.app.background:
        return
    scene = bpy.context.scene
    engine = scene.render.engine
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = scene.render.resolution_y = 8
    bpy.ops.render.render()
    scene.render.engine = engine


def make_image(name, pixels, non_color):
    h, w, _ = pixels.shape
    img = bpy.data.images.new(name, w, h, alpha=False)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(pixels.astype(np.float32).ravel())
    path = os.path.join(TMP, name + ".png")
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    img.source = 'FILE'
    img.filepath = path
    img.reload()
    return img


def new_material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat, mat.node_tree, next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')


def build_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # Textures ------------------------------------------------------------
    checker = np.zeros((32, 32, 4), np.float32)
    yy, xx = np.mgrid[0:32, 0:32]
    on = ((xx // 8 + yy // 8) % 2).astype(bool)
    checker[..., :3] = np.where(on[..., None], (0.8, 0.5, 0.2), (0.25, 0.12, 0.05))
    checker[..., 3] = 1
    img_wood = make_image("wood_BaseColor", checker, non_color=False)

    orm = np.ones((16, 16, 4), np.float32)
    orm[..., 0], orm[..., 1], orm[..., 2] = 0.3, 0.65, 1.0   # AO, Roughness, Metallic
    img_orm = make_image("TEX_OcclusionRoughnessMetallic", orm, non_color=True)

    nrm = np.ones((16, 16, 4), np.float32)
    nrm[..., :3] = (0.75, 0.5, 0.933)  # tilted towards +U
    img_nrm = make_image("carbon_Normal", nrm, non_color=True)

    # Materials -----------------------------------------------------------
    wood, tree, bsdf = new_material("Wood")
    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.image = img_wood
    mapping = tree.nodes.new('ShaderNodeMapping')
    mapping.inputs["Scale"].default_value = (3.0, 3.0, 1.0)
    coord = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7

    keytar, tree, bsdf = new_material("MAT_Keytar")
    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.image = img_orm
    sep = tree.nodes.new('ShaderNodeSeparateColor')
    tree.links.new(tex.outputs["Color"], sep.inputs["Color"])
    tree.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    mix = tree.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs["Factor"].default_value = 1.0
    _input = {s.identifier: s for s in mix.inputs}
    _input["A_Color"].default_value = (0.6, 0.1, 0.1, 1.0)
    _input["B_Color"].default_value = (0.5, 1.0, 1.0, 1.0)
    tree.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    strings, tree, bsdf = new_material("strings_metal")
    bsdf.inputs["Base Color"].default_value = (0.9, 0.85, 0.7, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.3

    carbon, tree, bsdf = new_material("carbon_fiber")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1.0)
    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.image = img_nrm
    nmap = tree.nodes.new('ShaderNodeNormalMap')
    tree.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    tree.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    # Mesh ----------------------------------------------------------------
    mesh = bpy.data.meshes.new("guitar")
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    def quad(corners, uvs, mat_index):
        verts = [bm.verts.new(c) for c in corners]
        face = bm.faces.new(verts)
        face.material_index = mat_index
        for loop, uv in zip(face.loops, uvs):
            loop[uv_layer].uv = uv
        return face

    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad([(-2, -1, 0), (0, -1, 0), (0, 1, 0), (-2, 1, 0)], square, 0)            # wood body
    quad([(0, -1, 0), (1, -1, 0), (1, 1, 0), (0, 1, 0)], square, 1)              # ORM panel
    quad([(-2, -0.02, 0.002), (1, -0.02, 0.002), (1, 0.02, 0.002), (-2, 0.02, 0.002)],
         square, 2)                                                              # floating string
    rotated = [(1, 0), (1, 1), (0, 1), (0, 0)]                                   # UV rotated 90 deg
    quad([(1.2, -1, 0), (2.2, -1, 0), (2.2, 0, 0), (1.2, 0, 0)], rotated, 3)
    mirrored = [(1, 0), (0, 0), (0, 1), (1, 1)]                                  # UV mirrored in U
    quad([(1.2, 0.2, 0), (2.2, 0.2, 0), (2.2, 1.2, 0), (1.2, 1.2, 0)], mirrored, 3)
    quad([(-2, 1.2, 0), (-1, 1.2, 0), (-1, 2.2, 0), (-2, 2.2, 0)], square, 4)    # empty slot
    bm.to_mesh(mesh)
    bm.free()
    for i in range(7):  # fill Blender's 8 UV map limit with unused maps, like many game rips
        mesh.uv_layers.new(name=f"UV_unused_{i}")

    obj = bpy.data.objects.new("guitar.036", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for mat in (wood, keytar, strings, carbon, None):
        mesh.materials.append(mat)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def read_pixels(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    arr = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(arr)
    bpy.data.images.remove(img)
    return arr.reshape(h, w, 4)


def check(cond, message):
    print(("PASS  " if cond else "FAIL  ") + message)
    return bool(cond)


def main():
    init_gpu_headless()
    source = build_scene()
    ok = True

    report = baker.diagnose(source)
    print(report)
    ok &= check("[G]" in report and "[B]" in report, "ORM texture resolved to G/B channels")
    ok &= check("empty slot" in report, "empty slot reported")

    gpu_result = baker.bake_gpu_atlas(source, resolution=RES, margin=4,
                                      output_dir=os.path.join(TMP, "gpu"), hide_source=False)
    gpu = {ch: read_pixels(p) for ch, p in gpu_result["paths"].items()}
    target = gpu_result["object"]
    ok &= check(len(target.material_slots) == 1, "atlas object has one material")
    ok &= check([l.name for l in target.data.uv_layers] == [baker.BAKE_UV_NAME],
                "atlas object keeps only the atlas UV map")
    ok &= check(len(source.data.uv_layers) == 8, "original object keeps all 8 UV maps")
    ok &= check(gpu_result["coverage"] > 20.0, f"UV coverage {gpu_result['coverage']:.1f}%")

    cyc_result = baker.bake_cycles_atlas(source, resolution=RES, margin=4, samples=4,
                                         output_dir=os.path.join(TMP, "cycles"), hide_source=False)
    cyc = {ch: read_pixels(p) for ch, p in cyc_result["paths"].items()}

    # Compare only baked pixels: Cycles leaves the empty background black,
    # the GPU path fills it with a neutral value. Texture filtering differs
    # slightly at hard edges (mipmapped bilinear vs. jittered Cycles samples),
    # so the check uses the median (catches colour-space / channel mistakes,
    # which shift every pixel) and the mean (catches wrong islands).
    baked = cyc["BaseColor"][..., :3].max(axis=2) > 0.0
    for ch in baker.CHANNELS:
        diff = np.abs(gpu[ch][..., :3] - cyc[ch][..., :3]).max(axis=2)[baked] * 255.0
        median, mean = np.median(diff), diff.mean()
        ok &= check(median <= 1.0 and mean <= 6.0,
                    f"{ch:<10} GPU vs Cycles: median {median:.2f}, mean {mean:.2f} (8-bit levels)")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    print("Output:", TMP)
    return ok


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
