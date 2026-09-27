# SPDX-License-Identifier: GPL-3.0-or-later
"""
Headless test for PBR Atlas Baker.

Builds a multi-material mesh with random-noise textures that covers the
tricky cases (ORM channel packing, duplicated materials, a texture only
partly used, tiling through a Mapping node, a rotated normal-mapped island,
alpha, emission, flat-colour materials, an empty slot and 8 UV maps), builds
the atlas and checks that, for every face, the atlas texel under the new UV
is byte-for-byte the source texel under the old UV.

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

TMP = tempfile.mkdtemp(prefix="pbr_atlas_test_")
RNG = np.random.default_rng(7)
PADDING = 4


def noise(size, gray=False):
    px = RNG.integers(0, 256, (size, size, 4), dtype=np.uint8)
    if gray:
        px[..., 1] = px[..., 2] = px[..., 0]
    px[..., 3] = 255
    return px


def make_image(name, pixels, non_color):
    h, w, _ = pixels.shape
    img = bpy.data.images.new(name, w, h, alpha=False)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set((pixels.astype(np.float32) / 255.0).ravel())
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
    tree = mat.node_tree
    return mat, tree, next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')


def image_node(tree, image):
    node = tree.nodes.new('ShaderNodeTexImage')
    node.image = image
    return node


def orm_material(name, base_img, orm_img):
    mat, tree, bsdf = new_material(name)
    tree.links.new(image_node(tree, base_img).outputs["Color"], bsdf.inputs["Base Color"])
    sep = tree.nodes.new('ShaderNodeSeparateColor')
    tree.links.new(image_node(tree, orm_img).outputs["Color"], sep.inputs["Color"])
    tree.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    return mat


def build_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    px = {
        "wood": noise(32), "keytar": noise(32), "orm": noise(32), "normal": noise(32),
        "alpha": noise(32, gray=True), "big": noise(64), "tiled": noise(16),
        "mixed_base": noise(32),
    }
    ramp = np.zeros((16, 16, 4), np.uint8)
    ramp[..., :3] = (np.arange(16) * 16)[None, :, None]  # smooth left-to-right gradient
    ramp[..., 3] = 255
    px["ramp"] = ramp
    img = {
        "wood": make_image("wood_BaseColor", px["wood"], False),
        "keytar": make_image("TEX_Keytar_BaseColor.063", px["keytar"], False),
        "keytar_copy": make_image("TEX_Keytar_BaseColor.064", px["keytar"], False),
        "orm": make_image("TEX_Keytar_ORM.063", px["orm"], True),
        "orm_copy": make_image("TEX_Keytar_ORM.064", px["orm"], True),
        "normal": make_image("carbon_Normal", px["normal"], True),
        "alpha": make_image("carbon_Alpha", px["alpha"], True),
        "big": make_image("decal_big", px["big"], False),
        "tiled": make_image("tiled", px["tiled"], False),
        "mixed_base": make_image("mixed_BaseColor", px["mixed_base"], False),
        "ramp": make_image("mixed_Roughness_16px", px["ramp"], True),
    }

    wood, tree, bsdf = new_material("Wood")
    tree.links.new(image_node(tree, img["wood"]).outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7

    keytar_a = orm_material("MAT_Keytar.063", img["keytar"], img["orm"])
    keytar_b = orm_material("MAT_Keytar.064", img["keytar_copy"], img["orm_copy"])

    strings, tree, bsdf = new_material("strings_metal")
    bsdf.inputs["Base Color"].default_value = (0.9, 0.85, 0.7, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.2, 0.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 2.0

    carbon, tree, bsdf = new_material("carbon_fiber")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1.0)
    nmap = tree.nodes.new('ShaderNodeNormalMap')
    tree.links.new(image_node(tree, img["normal"]).outputs["Color"], nmap.inputs["Color"])
    tree.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    tree.links.new(image_node(tree, img["alpha"]).outputs["Color"], bsdf.inputs["Alpha"])

    partial, tree, bsdf = new_material("decal_partial")
    tree.links.new(image_node(tree, img["big"]).outputs["Color"], bsdf.inputs["Base Color"])

    tiled, tree, bsdf = new_material("tiled")
    tex = image_node(tree, img["tiled"])
    mapping = tree.nodes.new('ShaderNodeMapping')
    mapping.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
    coord = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    mixed, tree, bsdf = new_material("mixed_resolution")  # 32 px colour + 16 px roughness
    tree.links.new(image_node(tree, img["mixed_base"]).outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(image_node(tree, img["ramp"]).outputs["Color"], bsdf.inputs["Roughness"])

    mesh = bpy.data.meshes.new("guitar")
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    faces = [  # (material index, UVs)
        (0, square),
        (0, [(1, 0), (0, 0), (0, 1), (1, 1)]),       # mirrored island sharing the texture
        (1, square),
        (2, square),
        (3, square),
        (4, [(1, 0), (1, 1), (0, 1), (0, 0)]),       # UV island rotated 90 degrees
        (5, [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),  # uses a quarter of the texture
        (6, square),
        (7, square),
        (8, square),
    ]
    for i, (mat_index, uvs) in enumerate(faces):
        x = i * 1.5
        verts = [bm.verts.new(c) for c in ((x, 0, 0), (x + 1, 0, 0), (x + 1, 1, 0), (x, 1, 0))]
        face = bm.faces.new(verts)
        face.material_index = mat_index
        for loop, uv in zip(face.loops, uvs):
            loop[uv_layer].uv = uv
    bm.to_mesh(mesh)
    bm.free()
    for i in range(7):  # fill Blender's 8 UV map limit, like many game rips
        mesh.uv_layers.new(name=f"UV_unused_{i}")

    obj = bpy.data.objects.new("guitar.036", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for mat in (wood, keytar_a, keytar_b, strings, carbon, partial, tiled, None, mixed):
        mesh.materials.append(mat)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    srgb = lambda v: int(round(float(baker._linear_to_srgb(np.float64(v))) * 255))
    lin = lambda v: int(round(v * 255))
    flat_normal = (128, 128, 255)
    # Expected texels per material: ("tex", pixels, channels, uv scale) or a constant.
    expect = {
        0: {"BaseColor": ("tex", px["wood"], [0, 1, 2], 1), "Roughness": lin(0.7),
            "Metallic": 0, "Normal": flat_normal, "Alpha": 255},
        1: {"BaseColor": ("tex", px["keytar"], [0, 1, 2], 1), "Roughness": ("tex", px["orm"], [1], 1),
            "Metallic": ("tex", px["orm"], [2], 1), "Normal": flat_normal, "Alpha": 255},
        3: {"BaseColor": tuple(srgb(v) for v in (0.9, 0.85, 0.7)), "Roughness": lin(0.3),
            "Metallic": 255, "Normal": flat_normal, "Alpha": 255,
            "Emission": tuple(srgb(v) for v in (1.0, 0.2, 0.0))},
        4: {"BaseColor": (srgb(0.05),) * 3, "Normal": ("tex", px["normal"], [0, 1, 2], 1),
            "Alpha": ("tex", px["alpha"], [0], 1)},
        5: {"BaseColor": ("tex", px["big"], [0, 1, 2], 1)},
        6: {"BaseColor": ("tex", px["tiled"], [0, 1, 2], 2)},
        7: {"BaseColor": (srgb(0.8),) * 3, "Roughness": lin(0.5), "Metallic": 0,
            "Normal": flat_normal, "Alpha": 255, "Emission": (0, 0, 0)},
    }
    expect[2] = expect[1]
    # The 16 px roughness is upsampled onto the 32 px colour grid (bilinear), so it
    # is compared against a smooth sample of the source with a small tolerance.
    expect[8] = {"BaseColor": ("tex", px["mixed_base"], [0, 1, 2], 1),
                 "Roughness": ("smooth", px["ramp"], [0], 1)}
    return obj, expect


def atlas_pixels(image):
    w, h = image.size
    arr = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(arr)
    return np.rint(arr.reshape(h, w, 4) * 255).astype(np.int64)


def check(cond, message):
    print(("PASS  " if cond else "FAIL  ") + message)
    return bool(cond)


def main():
    source, expect = build_scene()
    ok = True

    report = baker.diagnose(source)
    print(report)
    ok &= check("[G]" in report and "[B]" in report, "ORM texture resolved to G/B channels")

    old_uv = baker._uv_array(source.data, "UVMap").astype(np.float64)
    result = baker.build_atlas(source, max_size=2048, padding=PADDING, hide_source=False)
    target, size = result["object"], result["size"]
    new_uv = baker._uv_array(target.data, baker.ATLAS_UV_NAME).astype(np.float64)
    atlases = {name: atlas_pixels(img) for name, img in result["images"].items()}
    output_of = {"BaseColor": ("BaseColor", [0, 1, 2]), "Alpha": ("BaseColor", [3]),
                 "Roughness": ("Roughness", [0]), "Metallic": ("Metallic", [0]),
                 "Normal": ("Normal", [0, 1, 2]), "Emission": ("Emission", [0, 1, 2])}

    # Texel-for-texel check at interior points of every face.
    weights = np.array([[0.31, 0.27, 0.23, 0.19], [0.13, 0.41, 0.17, 0.29],
                        [0.22, 0.14, 0.38, 0.26], [0.36, 0.18, 0.11, 0.35]])
    mesh = source.data
    mismatches = []
    for face in mesh.polygons:
        loops = list(face.loop_indices)
        for w in weights:
            src_uv = w @ old_uv[loops]
            dst_uv = w @ new_uv[loops]
            ax, ay = np.floor(dst_uv * size).astype(int)
            for channel, want in expect[face.material_index].items():
                out, comps = output_of[channel]
                got = tuple(atlases[out][ay, ax, comps])
                if isinstance(want, tuple) and want and want[0] == "smooth":
                    _, pixels, chans, _ = want
                    th, tw = pixels.shape[:2]
                    atlas_centre = (np.array([ax, ay]) + 0.5) / size
                    # map the atlas texel centre back to the source UV of this face
                    a = new_uv[loops[1]] - new_uv[loops[0]], new_uv[loops[3]] - new_uv[loops[0]]
                    b = old_uv[loops[1]] - old_uv[loops[0]], old_uv[loops[3]] - old_uv[loops[0]]
                    coef = np.linalg.solve(np.array(a).T, atlas_centre - new_uv[loops[0]])
                    uv = old_uv[loops[0]] + np.array(b).T @ coef
                    x = np.clip(uv[0] * tw - 0.5, 0, tw - 1)
                    x0 = int(np.floor(x)); x1 = min(x0 + 1, tw - 1); f = x - x0
                    ref = pixels[0, x0, chans] * (1 - f) + pixels[0, x1, chans] * f
                    if np.abs(np.array(got) - ref).max() > 2:
                        mismatches.append((face.index, channel, got, tuple(np.round(ref, 1))))
                    continue
                if isinstance(want, tuple) and want and want[0] == "tex":
                    _, pixels, chans, scale = want
                    th, tw = pixels.shape[:2]
                    sx, sy = np.floor(src_uv * scale * (tw, th)).astype(int)
                    want = tuple(pixels[sy % th, sx % tw, chans])
                want = want if isinstance(want, tuple) else (want,) * len(comps)
                if got != want:
                    mismatches.append((face.index, channel, got, want))
    ok &= check(not mismatches, f"every face samples exactly its original texels "
                                f"({len(mismatches)} mismatches)")
    for m in mismatches[:8]:
        print("      face %d %s: got %s want %s" % m)

    # UV islands are moved, never rescaled: texel density is unchanged.
    textured = {0: 32, 1: 32, 2: 32, 4: 32, 5: 64, 6: 32, 8: 32}
    dens_ok = True
    for face in mesh.polygons:
        if face.material_index in textured:
            loops = list(face.loop_indices)
            src_len = np.linalg.norm(old_uv[loops[1]] - old_uv[loops[0]]) * textured[face.material_index]
            dst_len = np.linalg.norm(new_uv[loops[1]] - new_uv[loops[0]]) * size
            dens_ok &= abs(src_len - dst_len) < 1e-3
    ok &= check(dens_ok, "texel density preserved (UV islands not resized)")

    ok &= check(result["groups"] == 8, f"duplicate materials merged ({result['groups']} unique of 9 slots)")
    ok &= check(result["flat_chunks"] >= 2, f"flat-colour materials shrunk ({result['flat_chunks']} flat chunks)")
    ok &= check(size == 128, f"smallest atlas that fits was picked ({size} x {size}, "
                             f"{result['filled']:.0%} filled)")
    ok &= check(len(source.data.uv_layers) == 8 and len(source.material_slots) == 9,
                "original object untouched")
    ok &= check(len(target.material_slots) == 1 and
                [l.name for l in target.data.uv_layers] == [baker.ATLAS_UV_NAME],
                "atlas object has one material and one UV map")

    mat = target.material_slots[0].material
    bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    feed = lambda name: bsdf.inputs[name].links[0].from_node if bsdf.inputs[name].is_linked else None
    images = result["images"]
    ok &= check(feed("Base Color").image == images["BaseColor"]
                and feed("Alpha").image == images["BaseColor"]
                and feed("Roughness").image == images["Roughness"]
                and feed("Metallic").image == images["Metallic"]
                and feed("Normal").inputs["Color"].links[0].from_node.image == images["Normal"]
                and feed("Emission Color").image == images["Emission"]
                and bsdf.inputs["Emission Strength"].default_value == 2.0,
                "atlases plugged into the Principled BSDF")
    ok &= check(not result["paths"] and all(i.packed_file for i in images.values()),
                "textures stored inside the .blend, no files written")

    names = {channel: image.name for channel, image in images.items()}
    blend = os.path.join(TMP, "reopen_test.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.wm.open_mainfile(filepath=blend)
    same = all(np.array_equal(atlas_pixels(bpy.data.images[names[k]]), atlases[k]) for k in atlases)
    ok &= check(same, "atlases survive save + reopen unchanged")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    if not main():
        sys.exit(1)
