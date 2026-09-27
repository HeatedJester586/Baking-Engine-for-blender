# SPDX-License-Identifier: GPL-3.0-or-later
"""
PBR Atlas Baker - core pipeline.

Merges every material slot of a mesh into ONE set of PBR texture atlases
(Base Color, Roughness, Metallic, Normal) on a duplicate of the mesh that
carries a single master material.

Two methods are available:

* GPU    - rasterises every source texture straight into the new UV layout
           on the graphics card. No rays are cast, so floating geometry
           (strings, bridges, knobs) can never shadow or occlude the body.
* CYCLES - native Cycles surface bake (never "Selected to Active").

The add-on UI imports this module, but the file also runs on its own:
open it in Blender's Text Editor, edit the settings at the bottom and
press "Run Script".
"""

import math
import os
import time
from contextlib import contextmanager

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

CHANNELS = ("BaseColor", "Roughness", "Metallic", "Normal")
BAKE_UV_NAME = "UV_Atlas"
TARGET_SUFFIX = "_ATLAS"
MARKER = "pbr_atlas_baker"  # custom property set on objects this tool creates
REPORT_TEXT = "PBR_ATLAS_REPORT.txt"

_SOCKETS = {
    "BaseColor": "Base Color",
    "Roughness": "Roughness",
    "Metallic": "Metallic",
    "Normal": "Normal",
}
# Values used when a slot has no material / no Principled BSDF (linear).
_DEFAULTS = {
    "BaseColor": (0.8, 0.8, 0.8, 1.0),
    "Roughness": (0.5, 0.5, 0.5, 1.0),
    "Metallic": (0.0, 0.0, 0.0, 1.0),
    "Normal": (0.5, 0.5, 1.0, 1.0),
}
# 8-bit fill for atlas pixels that no UV island reaches, even after padding.
_BACKGROUND = {
    "BaseColor": (128, 128, 128),
    "Roughness": (128, 128, 128),
    "Metallic": (0, 0, 0),
    "Normal": (128, 128, 255),
}
_MISSING_TEXTURE = (1.0, 0.0, 1.0, 1.0)  # Blender's pink "missing image" colour
_LUMINANCE = (0.2126, 0.7152, 0.0722, 0.0)  # Blender's implicit colour -> float
_CHANNEL_OUTPUTS = {"Red": 0, "R": 0, "Green": 1, "G": 1, "Blue": 2, "B": 2, "Alpha": 3, "A": 3}
_WRAP_MODES = {"REPEAT": 0, "EXTEND": 1, "CLIP": 1, "MIRROR": 2}
_EDGE_ORDER = (0, 1, 1, 2, 2, 0)


# ---------------------------------------------------------------------------
# Material analysis
# ---------------------------------------------------------------------------

def _rgba(value):
    """Socket default value (float, 3- or 4-vector) -> RGBA tuple."""
    if isinstance(value, (int, float)):
        return (float(value),) * 3 + (1.0,)
    value = tuple(value)
    if len(value) >= 4:
        return tuple(float(v) for v in value[:4])
    if len(value) == 3:
        return tuple(float(v) for v in value) + (1.0,)
    return (float(value[0]),) * 3 + (1.0,)


def _vec4(value):
    """Scalar or colour -> 4-tuple used by the affine helpers."""
    if isinstance(value, (int, float)):
        return (float(value),) * 4
    return _rgba(value)


class ChannelSource:
    """What one BSDF input evaluates to.

    Either a constant (``image is None`` -> ``offset`` holds the colour) or
    ``texture(uv) * scale + offset`` with an optional channel selection.
    All colours are scene-linear, exactly as Blender renders them.
    """

    def __init__(self, value=(0.0, 0.0, 0.0, 1.0)):
        self.image = None
        self.mask = None            # None = use RGB, else weights to reduce to grey
        self.scale = (1.0, 1.0, 1.0, 1.0)
        self.offset = _rgba(value)
        self.uv_layer = None        # None = the mesh's default (render) UV map
        self.uv_affine = None       # 4x4 Matrix from Mapping nodes
        self.wrap = "REPEAT"
        self.strength = 1.0         # normal maps
        self.tangent_uv = None      # normal maps: UV map that defines tangents

    @classmethod
    def from_image(cls, image):
        src = cls((0.0, 0.0, 0.0, 0.0))
        src.image = image
        return src

    def affine(self, mul, add=0.0):
        """Apply ``x * mul + add`` on top of the current value."""
        mul, add = _vec4(mul), _vec4(add)
        self.scale = tuple(s * m for s, m in zip(self.scale, mul))
        self.offset = tuple(o * m + a for o, m, a in zip(self.offset, mul, add))
        return self

    def select(self, weights):
        """Reduce to a grey value: dot(colour, weights)."""
        weights = tuple(float(w) for w in weights)
        if self.image is None:
            v = sum(o * w for o, w in zip(self.offset, weights))
            self.offset = (v, v, v, 1.0)
            return self
        if self.mask is None:
            self.mask = weights
            s = sum(sc * w for sc, w in zip(self.scale, weights))
            o = sum(of * w for of, w in zip(self.offset, weights))
        else:
            # Already grey: every channel holds the same value.
            s, o = self.scale[0], self.offset[0]
        self.scale = (s, s, s, 1.0)
        self.offset = (o, o, o, 0.0)
        return self

    def select_channel(self, index):
        weights = [0.0, 0.0, 0.0, 0.0]
        weights[index] = 1.0
        return self.select(weights)

    def to_scalar(self):
        """Colour feeding a float socket: Blender converts with luminance."""
        if self.image is None or self.mask is None:
            self.select(_LUMINANCE)
        return self

    def describe(self):
        if self.image is None:
            r, g, b, _ = self.offset
            if abs(r - g) < 1e-6 and abs(g - b) < 1e-6:
                return f"constant {r:.3f}"
            return f"constant ({r:.3f}, {g:.3f}, {b:.3f})"
        img = self.image
        text = f"image '{img.name}'"
        if self.mask is not None:
            picks = [c for c, w in zip("RGBA", self.mask) if w]
            text += f" [{picks[0]}]" if len(picks) == 1 else " [luminance]"
        extras = [img.colorspace_settings.name]
        try:
            extras.append("%dx%d" % tuple(img.size))
        except Exception:
            pass
        if self.scale[:3] != (1.0, 1.0, 1.0) or any(self.offset[:3]):
            extras.append("math")
        if self.uv_layer:
            extras.append(f"uv '{self.uv_layer}'")
        if self.uv_affine is not None:
            extras.append("mapping")
        if self.wrap != "REPEAT":
            extras.append(self.wrap.lower())
        if self.strength != 1.0:
            extras.append(f"strength {self.strength:g}")
        return f"{text} ({', '.join(extras)})"


def _upstream(socket):
    """Return ``(node, output_socket)`` feeding ``socket``.

    Reroutes and muted nodes are looked through; ``(None, None)`` means
    nothing usable is connected.
    """
    for _ in range(64):
        if not socket.is_linked:
            return None, None
        link = socket.links[0]
        if link.is_muted:
            return None, None
        node, out = link.from_node, link.from_socket
        if node.type == 'REROUTE':
            socket = node.inputs[0]
            continue
        if node.mute:
            through = next((l.from_socket for l in node.internal_links if l.to_socket == out), None)
            if through is None:
                return None, None
            socket = through
            continue
        return node, out
    return None, None


def _input(node, identifier):
    return next((s for s in node.inputs if s.identifier == identifier), None)


def _is_data(image):
    cs = image.colorspace_settings
    return bool(getattr(cs, "is_data", False)) or cs.name in ("Non-Color", "Raw", "Generic Data", "Data")


def _image_loaded(image):
    try:
        w, h = image.size
        return w > 0 and h > 0
    except Exception:
        return False


def _mapping_matrix(node, warn):
    if node.vector_type not in ('POINT', 'TEXTURE'):
        warn(f"Mapping '{node.name}' of type {node.vector_type} is ignored")
        return None
    values = {}
    for name, default in (("Location", (0, 0, 0)), ("Rotation", (0, 0, 0)), ("Scale", (1, 1, 1))):
        sock = node.inputs.get(name)
        if sock is not None and sock.is_linked:
            warn(f"Mapping '{node.name}' has a linked {name} input; mapping ignored")
            return None
        values[name] = tuple(sock.default_value) if sock is not None else default
    m = Matrix.LocRotScale(Vector(values["Location"]), Euler(values["Rotation"], 'XYZ'),
                           Vector(values["Scale"]))
    return m.inverted_safe() if node.vector_type == 'TEXTURE' else m


def _resolve_uv(socket, src, warn):
    node, out = _upstream(socket)
    affine = None
    while node is not None:
        if node.type == 'MAPPING':
            m = _mapping_matrix(node, warn)
            if m is not None:
                affine = m if affine is None else affine @ m
            node, out = _upstream(node.inputs["Vector"])
            continue
        if node.type == 'UVMAP':
            src.uv_layer = node.uv_map or None
        elif not (node.type == 'TEX_COORD' and out.identifier == "UV"):
            warn(f"'{node.name}' texture coordinates are not supported; using the UV map instead")
        break
    src.uv_affine = affine


def _resolve_image(node, out, warn):
    image = node.image
    if image is None:
        warn(f"Image Texture '{node.name}' has no image assigned (renders black)")
        return ChannelSource((0.0, 0.0, 0.0, 1.0))
    if image.source == 'TILED':
        warn(f"image '{image.name}' is a UDIM tile set, which is not supported")
        return ChannelSource(_MISSING_TEXTURE)
    if not _image_loaded(image):
        warn(f"image '{image.name}' could not be loaded (missing file? '{image.filepath}') - "
             "it will appear pink")
        return ChannelSource(_MISSING_TEXTURE)
    src = ChannelSource.from_image(image)
    src.wrap = node.extension
    _resolve_uv(node.inputs["Vector"], src, warn)
    if out.identifier == "Alpha":
        src.select_channel(3)
    return src


def _resolve_math(node, warn, depth):
    op = node.operation
    a, b = node.inputs[0], node.inputs[1]
    la, lb = _upstream(a)[0] is not None, _upstream(b)[0] is not None
    if not la and not lb:
        x, y = a.default_value, b.default_value
        consts = {'MULTIPLY': x * y, 'ADD': x + y, 'SUBTRACT': x - y,
                  'DIVIDE': x / y if y else 0.0}
        if op in consts:
            return ChannelSource(consts[op])
        warn(f"Math '{node.name}' ({op}) is not supported; using its first input")
        return ChannelSource(x)
    if la and lb:
        warn(f"Math '{node.name}' combines two textures; only the first is used")
        return _resolve(a, warn, depth + 1).to_scalar()
    linked, const = (a, b.default_value) if la else (b, a.default_value)
    inner = _resolve(linked, warn, depth + 1).to_scalar()
    if op == 'MULTIPLY':
        return inner.affine(const)
    if op == 'ADD':
        return inner.affine(1.0, const)
    if op == 'SUBTRACT':
        return inner.affine(1.0, -const) if la else inner.affine(-1.0, const)
    if op == 'DIVIDE' and la and const:
        return inner.affine(1.0 / const)
    warn(f"Math '{node.name}' ({op}) is not supported; passing its texture through")
    return inner


def _resolve_mix(node, warn, depth):
    if node.type == 'MIX':
        kind = {'RGBA': "Color", 'FLOAT': "Float"}.get(node.data_type)
        if kind is None:
            warn(f"Mix '{node.name}' of type {node.data_type} is not supported")
            return ChannelSource(_DEFAULTS["BaseColor"])
        fac, a, b = _input(node, "Factor_Float"), _input(node, "A_" + kind), _input(node, "B_" + kind)
        blend = node.blend_type if kind == "Color" else 'MIX'
    else:  # legacy MixRGB
        fac, a, b = node.inputs["Fac"], node.inputs["Color1"], node.inputs["Color2"]
        blend = node.blend_type
    if fac.is_linked:
        warn(f"Mix '{node.name}' has a linked factor; using input A only")
        return _resolve(a, warn, depth + 1)
    f = min(max(fac.default_value, 0.0), 1.0)
    if f == 0.0:
        return _resolve(a, warn, depth + 1)
    if f == 1.0 and blend == 'MIX':
        return _resolve(b, warn, depth + 1)
    la, lb = _upstream(a)[0] is not None, _upstream(b)[0] is not None
    if blend not in ('MIX', 'MULTIPLY') or (la and lb):
        warn(f"Mix '{node.name}' ({blend}) is not supported; using input A only")
        return _resolve(a if la or not lb else b, warn, depth + 1)
    if not la and not lb:
        ca, cb = _rgba(a.default_value), _rgba(b.default_value)
        if blend == 'MIX':
            return ChannelSource(tuple(x * (1 - f) + y * f for x, y in zip(ca, cb)))
        return ChannelSource(tuple(x * (1 - f + f * y) for x, y in zip(ca, cb)))
    if la:
        inner, c = _resolve(a, warn, depth + 1), _rgba(b.default_value)
        if blend == 'MIX':
            return inner.affine(1 - f, tuple(x * f for x in c))
        return inner.affine(tuple(1 - f + f * x for x in c))
    inner, c = _resolve(b, warn, depth + 1), _rgba(a.default_value)
    if blend == 'MIX':
        return inner.affine(f, tuple(x * (1 - f) for x in c))
    return inner.affine(tuple(f * x for x in c), tuple(x * (1 - f) for x in c))


def _resolve(socket, warn, depth=0):
    """Describe what a colour/float socket evaluates to."""
    fallback = _rgba(getattr(socket, "default_value", 0.0))
    node, out = _upstream(socket)
    if node is None:
        return ChannelSource(fallback)
    if depth > 24:
        warn("node chain is too deep; using the socket's default value")
        return ChannelSource(fallback)
    kind = node.type
    if kind == 'TEX_IMAGE':
        return _resolve_image(node, out, warn)
    if kind in ('SEPARATE_COLOR', 'SEPARATE_RGB'):
        if getattr(node, "mode", 'RGB') != 'RGB':
            warn(f"'{node.name}' uses {node.mode} mode; treated as RGB")
        inner = _resolve(node.inputs[0], warn, depth + 1)
        index = _CHANNEL_OUTPUTS.get(out.identifier, _CHANNEL_OUTPUTS.get(out.name))
        return inner.select_channel(index) if index is not None else inner
    if kind == 'MATH':
        return _resolve_math(node, warn, depth)
    if kind in ('MIX', 'MIX_RGB'):
        return _resolve_mix(node, warn, depth)
    if kind == 'INVERT':
        fac = node.inputs["Fac"]
        if fac.is_linked:
            warn(f"Invert '{node.name}' has a linked factor; assuming 1.0")
        f = 1.0 if fac.is_linked else fac.default_value
        return _resolve(node.inputs["Color"], warn, depth + 1).affine(1 - 2 * f, f)
    if kind in ('RGB', 'VALUE'):
        return ChannelSource(out.default_value)
    if kind == 'VALTORGB':
        warn(f"Color Ramp '{node.name}' is ignored; its input is passed through")
        return _resolve(node.inputs["Fac"], warn, depth + 1)
    warn(f"'{node.name}' ({node.bl_idname}) is not supported; using the socket's default value")
    return ChannelSource(fallback)


def _resolve_normal(socket, warn, depth=0):
    flat = ChannelSource(_DEFAULTS["Normal"])
    node, _ = _upstream(socket)
    if node is None:
        return flat
    if node.type == 'NORMAL_MAP':
        if node.space != 'TANGENT':
            warn(f"Normal Map '{node.name}' uses {node.space} space; only tangent space is supported")
            return flat
        strength = node.inputs["Strength"]
        if strength.is_linked:
            warn(f"Normal Map '{node.name}' has a linked Strength; assuming 1.0")
        src = _resolve(node.inputs["Color"], warn, depth + 1)
        if src.image is None:
            return flat
        src.strength = 1.0 if strength.is_linked else strength.default_value
        src.tangent_uv = node.uv_map or None
        return src
    if node.type == 'BUMP' and depth < 8:
        warn(f"Bump '{node.name}': height detail is not transferred, only its Normal input")
        return _resolve_normal(node.inputs["Normal"], warn, depth + 1)
    warn(f"'{node.name}' ({node.bl_idname}) feeding Normal is not supported; using a flat normal")
    return flat


def _active_output(tree):
    outputs = [n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL']
    return next((n for n in outputs if n.is_active_output), outputs[0] if outputs else None)


def _surface_bsdf(material):
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return None
    out = _active_output(tree)
    if out is not None:
        node, _ = _upstream(out.inputs["Surface"])
        if node is not None and node.type == 'BSDF_PRINCIPLED':
            return node
    return next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def build_plan(obj):
    """Work out, for every material slot, where each PBR channel comes from.

    Returns a list (one entry per slot) of dicts with ``material``,
    ``sources`` ({channel: ChannelSource}) and ``warnings``.
    """
    slots = list(obj.material_slots) or [None]
    plan = []
    for slot in slots:
        material = slot.material if slot is not None else None
        warnings = []

        def warn(msg, _w=warnings):
            if msg not in _w:
                _w.append(msg)

        bsdf = _surface_bsdf(material) if material is not None else None
        if material is None:
            warn("empty slot - default values are used")
        elif bsdf is None:
            warn("no Principled BSDF found - default values are used")
        else:
            out = _active_output(material.node_tree)
            if out is None or _upstream(out.inputs["Surface"])[0] != bsdf:
                warn(f"Principled BSDF '{bsdf.name}' is not wired straight to the output; "
                     "using it anyway")

        sources = {}
        for channel in CHANNELS:
            if bsdf is None:
                src = ChannelSource(_DEFAULTS[channel])
            elif channel == "Normal":
                src = _resolve_normal(bsdf.inputs["Normal"], warn)
            else:
                src = _resolve(bsdf.inputs[_SOCKETS[channel]], warn)
                if channel != "BaseColor":
                    src.to_scalar()
            if src.image is not None:
                data = _is_data(src.image)
                if channel == "BaseColor" and data:
                    warn(f"Base Color image '{src.image.name}' is set to Non-Color")
                elif channel != "BaseColor" and not data:
                    warn(f"{channel} image '{src.image.name}' uses the "
                         f"'{src.image.colorspace_settings.name}' colour space, so Blender "
                         "gamma-decodes it; set it to Non-Color if it holds raw data")
            sources[channel] = src
        plan.append({"material": material, "sources": sources, "warnings": warnings})
    return plan


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _triangles(mesh):
    mesh.calc_loop_triangles()
    tris = mesh.loop_triangles
    count = len(tris)
    loops = np.empty(count * 3, np.int32)
    mats = np.empty(count, np.int32)
    tris.foreach_get("loops", loops)
    tris.foreach_get("material_index", mats)
    return loops.reshape(count, 3), mats


def _uv_array(mesh, name):
    arr = np.zeros(len(mesh.loops) * 2, np.float32)
    if name is None or name not in mesh.uv_layers:
        return arr.reshape(-1, 2)
    attr = mesh.attributes.get(name)
    if attr is not None and attr.data_type == 'FLOAT2' and attr.domain == 'CORNER':
        attr.data.foreach_get("vector", arr)
    else:
        mesh.uv_layers[name].data.foreach_get("uv", arr)
    return arr.reshape(-1, 2)


def _default_uv_name(mesh):
    layers = mesh.uv_layers
    if not len(layers):
        return None
    return next((l.name for l in layers if l.active_render), layers[0].name)


def _referenced_uv_maps(obj):
    """Names of UV maps / attributes that any of the object's materials read."""
    names, seen = set(), set()

    def scan(tree):
        if tree is None or tree.as_pointer() in seen:
            return
        seen.add(tree.as_pointer())
        for node in tree.nodes:
            uv_map = getattr(node, "uv_map", "")
            if uv_map:
                names.add(uv_map)
            if node.type == 'ATTRIBUTE' and node.attribute_name:
                names.add(node.attribute_name)
            if node.type == 'GROUP':
                scan(node.node_tree)

    for slot in obj.material_slots:
        if slot.material is not None:
            scan(slot.material.node_tree)
    return names


def _check_source(source):
    if source is None or source.type != 'MESH':
        raise RuntimeError("Select a mesh object to bake.")
    if not len(source.data.polygons):
        raise RuntimeError(f"'{source.name}' has no faces.")


def _prepare_target(source, angle_limit, island_margin, log):
    """Duplicate ``source`` and give the copy a fresh non-overlapping UV map."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    name = source.name + TARGET_SUFFIX
    old = bpy.data.objects.get(name)
    if old is not None:
        if not old.get(MARKER):
            raise RuntimeError(f"An object named '{name}' already exists and was not made by "
                               "PBR Atlas Baker. Rename it first.")
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    target = source.copy()
    target.data = source.data.copy()
    target.name = name
    target.data.name = name
    target[MARKER] = True
    collections = source.users_collection or (bpy.context.scene.collection,)
    for coll in collections:
        coll.objects.link(target)

    mesh = target.data
    default_uv = _default_uv_name(mesh)
    # Make room for the atlas UV map (Blender allows 8): drop UV maps on the
    # copy that no material reads. The original object is left untouched.
    keep = _referenced_uv_maps(source)
    if default_uv is not None:
        keep.add(default_uv)
    keep.discard(BAKE_UV_NAME)
    unused = [l.name for l in mesh.uv_layers if l.name not in keep]
    for uv_name in unused:
        mesh.uv_layers.remove(mesh.uv_layers[uv_name])
    if unused:
        log(f"  removed {len(unused)} unused UV map(s) from the copy: {', '.join(unused)}")
    if len(mesh.uv_layers) >= 8:
        raise RuntimeError("All 8 UV maps are used by the materials, so there is no room for the "
                           "atlas UV map. Remove one UV map from the object first.")
    bake_uv = mesh.uv_layers.new(name=BAKE_UV_NAME, do_init=False)
    mesh.uv_layers.active = bake_uv
    if default_uv is not None:
        mesh.uv_layers[default_uv].active_render = True

    view_layer = bpy.context.view_layer
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    target.hide_viewport = False
    target.hide_select = False
    target.hide_set(False)
    target.select_set(True)
    view_layer.objects.active = target

    log(f"Unwrapping '{target.name}' (Smart UV Project, {math.degrees(angle_limit):.1f} deg)...")
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bpy.ops.mesh.reveal(select=False)
        bpy.ops.mesh.select_all(action='SELECT')
        # angle_limit is in RADIANS. Passing degrees collapses every UV to (0, 0).
        bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    uv = _uv_array(target.data, BAKE_UV_NAME)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    log(f"  atlas UV range: U [{lo[0]:.3f}, {hi[0]:.3f}]  V [{lo[1]:.3f}, {hi[1]:.3f}]")
    if (hi - lo).max() < 1e-4:
        raise RuntimeError("Smart UV Project collapsed the UVs to a point; check the angle limit.")
    return target, default_uv


def _output_dir(path):
    """Folder for optional PNG copies; None = keep the textures in the .blend only."""
    if not path:
        return None
    if path.startswith("//") and not bpy.data.filepath:
        raise RuntimeError("Save the .blend file first, or choose a full folder path for the PNG files.")
    path = os.path.abspath(bpy.path.abspath(path))
    os.makedirs(path, exist_ok=True)
    return path


def _set_colorspace(image, is_color):
    names = ("sRGB",) if is_color else ("Non-Color", "Data", "Raw", "Generic Data")
    for name in names:
        try:
            image.colorspace_settings.name = name
            return
        except TypeError:
            continue


def _store_image(image, out_dir):
    """Pack ``image`` into the .blend as a PNG. When ``out_dir`` is set, also
    write a PNG copy there. Returns the copy's path, or None."""
    image.file_format = 'PNG'
    if out_dir is None:
        image.filepath_raw = f"//{image.name}.png"  # only names the packed file
        image.pack()
        return None
    path = os.path.join(out_dir, image.name + ".png")
    image.filepath_raw = path
    image.save()
    image.source = 'FILE'
    try:
        image.filepath = bpy.path.relpath(path) if bpy.data.filepath else path
    except ValueError:  # different drive on Windows
        image.filepath = path
    image.reload()
    image.pack()
    return path


def _finalize_target(target, images, hide_source, source):
    """Swap every slot for one master material that uses the atlases."""
    mesh = target.data
    for name in [l.name for l in mesh.uv_layers if l.name != BAKE_UV_NAME]:
        mesh.uv_layers.remove(mesh.uv_layers[name])
    uv = mesh.uv_layers[BAKE_UV_NAME]
    mesh.uv_layers.active = uv
    uv.active_render = True

    mat_name = f"M_{target.name}"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (400, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def tex(channel, y):
        node = nodes.new('ShaderNodeTexImage')
        node.image = images[channel]
        node.label = channel
        node.location = (-450, y)
        return node

    links.new(tex("BaseColor", 300).outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex("Metallic", 30).outputs["Color"], bsdf.inputs["Metallic"])
    links.new(tex("Roughness", -240).outputs["Color"], bsdf.inputs["Roughness"])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (-150, -520)
    links.new(tex("Normal", -520).outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    for slot in target.material_slots:
        slot.link = 'DATA'
    mesh.materials.clear()
    mesh.materials.append(mat)
    mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), np.int32))
    mesh.update()

    if hide_source:
        source.hide_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    return mat


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose(obj):
    """Return a text report of what the baker will read from every slot."""
    lines = ["=" * 72]
    if obj is None or obj.type != 'MESH':
        lines += ["PBR ATLAS DIAGNOSTIC: no mesh object selected", "=" * 72]
        return "\n".join(lines)
    mesh = obj.data
    _, tri_mats = _triangles(mesh)
    plan = build_plan(obj)
    counts = np.bincount(np.clip(tri_mats, 0, len(plan) - 1), minlength=len(plan))
    uv_names = [l.name + (" (render)" if l.active_render else "") for l in mesh.uv_layers]

    lines += [
        f"PBR ATLAS DIAGNOSTIC: '{obj.name}'",
        "=" * 72,
        f"Triangles: {len(tri_mats)}   Material slots: {len(obj.material_slots)}",
        f"UV maps: {', '.join(uv_names) or 'NONE'}",
        f"Custom split normals: {'yes' if getattr(mesh, 'has_custom_normals', False) else 'no'}",
        "-" * 72,
    ]
    total_warnings = 0
    for i, entry in enumerate(plan):
        mat = entry["material"]
        lines.append(f"Slot {i:02d}  '{mat.name if mat else '<empty>'}'  ({counts[i]} triangles)")
        for channel in CHANNELS:
            lines.append(f"    {channel:<10} {entry['sources'][channel].describe()}")
        for w in entry["warnings"]:
            lines.append(f"    ! {w}")
        total_warnings += len(entry["warnings"])
        lines.append("-" * 72)
    lines.append(f"{total_warnings} warning(s).")
    return "\n".join(lines)


def write_report(obj):
    report = diagnose(obj)
    text = bpy.data.texts.get(REPORT_TEXT) or bpy.data.texts.new(REPORT_TEXT)
    text.clear()
    text.write(report)
    return report


# ---------------------------------------------------------------------------
# GPU rasteriser
# ---------------------------------------------------------------------------

_VERTEX_SOURCE = """
void main()
{
  v_uv = a_uv;
  v_rot = a_rot;
  gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAGMENT_SOURCE = """
vec3 linear_to_srgb(vec3 c)
{
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

vec2 wrap_uv(vec2 uv)
{
  if (u_wrap == 1) {
    return clamp(uv, 0.0, 1.0);
  }
  if (u_wrap == 2) {
    return 1.0 - abs(mod(uv, 2.0) - 1.0);
  }
  return fract(uv);
}

void main()
{
  /* Gradients come from the unwrapped UVs so tiling seams pick the right mip. */
  vec4 s = textureGrad(u_tex, wrap_uv(v_uv), dFdx(v_uv), dFdy(v_uv));
  vec3 c = (dot(u_mask, vec4(1.0)) > 0.5) ? vec3(dot(s, u_mask)) : s.rgb;
  c = c * u_scale.rgb + u_offset.rgb;
  if (u_normal != 0) {
    /* Re-express the tangent-space normal in the atlas UV's tangent frame. */
    vec3 n = c * 2.0 - 1.0;
    vec2 xy = vec2(dot(v_rot.xy, n.xy), dot(v_rot.zw, n.xy)) * u_strength;
    n = normalize(vec3(xy, max(n.z, 1e-4)));
    c = n * 0.5 + 0.5;
  }
  else if (u_srgb != 0) {
    c = linear_to_srgb(c);
  }
  FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


def _atlas_shader():
    import gpu

    iface = gpu.types.GPUStageInterfaceInfo("pbr_atlas_iface")
    iface.smooth('VEC2', "v_uv")
    iface.flat('VEC4', "v_rot")

    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC2', "a_pos")
    info.vertex_in(1, 'VEC2', "a_uv")
    info.vertex_in(2, 'VEC4', "a_rot")
    info.vertex_out(iface)
    info.sampler(0, 'FLOAT_2D', "u_tex")
    info.push_constant('VEC4', "u_scale")
    info.push_constant('VEC4', "u_offset")
    info.push_constant('VEC4', "u_mask")
    info.push_constant('FLOAT', "u_strength")
    info.push_constant('INT', "u_wrap")
    info.push_constant('INT', "u_normal")
    info.push_constant('INT', "u_srgb")
    info.fragment_out(0, 'VEC4', "FragColor")
    info.vertex_source(_VERTEX_SOURCE)
    info.fragment_source(_FRAGMENT_SOURCE)
    return gpu.shader.create_from_info(info)


def _dummy_texture():
    import gpu
    data = gpu.types.Buffer('FLOAT', 4, [1.0, 1.0, 1.0, 1.0])
    return gpu.types.GPUTexture((1, 1), format='RGBA32F', data=data)


def _render(size, draw):
    """Run ``draw`` into a cleared size x size RGBA8 target; return uint8 pixels."""
    import gpu

    offscreen = gpu.types.GPUOffScreen(size, size, format='RGBA8')
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.viewport_set(0, 0, size, size)
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.face_culling_set('NONE')
            draw()
            buf = fb.read_color(0, 0, size, size, 4, 0, 'UBYTE')
    finally:
        offscreen.free()
    buf.dimensions = size * size * 4
    try:
        pixels = np.frombuffer(buf, dtype=np.uint8).copy()
    except (TypeError, ValueError):
        pixels = np.array(buf.to_list(), dtype=np.uint8)
    return pixels.reshape(size, size, 4)


def _set_uniforms(shader, src, channel, dummy):
    import gpu

    textured = src.image is not None
    shader.uniform_sampler("u_tex", gpu.texture.from_image(src.image) if textured else dummy)
    shader.uniform_float("u_scale", src.scale if textured else (0.0, 0.0, 0.0, 0.0))
    shader.uniform_float("u_offset", src.offset)
    shader.uniform_float("u_mask", src.mask if (textured and src.mask) else (0.0, 0.0, 0.0, 0.0))
    shader.uniform_float("u_strength", float(src.strength))
    shader.uniform_int("u_wrap", _WRAP_MODES.get(src.wrap, 0))
    shader.uniform_int("u_normal", 1 if channel == "Normal" else 0)
    shader.uniform_int("u_srgb", 1 if channel == "BaseColor" else 0)


def _draw(shader, primitive, pos, uv, rot):
    from gpu_extras.batch import batch_for_shader

    batch = batch_for_shader(shader, primitive, {
        "a_pos": np.ascontiguousarray(pos, np.float32),
        "a_uv": np.ascontiguousarray(uv, np.float32),
        "a_rot": np.ascontiguousarray(rot, np.float32),
    })
    batch.draw(shader)


def _probe_flip(shader, dummy):
    """Draw into the lower half of a tiny target to learn the readback's row order."""
    pos = np.array([(0, 0), (1, 0), (1, 0.5), (0, 0), (1, 0.5), (0, 0.5)], np.float32)
    white = ChannelSource((1.0, 1.0, 1.0, 1.0))

    def draw():
        shader.bind()
        _set_uniforms(shader, white, "Roughness", dummy)
        _draw(shader, 'TRIS', pos, pos, np.tile((1.0, 0.0, 0.0, 1.0), (6, 1)))

    alpha = _render(8, draw)[..., 3]
    if alpha[0].min() == 255 and alpha[-1].max() == 0:
        return False
    if alpha[-1].min() == 255 and alpha[0].max() == 0:
        return True
    raise RuntimeError("GPU readback check failed: nothing was drawn. "
                       "Try switching the GPU backend in Preferences > System.")


def _tangent_rotation(src, dst):
    """Per-triangle 2x2 matrix taking tangent-space XY from the source UV
    layout to the atlas layout.

    ``src`` and ``dst`` are (n, 3, 2) UV triangles. The affine map between
    the two layouts is reduced to its closest rotation (or reflection, for
    mirrored islands). Returns (n, 4) rows of (m00, m01, m10, m11).
    """
    a1, a2 = src[:, 1] - src[:, 0], src[:, 2] - src[:, 0]
    b1, b2 = dst[:, 1] - dst[:, 0], dst[:, 2] - dst[:, 0]
    det = a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1]
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    m00 = (b1[:, 0] * a2[:, 1] - b2[:, 0] * a1[:, 1]) * inv
    m01 = (b2[:, 0] * a1[:, 0] - b1[:, 0] * a2[:, 0]) * inv
    m10 = (b1[:, 1] * a2[:, 1] - b2[:, 1] * a1[:, 1]) * inv
    m11 = (b2[:, 1] * a1[:, 0] - b1[:, 1] * a2[:, 0]) * inv
    mirrored = (m00 * m11 - m01 * m10) < 0.0
    angle = np.where(mirrored, np.arctan2(m01 + m10, m00 - m11), np.arctan2(m10 - m01, m00 + m11))
    c, s = np.cos(angle), np.sin(angle)
    rot = np.stack([c, np.where(mirrored, s, -s), s, np.where(mirrored, -c, c)], axis=1)
    rot[~ok] = (1.0, 0.0, 0.0, 1.0)
    return rot.astype(np.float32)


def _apply_affine(uv, m):
    a = np.array([[m[0][0], m[0][1]], [m[1][0], m[1][1]]], np.float32)
    t = np.array([m[0][3], m[1][3]], np.float32)
    return uv @ a.T + t


class _MeshData:
    """Triangle and UV arrays of the target mesh, read once."""

    def __init__(self, mesh, default_uv, slot_count):
        self.mesh = mesh
        self.default_uv = default_uv
        self.tri_loops, mats = _triangles(mesh)
        self.tri_mats = np.clip(mats, 0, max(slot_count - 1, 0))
        self.bake_uv = _uv_array(mesh, BAKE_UV_NAME)
        self._uv_cache = {}

    def uv(self, name):
        name = name or self.default_uv
        if name not in self._uv_cache:
            self._uv_cache[name] = _uv_array(self.mesh, name)
        return self._uv_cache[name]

    def slot_triangles(self, slot):
        return np.flatnonzero(self.tri_mats == slot)

    def geometry(self, tris, src, channel):
        loops = self.tri_loops[tris]
        dst = self.bake_uv[loops]
        uv = self.uv(src.uv_layer)[loops]
        if src.uv_affine is not None:
            uv = _apply_affine(uv, src.uv_affine)
        if channel == "Normal" and src.image is not None:
            rot = _tangent_rotation(self.uv(src.tangent_uv)[loops], dst)
        else:
            rot = np.tile(np.array((1.0, 0.0, 0.0, 1.0), np.float32), (len(tris), 1))
        return dst, uv, rot


def _draw_channel(shader, dummy, mesh_data, plan, channel):
    for slot, entry in enumerate(plan):
        tris = mesh_data.slot_triangles(slot)
        if not tris.size:
            continue
        src = entry["sources"][channel]
        dst, uv, rot = mesh_data.geometry(tris, src, channel)
        shader.bind()
        _set_uniforms(shader, src, channel, dummy)
        # Edges first: guarantees sub-pixel slivers (strings, fret wires) get pixels.
        order = list(_EDGE_ORDER)
        _draw(shader, 'LINES', dst[:, order].reshape(-1, 2), uv[:, order].reshape(-1, 2),
              np.repeat(rot, 6, axis=0))
        _draw(shader, 'TRIS', dst.reshape(-1, 2), uv.reshape(-1, 2), np.repeat(rot, 3, axis=0))


_NEIGHBOURS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _dilation_index(covered, margin):
    """For each pixel, the index of the covered pixel it should copy.

    Grows islands outwards by ``margin`` pixels (-1 = still empty).
    """
    h, w = covered.shape
    idx = np.full((h, w), -1, np.int32)
    idx[covered] = np.flatnonzero(covered)
    for _ in range(margin):
        if not (idx < 0).any():
            break
        prev = idx.copy()
        for dy, dx in _NEIGHBOURS:
            dst = idx[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
            src = prev[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            take = (dst < 0) & (src >= 0)
            dst[take] = src[take]
    return idx.ravel()


def _pad(pixels, fill_idx, background):
    flat = pixels.reshape(-1, 4)
    out = flat[np.maximum(fill_idx, 0)]
    out[fill_idx < 0, :3] = background
    out[:, 3] = 255
    return out


def bake_gpu_atlas(source, resolution=4096, margin=16, angle_limit=math.radians(66.0),
                   island_margin=0.005, output_dir=None, hide_source=True, log=print):
    """Rasterise every material of ``source`` into four PBR atlases.

    The atlases are packed into the .blend; ``output_dir`` optionally also
    writes PNG copies. Returns a dict with the new object, the images, the
    PNG paths (if any) and statistics.
    """
    t_start = time.time()
    _check_source(source)
    out_dir = _output_dir(output_dir)

    plan = build_plan(source)
    warning_count = 0
    for i, entry in enumerate(plan):
        for w in entry["warnings"]:
            log(f"  [slot {i:02d}] {w}")
            warning_count += 1

    target, default_uv = _prepare_target(source, angle_limit, island_margin, log)
    mesh_data = _MeshData(target.data, default_uv, len(plan))

    shader = _atlas_shader()
    dummy = _dummy_texture()
    flip = _probe_flip(shader, dummy)

    fill_idx = None
    coverage = 0.0
    images, paths = {}, {}
    for channel in CHANNELS:
        log(f"Rasterising {channel} ({resolution}x{resolution})...")
        pixels = _render(resolution, lambda: _draw_channel(shader, dummy, mesh_data, plan, channel))
        if flip:
            pixels = pixels[::-1]
        if fill_idx is None:
            covered = pixels[..., 3] > 127
            coverage = float(covered.mean()) * 100.0
            log(f"  UV coverage {coverage:.1f}%; padding islands by {margin} px...")
            fill_idx = _dilation_index(covered, margin)
        pixels = _pad(pixels, fill_idx, _BACKGROUND[channel])

        name = f"{target.name}_{channel}_{resolution}"
        old = bpy.data.images.get(name)
        if old is not None:
            bpy.data.images.remove(old)
        image = bpy.data.images.new(name, resolution, resolution, alpha=False)
        _set_colorspace(image, channel == "BaseColor")
        image.pixels.foreach_set((pixels.astype(np.float32) * (1.0 / 255.0)).ravel())
        path = _store_image(image, out_dir)
        images[channel] = image
        if path:
            paths[channel] = path
            log(f"  saved {path}")

    _finalize_target(target, images, hide_source, source)
    seconds = time.time() - t_start
    log(f"Done in {seconds:.1f} s -> '{target.name}'")
    return {"object": target, "images": images, "paths": paths, "coverage": coverage,
            "warnings": warning_count, "seconds": seconds}


# ---------------------------------------------------------------------------
# Cycles surface bake (alternative)
# ---------------------------------------------------------------------------

def _feed(tree, socket, target_socket, default):
    """Connect whatever drives ``socket`` into ``target_socket`` (or copy its value)."""
    if socket is not None and socket.is_linked:
        tree.links.new(socket.links[0].from_socket, target_socket)
    else:
        value = socket.default_value if socket is not None else default
        target_socket.default_value = _rgba(value)


@contextmanager
def _bake_setup(materials, image, socket_name, default):
    """Temporarily add the bake target node to every material and, for
    colour/data passes, route the wanted BSDF input through an Emission shader.
    Everything is restored afterwards."""
    added, relinks = [], []
    try:
        for mat in materials:
            tree = mat.node_tree
            node = tree.nodes.new('ShaderNodeTexImage')
            node.image = image
            added.append((tree, node))
            for n in tree.nodes:
                n.select = False
            node.select = True
            tree.nodes.active = node
            if socket_name is None:
                continue
            out = _active_output(tree)
            if out is None:
                continue
            surface = out.inputs["Surface"]
            previous = surface.links[0].from_socket if surface.is_linked else None
            emit = tree.nodes.new('ShaderNodeEmission')
            added.append((tree, emit))
            bsdf = _surface_bsdf(mat)
            _feed(tree, bsdf.inputs[socket_name] if bsdf else None, emit.inputs["Color"], default)
            tree.links.new(emit.outputs["Emission"], surface)
            relinks.append((tree, surface, previous))
        yield
    finally:
        for tree, node in reversed(added):
            tree.nodes.remove(node)
        for tree, surface, previous in relinks:
            if previous is not None:
                tree.links.new(previous, surface)


def bake_cycles_atlas(source, resolution=4096, margin=16, angle_limit=math.radians(66.0),
                      island_margin=0.005, output_dir=None, samples=8,
                      clear_custom_normals=True, hide_source=True, log=print):
    """Native Cycles surface bake of the same four atlases.

    Base Color, Roughness and Metallic are baked through an Emission shader,
    so metal parts do not come out black (the Diffuse Color pass multiplies
    by 1 - metallic). The normal pass is a regular tangent-space bake.
    """
    t_start = time.time()
    _check_source(source)
    out_dir = _output_dir(output_dir)

    target, default_uv = _prepare_target(source, angle_limit, island_margin, log)
    mesh = target.data
    if clear_custom_normals and getattr(mesh, "has_custom_normals", False):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        log("  cleared custom split normals on the copy")

    temp_materials = []
    for slot in target.material_slots:
        if slot.material is None or slot.material.node_tree is None:
            temp = bpy.data.materials.new("PBR_ATLAS_TEMP")
            if temp.node_tree is None:
                temp.use_nodes = True
            slot.material = temp
            temp_materials.append(temp)
    if not target.material_slots:
        temp = bpy.data.materials.new("PBR_ATLAS_TEMP")
        if temp.node_tree is None:
            temp.use_nodes = True
        mesh.materials.append(temp)
        temp_materials.append(temp)
    materials = list({s.material.name: s.material for s in target.material_slots}.values())

    scene = bpy.context.scene
    saved = (scene.render.engine, scene.cycles.samples, scene.cycles.device)
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    try:
        if bpy.context.preferences.addons["cycles"].preferences.has_active_device():
            scene.cycles.device = 'GPU'
    except (KeyError, AttributeError):
        pass

    images, paths = {}, {}
    try:
        for channel in CHANNELS:
            log(f"Baking {channel} with Cycles ({resolution}x{resolution})...")
            name = f"{target.name}_{channel}_{resolution}"
            old = bpy.data.images.get(name)
            if old is not None:
                bpy.data.images.remove(old)
            image = bpy.data.images.new(name, resolution, resolution, alpha=False)
            _set_colorspace(image, channel == "BaseColor")
            is_normal = channel == "Normal"
            with _bake_setup(materials, image, None if is_normal else _SOCKETS[channel],
                             _DEFAULTS[channel]):
                bpy.ops.object.bake(
                    type='NORMAL' if is_normal else 'EMIT',
                    use_selected_to_active=False,  # surface bake: no rays between objects
                    margin=margin, margin_type='EXTEND', use_clear=True,
                    target='IMAGE_TEXTURES', uv_layer=BAKE_UV_NAME, normal_space='TANGENT')
            path = _store_image(image, out_dir)
            images[channel] = image
            if path:
                paths[channel] = path
                log(f"  saved {path}")
    finally:
        scene.render.engine, scene.cycles.samples, scene.cycles.device = saved

    _finalize_target(target, images, hide_source, source)
    for temp in temp_materials:
        if temp.users == 0:
            bpy.data.materials.remove(temp)
    seconds = time.time() - t_start
    log(f"Done in {seconds:.1f} s -> '{target.name}'")
    return {"object": target, "images": images, "paths": paths, "coverage": None,
            "warnings": 0, "seconds": seconds}


# ---------------------------------------------------------------------------
# Run from the Text Editor
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OBJECT_NAME = ""                   # "" = the active object, e.g. "guitar.036"
    METHOD = "GPU"                     # "GPU" or "CYCLES"
    RESOLUTION = 4096                  # 1024, 2048, 4096 or 8192
    MARGIN = 16                        # edge padding in pixels
    ANGLE_LIMIT_DEGREES = 66.0         # Smart UV Project angle limit
    ISLAND_MARGIN = 0.005              # space between UV islands (0..1 UV units)
    OUTPUT_DIR = None                  # None = textures only inside the .blend;
                                       # or a folder for PNG copies, e.g. "//atlas_textures/"
    DIAGNOSE_ONLY = False              # True = only write the report

    source_obj = bpy.data.objects.get(OBJECT_NAME) if OBJECT_NAME else bpy.context.active_object
    print(write_report(source_obj))
    if not DIAGNOSE_ONLY:
        options = dict(resolution=RESOLUTION, margin=MARGIN,
                       angle_limit=math.radians(ANGLE_LIMIT_DEGREES),
                       island_margin=ISLAND_MARGIN, output_dir=OUTPUT_DIR)
        if METHOD.upper() == "CYCLES":
            bake_cycles_atlas(source_obj, **options)
        else:
            bake_gpu_atlas(source_obj, **options)
