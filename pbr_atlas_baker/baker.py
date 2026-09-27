# SPDX-License-Identifier: GPL-3.0-or-later
"""
PBR Atlas Baker - texture repacking engine.

Combines every material of a mesh into ONE material with one set of texture
atlases. Nothing is baked or ray traced and nothing is re-unwrapped:

1. The object is duplicated; the original is never modified.
2. For every material, the parts of its textures that the UVs actually use
   are found (UV islands -> texel rectangles called "chunks") and cut out of
   every channel: Base Color, Alpha, Roughness, Metallic, Normal, Emission.
3. The chunks are packed as tightly as possible into one square atlas. The
   smallest power-of-two size that holds them is picked automatically, up
   to a maximum. Chunks are copied texel for texel, so nothing is resampled.
4. The UVs are only moved (never rescaled) so every face points at its
   chunk, all materials are replaced by one material, and the atlases are
   stored inside the .blend.

A chunk is only made smaller when that loses nothing (the chunk is a single
flat colour) or, with a warning, when the atlas would otherwise be larger
than the maximum size.

The add-on UI imports this module, but the file also runs on its own:
open it in Blender's Text Editor, edit the settings at the bottom and press
"Run Script".
"""

import hashlib
import math
import os
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

CHANNELS = ("BaseColor", "Alpha", "Roughness", "Metallic", "Normal", "Emission", "Height")
ATLAS_UV_NAME = "UV_Atlas"
TARGET_SUFFIX = "_ATLAS"
MARKER = "pbr_atlas_baker"  # custom property set on objects this tool creates
REPORT_TEXT = "PBR_ATLAS_REPORT.txt"
LOG_TEXT = "PBR_ATLAS_BUILD_LOG.txt"

_SOCKETS = {
    "BaseColor": ("Base Color",),
    "Alpha": ("Alpha",),
    "Roughness": ("Roughness",),
    "Metallic": ("Metallic",),
    "Emission": ("Emission Color", "Emission"),
}
# Values used when a slot has no material / no Principled BSDF (linear).
_DEFAULTS = {
    "BaseColor": (0.8, 0.8, 0.8, 1.0),
    "Alpha": (1.0, 1.0, 1.0, 1.0),
    "Roughness": (0.5, 0.5, 0.5, 1.0),
    "Metallic": (0.0, 0.0, 0.0, 1.0),
    "Normal": (0.5, 0.5, 1.0, 1.0),
    "Emission": (0.0, 0.0, 0.0, 1.0),
    "Height": (0.0, 0.0, 0.0, 1.0),
}
_SCALAR_CHANNELS = ("Alpha", "Roughness", "Metallic", "Height")
_COLOR_CHANNELS = ("BaseColor", "Emission")
# How each channel is stored in its atlas.
_KIND = {"BaseColor": "srgb", "Emission": "srgb", "Normal": "normal", "Height": "linear",
         "Alpha": "linear", "Roughness": "linear", "Metallic": "linear"}
# Atlas textures: (name, channel for RGB, channel for alpha)
_OUTPUTS = (
    ("BaseColor", "BaseColor", "Alpha"),
    ("Roughness", "Roughness", None),
    ("Metallic", "Metallic", None),
    ("Normal", "Normal", None),
    ("Emission", "Emission", None),
    ("Height", "Height", None),
)
# 8-bit fill for atlas texels that no chunk covers.
_BACKGROUND = {
    "BaseColor": (128, 128, 128),
    "Roughness": (128, 128, 128),
    "Metallic": (0, 0, 0),
    "Normal": (128, 128, 255),
    "Emission": (0, 0, 0),
    "Height": (0, 0, 0),  # replaced by the displacement midlevel
}
_ATLAS_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
AUTO_SIZE_LIMIT = 16384  # largest atlas "Auto" will build (GPU texture limit)
_MISSING_TEXTURE = (1.0, 0.0, 1.0, 1.0)  # Blender's pink "missing image" colour
_LUMINANCE = (0.2126, 0.7152, 0.0722, 0.0)  # Blender's implicit colour -> float
_CHANNEL_OUTPUTS = {"Red": 0, "R": 0, "Green": 1, "G": 1, "Blue": 2, "B": 2, "Alpha": 3, "A": 3}


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
        # Color Ramp: grey = texture . ramp_input[0] * ramp_input[1] + ramp_input[2],
        # colour = ramp(grey); mask / scale / offset then apply to that colour.
        self.ramp = None            # (N, 4) lookup table over 0..1
        self.ramp_input = None      # (weights, scale, offset) producing the grey

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

    def apply_ramp(self, lut):
        """Feed the current (grey) value through a Color Ramp lookup table."""
        if self.image is None:
            x = min(max(self.offset[0], 0.0), 1.0)
            self.offset = tuple(float(v) for v in _ramp_lookup(lut, np.array(x, np.float32)))
            return self
        if self.ramp is not None:  # ramp after ramp: fold into one table
            grey = _apply_stage(self.ramp, self.mask, self.scale, self.offset)
            lut = _ramp_lookup(lut, grey)
            self.ramp = lut
        else:
            weights = self.mask or _LUMINANCE
            self.ramp_input = (tuple(weights), self.scale[0], self.offset[0])
            self.ramp = lut
        self.mask = None
        self.scale = (1.0, 1.0, 1.0, 1.0)
        self.offset = (0.0, 0.0, 0.0, 0.0)
        return self

    def simplify(self):
        """A texture multiplied by zero is just its offset."""
        if self.image is not None and self.ramp is None and not any(self.scale[:3]):
            self.image, self.mask = None, None
            self.offset = tuple(self.offset[:3]) + (1.0,)
        return self

    def key(self, texture_key, default_uv):
        """Hashable description: equal keys produce identical texels."""
        def r(values):
            return tuple(round(float(v), 6) for v in values)
        if self.image is None:
            return ("const",) + r(self.offset[:3])
        affine = None if self.uv_affine is None else r(v for row in self.uv_affine for v in row)
        ramp = None
        if self.ramp is not None:
            ramp = (hashlib.blake2b(np.ascontiguousarray(self.ramp), digest_size=8).hexdigest(),
                    r(self.ramp_input[0]), r(self.ramp_input[1:]))
        return ("tex", texture_key(self.image), self.mask and r(self.mask), r(self.scale),
                r(self.offset), self.uv_layer or default_uv, affine, self.wrap,
                round(float(self.strength), 6), self.tangent_uv or default_uv, ramp)

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
        if self.ramp is not None:
            extras.append("color ramp")
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
        lut = np.array([node.color_ramp.evaluate(float(x)) for x in np.linspace(0.0, 1.0, 1024)],
                       np.float32)
        src = _resolve(node.inputs["Fac"], warn, depth + 1).to_scalar().apply_ramp(lut)
        return src.select_channel(3) if out.identifier == "Alpha" else src
    warn(f"'{node.name}' ({node.bl_idname}) is not supported; using the socket's default value")
    return ChannelSource(fallback)


def _resolve_normal(socket, warn, depth=0, direct_normals=True):
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
        return _resolve_normal(node.inputs["Normal"], warn, depth + 1, direct_normals)
    # A normal-map texture wired straight into Normal (common in game rips):
    # treat it as the tangent-space normal map it is meant to be.
    src = _resolve(socket, warn, depth + 1)
    if src.image is not None and direct_normals:
        warn(f"'{src.image.name}' is plugged straight into Normal without a Normal Map node; "
             "it is used as a real normal map (Blender itself does not, so the original "
             "shows less bump detail)", note=True)
        return src
    if src.image is not None:
        warn(f"'{src.image.name}' is plugged straight into Normal without a Normal Map node, so "
             "Blender does not use it as a normal map; it is left out to match the original",
             note=True)
        return flat
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



def _bsdf_input(bsdf, channel):
    return next((bsdf.inputs.get(n) for n in _SOCKETS[channel] if bsdf.inputs.get(n)), None)


def _check_colorspace(channel, src, warn):
    if src.image is None or (src.mask is not None and not any(src.mask[:3])):
        return  # constants and alpha channels are never colour managed
    data = _is_data(src.image)
    if channel in _COLOR_CHANNELS and data:
        warn(f"{channel} image '{src.image.name}' is set to Non-Color", note=True)
    elif channel not in _COLOR_CHANNELS and channel != "Alpha" and not data:
        warn(f"{channel} image '{src.image.name}' is tagged "
             f"'{src.image.colorspace_settings.name}', so Blender gamma-decodes it; the atlas "
             "keeps that look (enable 'Raw Data Maps' for the raw values)", note=True)


def _resolve_displacement(material, warn):
    """Height source wired into the Material Output's Displacement, and its
    (midlevel, scale). ``(constant 0, None)`` when nothing is connected."""
    none = ChannelSource(_DEFAULTS["Height"])
    tree = getattr(material, "node_tree", None)
    out = _active_output(tree) if tree is not None else None
    socket = out.inputs.get("Displacement") if out is not None else None
    if socket is None:
        return none, None
    node, _ = _upstream(socket)
    if node is None:
        return none, None
    if node.type == 'DISPLACEMENT':
        for name in ("Midlevel", "Scale"):
            if node.inputs[name].is_linked:
                warn(f"Displacement '{node.name}' has a linked {name}; its default value is used")
        src = _resolve(node.inputs["Height"], warn).to_scalar()
        return src, (round(node.inputs["Midlevel"].default_value, 6),
                     round(node.inputs["Scale"].default_value, 6))
    if node.type == 'VECTOR_DISPLACEMENT':
        warn(f"Vector Displacement '{node.name}' is not supported; displacement ignored")
        return none, None
    # A texture wired straight into the Displacement output: used as height.
    return _resolve(socket, warn).to_scalar(), (0.0, 1.0)


def build_plan(obj, direct_normals=True, overrides=None):
    """Work out, for every material slot, where each PBR channel comes from.

    ``overrides`` maps a material name to {"normal": bool, "displacement":
    bool}, overriding ``direct_normals`` and the automatic displacement choice
    for that material. Returns a list (one entry per slot) of dicts with
    ``material``, ``sources`` ({channel: ChannelSource}) and ``warnings``.
    """
    slots = list(obj.material_slots) or [None]
    plan = []
    for slot in slots:
        material = slot.material if slot is not None else None
        warnings, notes = [], []

        def warn(msg, note=False, _w=warnings, _n=notes):
            """Problems go to warnings; things handled automatically go to notes."""
            target = _n if note else _w
            if msg not in target:
                target.append(msg)

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
        height, displacement = _resolve_displacement(material, warn)
        # A texture wired straight into Normal replaces the shading normal, so
        # Blender throws away the displacement bump for this material. When the
        # atlas copies the original look, leave that displacement out too.
        normal_node = _upstream(bsdf.inputs["Normal"])[0] if bsdf is not None else None
        overridden = normal_node is not None and normal_node.type not in ('NORMAL_MAP', 'BUMP')
        choice = (overrides or {}).get(material.name, {}) if material is not None else {}
        use_normal = choice.get("normal", direct_normals)
        has_displacement = displacement is not None
        keep_displacement = choice.get("displacement", not overridden or use_normal)
        if has_displacement and not keep_displacement:
            if "displacement" in choice:
                warn("displacement turned off for this material", note=True)
            else:
                warn("its displacement does not show in Blender because a texture is plugged "
                     "straight into Normal; it is left out to match the original", note=True)
            height, displacement = ChannelSource(_DEFAULTS["Height"]), None
        for channel in CHANNELS:
            if channel == "Height":
                src = height
            elif bsdf is None:
                src = ChannelSource(_DEFAULTS[channel])
            elif channel == "Normal":
                src = _resolve_normal(bsdf.inputs["Normal"], warn, direct_normals=use_normal)
            else:
                socket = _bsdf_input(bsdf, channel)
                src = _resolve(socket, warn) if socket is not None else ChannelSource(_DEFAULTS[channel])
                if channel in _SCALAR_CHANNELS:
                    src.to_scalar()
                if channel == "Emission":
                    strength = bsdf.inputs.get("Emission Strength")
                    if strength is not None and strength.is_linked:
                        warn("Emission Strength is linked; assuming 1.0")
                    elif strength is not None:
                        src.affine(strength.default_value)
            src.simplify()
            _check_colorspace(channel, src, warn)
            sources[channel] = src
        plan.append({"material": material, "sources": sources, "warnings": warnings,
                     "notes": notes, "displacement": displacement,
                     "unwired_normal": overridden, "has_displacement": has_displacement,
                     "use_normal": use_normal, "keep_displacement": keep_displacement})
    _check_tiling(obj, plan)
    return plan


def _check_tiling(obj, plan):
    """Warn about materials whose UVs repeat their texture: the repeats have to
    be stored side by side in the atlas, which can take a lot of room."""
    mesh = obj.data
    if not len(mesh.polygons):
        return
    loops = _MeshLoops(mesh)
    face_slot = np.clip(loops.face_material, 0, len(plan) - 1)
    for slot, entry in enumerate(plan):
        textured = [s for s in entry["sources"].values() if s.image is not None]
        faces = np.flatnonzero(face_slot == slot)
        if not textured or not faces.size:
            continue
        src = max(textured, key=lambda s: s.image.size[0] * s.image.size[1])
        idx, _ = loops.loops_of(faces)
        affine = _affine2x3(src.uv_affine)
        t = loops.uv(src.uv_layer)[idx] @ affine[:, :2].T + affine[:, 2]
        span = t.max(axis=0) - t.min(axis=0)
        if span.max() > 1.05:
            w, h = src.image.size
            entry["warnings"].append(
                f"its UVs repeat '{src.image.name}' {span[0]:.1f} x {span[1]:.1f} times, so it "
                f"needs about {int(span[0] * w)} x {int(span[1] * h)} px in the atlas "
                "(it is shrunk if that does not fit)")


# ---------------------------------------------------------------------------
# Mesh and file helpers
# ---------------------------------------------------------------------------

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


def _check_source(source):
    if source is None or source.type != 'MESH':
        raise RuntimeError("Select a mesh object.")
    if source.get(MARKER):
        raise RuntimeError(f"'{source.name}' is already an atlas; select the original object.")
    if not len(source.data.polygons):
        raise RuntimeError(f"'{source.name}' has no faces.")


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


def _make_target(source):
    """Duplicate ``source`` (object and mesh data) as ``<name>_ATLAS``."""
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
    for coll in source.users_collection or (bpy.context.scene.collection,):
        coll.objects.link(target)
    target.hide_viewport = False
    target.hide_select = False
    return target


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose(obj, direct_normals=True):
    """Return a text report of what will be read from every material slot."""
    lines = ["=" * 72]
    if obj is None or obj.type != 'MESH':
        lines += ["PBR ATLAS DIAGNOSTIC: no mesh object selected", "=" * 72]
        return "\n".join(lines)
    mesh = obj.data
    plan = build_plan(obj, direct_normals)
    face_mat = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("material_index", face_mat)
    counts = np.bincount(np.clip(face_mat, 0, len(plan) - 1), minlength=len(plan))
    uv_names = [l.name + (" (render)" if l.active_render else "") for l in mesh.uv_layers]

    lines += [
        f"PBR ATLAS DIAGNOSTIC: '{obj.name}'",
        "=" * 72,
        f"Faces: {len(face_mat)}   Material slots: {len(obj.material_slots)}",
        f"UV maps: {', '.join(uv_names) or 'NONE'}",
        "-" * 72,
    ]
    total_warnings = 0
    for i, entry in enumerate(plan):
        mat = entry["material"]
        lines.append(f"Slot {i:02d}  '{mat.name if mat else '<empty>'}'  ({counts[i]} faces)")
        for channel in CHANNELS:
            lines.append(f"    {channel:<10} {entry['sources'][channel].describe()}")
        for w in entry["warnings"]:
            lines.append(f"    ! {w}")
        for n in entry["notes"]:
            lines.append(f"    - {n}")
        total_warnings += len(entry["warnings"])
        lines.append("-" * 72)
    lines.append(f"{total_warnings} warning(s).")
    return "\n".join(lines)


_LABELS = {"BaseColor": "Base Color", "Alpha": "Alpha", "Roughness": "Roughness",
           "Metallic": "Metallic", "Normal": "Normal", "Emission": "Emission",
           "Height": "Displacement"}


def _short(src):
    if src.image is None:
        r, g, b, _ = src.offset
        return f"{r:.2f}" if abs(r - g) < 1e-6 and abs(g - b) < 1e-6 else f"({r:.2f}, {g:.2f}, {b:.2f})"
    text = src.image.name
    if src.mask is not None:
        picks = [c for c, w in zip("RGBA", src.mask) if w]
        text += f" [{picks[0]}]" if len(picks) == 1 else ""
    return text


def check_materials(obj, direct_normals=True, overrides=None):
    """What every material slot feeds into each channel, for the UI checker.

    Returns a list of dicts: ``name``, ``faces``, ``textures`` [(label, text)],
    ``constants`` [(label, text)], ``emission`` and ``displacement`` (text or
    None) and ``warnings``.
    """
    mesh = obj.data
    plan = build_plan(obj, direct_normals, overrides)
    face_mat = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("material_index", face_mat)
    counts = np.bincount(np.clip(face_mat, 0, len(plan) - 1), minlength=len(plan))
    rows = []
    for i, entry in enumerate(plan):
        mat, sources = entry["material"], entry["sources"]
        textures, constants = [], []
        for channel in ("BaseColor", "Alpha", "Roughness", "Metallic", "Normal"):
            src = sources[channel]
            text = "flat" if channel == "Normal" and src.image is None else _short(src)
            (textures if src.image is not None else constants).append((_LABELS[channel], text))
        emission = sources["Emission"]
        emits = emission.image is not None or any(emission.offset[:3])
        height = sources["Height"]
        rows.append({
            "name": f"{i:02d}  {mat.name if mat else '<empty slot>'}",
            "faces": int(counts[i]),
            "textures": textures,
            "constants": constants,
            "emission": _short(emission) if emits else None,
            "displacement": _short(height) if entry["displacement"] is not None else None,
            "warnings": list(entry["warnings"]),
            "material": mat.name if mat else None,
            "unwired_normal": entry["unwired_normal"],
            "has_displacement": entry["has_displacement"],
            "use_normal": entry["use_normal"],
            "keep_displacement": entry["keep_displacement"],
        })
    return rows


def write_report(obj, direct_normals=True):
    report = diagnose(obj, direct_normals)
    text = bpy.data.texts.get(REPORT_TEXT) or bpy.data.texts.new(REPORT_TEXT)
    text.clear()
    text.write(report)
    return report


# ---------------------------------------------------------------------------
# Texture access
# ---------------------------------------------------------------------------

_SRGB_TO_LINEAR = np.array(
    [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in np.arange(256) / 255.0],
    np.float32)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _wrap_index(index, size, wrap):
    """Texel index -> valid index, following the Image Texture node's Extension."""
    if wrap == 'REPEAT':
        return np.mod(index, size)
    if wrap == 'MIRROR':
        m = np.mod(index, 2 * size)
        return np.where(m < size, m, 2 * size - 1 - m)
    return np.clip(index, 0, size - 1)


def _affine2x3(matrix):
    """Mapping-node matrix (or None) -> 2x3 array acting on (u, v, 1)."""
    if matrix is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return np.array([[matrix[0][0], matrix[0][1], matrix[0][3]],
                     [matrix[1][0], matrix[1][1], matrix[1][3]]])


class _Texture:
    """Pixels of one source image, kept in their original precision."""

    def __init__(self, image):
        width, height = image.size
        buf = np.empty(width * height * 4, np.float32)
        image.pixels.foreach_get(buf)
        self.is_float = bool(image.is_float)
        if self.is_float:
            self.data = buf.reshape(height, width, 4)
        else:
            np.multiply(buf, 255.0, out=buf)
            np.rint(buf, out=buf)
            self.data = buf.astype(np.uint8).reshape(height, width, 4)
        self.width, self.height = width, height
        self.srgb = not self.is_float and not _is_data(image)
        digest = hashlib.blake2b(self.data, digest_size=16).hexdigest()
        self.key = (digest, width, height, self.is_float, self.srgb)
        self._mean = None
        self._reduced = {}

    def linear(self, raw, decode=True):
        """Raw texels -> float32 values. ``decode`` turns sRGB colours into
        scene-linear ones; data maps (roughness, normals...) use the raw values."""
        if self.is_float:
            return raw.astype(np.float32)
        out = raw.astype(np.float32) * (1.0 / 255.0)
        if self.srgb and decode:
            out[..., :3] = _SRGB_TO_LINEAR[raw[..., :3]]
        return out

    def block(self, x0, y0, w, h, wrap):
        """Texels [x0, x0 + w) x [y0, y0 + h) exactly as stored."""
        if 0 <= x0 and x0 + w <= self.width and 0 <= y0 and y0 + h <= self.height:
            return self.data[y0:y0 + h, x0:x0 + w]
        xs = _wrap_index(np.arange(x0, x0 + w), self.width, wrap)
        ys = _wrap_index(np.arange(y0, y0 + h), self.height, wrap)
        return self.data[ys[:, None], xs[None, :]]

    def reduced(self, k, decode):
        """Linear values averaged over k x k texel blocks (like a mipmap level)."""
        key = (k, decode)
        if key not in self._reduced:
            h, w = self.height // k * k, self.width // k * k
            lin = self.linear(self.data[:h, :w], decode)
            self._reduced[key] = lin.reshape(h // k, k, w // k, k, 4).mean(axis=(1, 3))
        return self._reduced[key]

    def bilinear(self, px, py, wrap, decode=True, k=1):
        """Linear values at texel coordinates (texel centres sit at i + 0.5).
        ``k`` > 1 samples a k-times smaller averaged copy, for shrinking
        without aliasing."""
        if k > 1:
            data = self.reduced(k, decode)
            height, width = data.shape[:2]
            px, py = px / k, py / k

            def lin(texels):
                return texels
        else:
            data, height, width = self.data, self.height, self.width

            def lin(texels):
                return self.linear(texels, decode)
        x, y = px - 0.5, py - 0.5
        xf, yf = np.floor(x), np.floor(y)
        fx, fy = (x - xf)[..., None], (y - yf)[..., None]
        xi, yi = xf.astype(np.int64), yf.astype(np.int64)
        x0, x1 = _wrap_index(xi, width, wrap), _wrap_index(xi + 1, width, wrap)
        y0, y1 = _wrap_index(yi, height, wrap), _wrap_index(yi + 1, height, wrap)
        lower = lin(data[y0, x0]) * (1 - fx) + lin(data[y0, x1]) * fx
        upper = lin(data[y1, x0]) * (1 - fx) + lin(data[y1, x1]) * fx
        return lower * (1 - fy) + upper * fy

    def mean(self, decode=True):
        if self._mean is None:
            self._mean = {}
        if decode not in self._mean:
            self._mean[decode] = self.linear(self.data, decode).reshape(-1, 4).mean(axis=0)
        return self._mean[decode]


class _TextureCache:
    def __init__(self):
        self._textures = {}

    def get(self, image):
        ptr = image.as_pointer()
        if ptr not in self._textures:
            self._textures[ptr] = _Texture(image)
        return self._textures[ptr]

    def key(self, image):
        return self.get(image).key


class _MeshLoops:
    """Face, loop and UV arrays of the source mesh."""

    def __init__(self, mesh):
        self.mesh = mesh
        n_faces = len(mesh.polygons)
        ints = np.empty(n_faces, np.int32)
        mesh.polygons.foreach_get("loop_start", ints)
        self.loop_start = ints.astype(np.int64)
        mesh.polygons.foreach_get("loop_total", ints)
        self.loop_total = ints.astype(np.int64)
        mesh.polygons.foreach_get("material_index", ints)
        self.face_material = ints.astype(np.int64)
        verts = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get("vertex_index", verts)
        self.loop_vertex = verts.astype(np.int64)
        self.default_uv = _default_uv_name(mesh)
        self._uv = {}

    def uv(self, name):
        name = name or self.default_uv
        if name not in self._uv:
            self._uv[name] = _uv_array(self.mesh, name).astype(np.float64)
        return self._uv[name]

    def loops_of(self, faces):
        """Loop indices of ``faces`` and, per loop, the position of its face in ``faces``."""
        starts, totals = self.loop_start[faces], self.loop_total[faces]
        local_face = np.repeat(np.arange(len(faces)), totals)
        offsets = np.concatenate(([0], np.cumsum(totals)[:-1]))
        loops = np.repeat(starts - offsets, totals) + np.arange(int(totals.sum()))
        return loops, local_face


# ---------------------------------------------------------------------------
# Material groups and chunks
# ---------------------------------------------------------------------------

class _Master:
    """The texel grid a group's chunks are cut from: its largest texture."""

    def __init__(self, texture, uv_layer, affine, wrap):
        self.texture = texture
        self.width, self.height = texture.width, texture.height
        self.uv_layer, self.affine, self.wrap = uv_layer, affine, wrap
        self.texture_name = ""
        self._inverse = np.linalg.inv(affine[:, :2])

    def to_texels(self, uv):
        t = uv @ self.affine[:, :2].T + self.affine[:, 2]
        return t * (self.width, self.height)

    def to_uv(self, px, py):
        tx = px / self.width - self.affine[0, 2]
        ty = py / self.height - self.affine[1, 2]
        inv = self._inverse
        return inv[0, 0] * tx + inv[0, 1] * ty, inv[1, 0] * tx + inv[1, 1] * ty


class _Chunk:
    """A rectangle of source texels, and where it lands in the atlas."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = int(x0), int(y0), int(w), int(h)
        self.base_scale = 1.0
        self.scale = 1.0
        self.aw, self.ah = self.w, self.h   # size in the atlas
        self.ax = self.ay = 0               # position in the atlas
        self.flat = None                    # {channel: texel} when it is one flat colour

    def rescale(self, scale):
        self.scale = scale
        self.aw = max(1, int(math.ceil(self.w * scale)))
        self.ah = max(1, int(math.ceil(self.h * scale)))

    def make_flat(self, values, size):
        self.flat = values
        self.aw = self.ah = size


class _Group:
    """Material slots whose channels are identical, packed together."""

    def __init__(self, sources):
        self.sources = sources
        self.slots = []
        self.faces = []
        self.master = None
        self.chunks = []
        self.loops = None       # loop indices of the group's faces
        self.loop_texel = None  # per loop: position on the master texel grid
        self.loop_chunk = None  # per loop: index into self.chunks
        self.normal_rotation = None
        self.displacement = None  # (midlevel, scale) of its Displacement, if any


def _group_slots(plan, loops, cache):
    face_slot = np.clip(loops.face_material, 0, len(plan) - 1)
    groups = {}
    for slot, entry in enumerate(plan):
        faces = np.flatnonzero(face_slot == slot)
        if not faces.size:
            continue
        key = tuple(entry["sources"][ch].key(cache.key, loops.default_uv) for ch in CHANNELS)
        key += (entry["displacement"],)
        group = groups.setdefault(key, _Group(entry["sources"]))
        group.displacement = entry["displacement"]
        group.slots.append(slot)
        group.faces.append(faces)
    for group in groups.values():
        group.faces = np.concatenate(group.faces)
    return list(groups.values())


def _pick_master(group, cache, default_uv, warn):
    best = None
    for channel in CHANNELS:
        src = group.sources[channel]
        if src.image is None:
            continue
        tex = cache.get(src.image)
        if best is None or tex.width * tex.height > best[0].width * best[0].height:
            best = (tex, src)
    if best is None:
        return None
    tex, src = best
    affine = _affine2x3(src.uv_affine)
    if abs(np.linalg.det(affine[:, :2])) < 1e-12:
        warn(f"a Mapping node on '{src.image.name}' collapses the UVs; it is ignored")
        affine = _affine2x3(None)
    master = _Master(tex, src.uv_layer or default_uv, affine, src.wrap)
    master.texture_name = src.image.name
    return master


def _normal_rotation(group, default_uv, warn):
    """Tangent-space fix-up for normal maps when a Mapping node rotates or
    mirrors the texture (moving the UVs into the atlas then turns the
    tangents). None when nothing needs to change, which is the usual case."""
    src = group.sources["Normal"]
    if src.image is None or group.master is None:
        return None
    if (src.tangent_uv or default_uv) != group.master.uv_layer:
        warn(f"normal map '{src.image.name}' uses a different UV map than its material's "
             "other textures; its tangents may not match")
        return None
    (a, b), (c, d) = group.master.affine[:, :2] * [[group.master.width], [group.master.height]]
    if a * d - b * c < 0:
        angle = math.atan2(b + c, a - d)
        rot = (math.cos(angle), math.sin(angle), math.sin(angle), -math.cos(angle))
    else:
        angle = math.atan2(c - b, a + d)
        rot = (math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle))
    return None if np.allclose(rot, (1.0, 0.0, 0.0, 1.0), atol=1e-6) else rot


def _components(edge_a, edge_b, count):
    """Connected components of ``count`` nodes joined by edges -> labels 0..k-1."""
    label = np.arange(count)
    while True:
        new = label.copy()
        np.minimum.at(new, edge_a, label[edge_b])
        np.minimum.at(new, edge_b, label[edge_a])
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    return np.unique(label, return_inverse=True)[1].ravel()


def _merge_overlaps(rects):
    """Merge overlapping rectangles until none overlap.

    ``rects`` is (n, 4) of x0, y0, x1, y1 (end exclusive). Returns the merged
    rectangles and, for every input rectangle, the index of its merged one.
    """
    owner = np.arange(len(rects))
    while True:
        n = len(rects)
        edges_a, edges_b = [], []
        for start in range(0, n, 512):
            blk = rects[start:start + 512]
            hit = ((blk[:, None, 0] < rects[None, :, 2]) & (rects[None, :, 0] < blk[:, None, 2]) &
                   (blk[:, None, 1] < rects[None, :, 3]) & (rects[None, :, 1] < blk[:, None, 3]))
            a, b = np.nonzero(hit)
            edges_a.append(a + start)
            edges_b.append(b)
        label = _components(np.concatenate(edges_a), np.concatenate(edges_b), n)
        k = int(label.max()) + 1
        if k == n:
            return rects, owner
        merged = np.empty((k, 4), np.int64)
        merged[:, :2] = np.iinfo(np.int64).max
        merged[:, 2:] = np.iinfo(np.int64).min
        for col, op in ((0, np.minimum), (1, np.minimum), (2, np.maximum), (3, np.maximum)):
            column = merged[:, col].copy()
            op.at(column, label, rects[:, col])
            merged[:, col] = column
        owner = label[owner]
        rects = merged


def _cut_chunks(group, loops, padding):
    """Find the texel rectangles the group's UV islands use."""
    idx, local_face = loops.loops_of(group.faces)
    uv = loops.uv(group.master.uv_layer)[idx]
    texel = group.master.to_texels(uv)

    # UV islands: faces that share a vertex with the same UV coordinate.
    quant = np.round(uv * (1 << 20)).astype(np.int64)
    keys = np.stack([loops.loop_vertex[idx], quant[:, 0], quant[:, 1]], axis=1)
    key_id = np.unique(keys, axis=0, return_inverse=True)[1].ravel()
    n_faces = len(group.faces)
    labels = _components(local_face, n_faces + key_id, n_faces + int(key_id.max()) + 1)
    island = np.unique(labels[:n_faces], return_inverse=True)[1].ravel()
    loop_island = island[local_face]

    n = int(island.max()) + 1
    lo = np.full((n, 2), np.inf)
    hi = np.full((n, 2), -np.inf)
    np.minimum.at(lo, loop_island, texel)
    np.maximum.at(hi, loop_island, texel)
    rects = np.empty((n, 4), np.int64)
    rects[:, :2] = np.floor(lo) - padding
    rects[:, 2:] = np.ceil(hi) + padding
    rects[:, 2:] = np.maximum(rects[:, 2:], rects[:, :2] + 1)

    merged, owner = _merge_overlaps(rects)
    group.chunks = [_Chunk(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in merged]
    group.loops = idx
    group.loop_texel = texel
    group.loop_chunk = owner[loop_island]


# ---------------------------------------------------------------------------
# Chunk pixels
# ---------------------------------------------------------------------------

def _ramp_lookup(lut, x):
    """Colour Ramp table lookup with linear interpolation: x (...) -> (..., 4)."""
    pos = np.clip(x, 0.0, 1.0) * (len(lut) - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), len(lut) - 2)
    f = (pos - i0)[..., None]
    return lut[i0] * (1.0 - f) + lut[i0 + 1] * f


def _apply_stage(linear, mask, scale, offset):
    """Grey value of ``linear`` (..., 4) through one mask / scale / offset stage."""
    weights = np.asarray(mask or _LUMINANCE, np.float32)
    return (linear @ weights) * scale[0] + offset[0]


def _apply(src, linear, kind):
    """Colour Ramp, channel selection and factor math on linear values."""
    if src.ramp is not None:
        weights, scale, offset = src.ramp_input
        linear = _ramp_lookup(src.ramp, (linear @ np.asarray(weights, np.float32)) * scale + offset)
    if src.mask is not None:
        v = linear @ np.asarray(src.mask, np.float32)
        v = (v * src.scale[0] + src.offset[0])[..., None]
        return v if kind == "linear" else np.repeat(v, 3, axis=-1)
    v = linear[..., :3] * np.asarray(src.scale[:3], np.float32) + np.asarray(src.offset[:3], np.float32)
    return v[..., :1] if kind == "linear" else v


def _encode(values, kind, rotation=None, strength=1.0):
    """Linear values -> 8-bit texels for the atlas."""
    if kind == "srgb":
        values = _linear_to_srgb(values)
    elif kind == "normal":
        n = values * 2.0 - 1.0
        x, y, z = n[..., 0], n[..., 1], np.maximum(n[..., 2], 1e-4)
        if rotation is not None:
            x, y = rotation[0] * x + rotation[1] * y, rotation[2] * x + rotation[3] * y
        x, y = x * strength, y * strength
        length = np.sqrt(x * x + y * y + z * z)
        values = np.stack([x, y, z], axis=-1) / length[..., None] * 0.5 + 0.5
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _verbatim_channels(src, tex, kind, rotation, decode=True):
    """Texel channels that can be copied byte for byte, or None when a
    conversion is needed (colour space, factor math, normal strength...)."""
    if tex.is_float or src.ramp is not None:
        return None
    if src.mask is None:
        if tuple(src.scale[:3]) != (1.0, 1.0, 1.0) or any(src.offset[:3]):
            return None
        if kind == "srgb" and tex.srgb:
            return [0, 1, 2]
        if kind == "normal" and src.strength == 1.0 and rotation is None:
            return [0, 1, 2]
        return None
    hot = [i for i, w in enumerate(src.mask) if w]
    if len(hot) != 1 or src.mask[hot[0]] != 1.0 or src.scale[0] != 1.0 or src.offset[0] != 0.0:
        return None
    k = hot[0]
    if kind == "linear" and (k == 3 or not (decode and tex.srgb)):
        return [k]
    if kind == "srgb" and k < 3 and tex.srgb:
        return [k, k, k]
    return None


def _resample(chunk, master, src, tex, decode):
    """Linear values of ``src`` on the chunk's atlas texels (bilinear)."""
    affine = _affine2x3(src.uv_affine)
    out = np.empty((chunk.ah, chunk.aw, 4), np.float32)
    # Source texels per atlas texel; shrink with an averaged copy to avoid aliasing.
    ratio = abs(np.linalg.det(affine[:, :2] @ np.linalg.inv(master.affine[:, :2])))
    footprint = math.sqrt(ratio * tex.width * tex.height / (master.width * master.height)) / chunk.scale
    k = int(footprint) if footprint >= 2.0 else 1
    k = max(1, min(k, tex.width, tex.height))
    xs = chunk.x0 + (np.arange(chunk.aw) + 0.5) / chunk.scale
    for row in range(0, chunk.ah, 256):
        ys = chunk.y0 + (np.arange(row, min(row + 256, chunk.ah)) + 0.5) / chunk.scale
        px, py = np.meshgrid(xs, ys)
        u, v = master.to_uv(px, py)
        tx = (affine[0, 0] * u + affine[0, 1] * v + affine[0, 2]) * tex.width
        ty = (affine[1, 0] * u + affine[1, 1] * v + affine[1, 2]) * tex.height
        out[row:row + len(ys)] = tex.bilinear(tx, ty, src.wrap, decode, k)
    return out


def _block(ctx, group, chunk, channel):
    """8-bit texels of ``channel`` for one chunk: (ah, aw, 3), or (ah, aw, 1)
    for Alpha / Roughness / Metallic."""
    kind = _KIND[channel]
    shape = (chunk.ah, chunk.aw, 1 if kind == "linear" else 3)
    if chunk.flat is not None:
        return np.broadcast_to(chunk.flat[channel], shape)
    src = group.sources[channel]
    rotation = group.normal_rotation if channel == "Normal" else None
    # Colour maps are always colour managed. Data maps tagged sRGB are decoded
    # too, like Blender renders them, unless raw values were asked for.
    # Normal maps are always read raw (tangent-space vectors).
    decode = kind == "srgb" or (kind == "linear" and not ctx.raw_data)
    strength = src.strength if channel == "Normal" else 1.0
    if src.image is None:
        value = np.asarray(src.offset[:shape[2]], np.float32).reshape(1, 1, -1)
        return np.broadcast_to(_encode(value, kind, rotation, strength), shape)

    tex = ctx.cache.get(src.image)
    master = group.master
    uv_layer = src.uv_layer or ctx.default_uv
    same_grid = (chunk.scale == 1.0 and tex.width == master.width and tex.height == master.height
                 and uv_layer == master.uv_layer
                 and np.allclose(_affine2x3(src.uv_affine), master.affine))
    if same_grid:
        raw = tex.block(chunk.x0, chunk.y0, chunk.w, chunk.h, src.wrap)
        channels = _verbatim_channels(src, tex, kind, rotation, decode)
        if channels == [0, 1, 2]:
            return raw[..., :3]
        if channels is not None:
            return raw[..., channels]
        linear = tex.linear(raw, decode)
    elif uv_layer != master.uv_layer:
        ctx.warn(f"'{src.image.name}' uses UV map '{uv_layer}' while its material's main texture "
                 f"uses '{master.uv_layer}'; its average colour is used")
        linear = tex.mean(decode).reshape(1, 1, 4)
    else:
        linear = _resample(chunk, master, src, tex, decode)
    encoded = _encode(_apply(src, linear, kind), kind, rotation, strength)
    return np.broadcast_to(encoded, shape)


def _flat_values(ctx, group, chunk, channels):
    """{channel: texel} if every channel of the chunk is one flat colour."""
    values = {}
    for channel in channels:
        block = _block(ctx, group, chunk, channel)
        first = block[0, 0].copy()
        if not (block == first).all():
            return None
        values[channel] = first
    return values


class _Context:
    def __init__(self, cache, default_uv, warn, raw_data=False):
        self.cache, self.default_uv, self.warn = cache, default_uv, warn
        self.raw_data = raw_data


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

class _MaxRects:
    """MaxRects bin packer (best short side fit), no rotation."""

    def __init__(self, size):
        self.free = [(0, 0, size, size)]

    def insert(self, w, h):
        best = None
        for fx, fy, fw, fh in self.free:
            if w <= fw and h <= fh:
                fit = (min(fw - w, fh - h), max(fw - w, fh - h))
                if best is None or fit < best[0]:
                    best = (fit, fx, fy)
        if best is None:
            return None
        _, x, y = best
        self._split(x, y, w, h)
        return x, y

    def _split(self, x, y, w, h):
        keep, new = [], []
        for rect in self.free:
            fx, fy, fw, fh = rect
            if x >= fx + fw or x + w <= fx or y >= fy + fh or y + h <= fy:
                keep.append(rect)
                continue
            if x > fx:
                new.append((fx, fy, x - fx, fh))
            if x + w < fx + fw:
                new.append((x + w, fy, fx + fw - x - w, fh))
            if y > fy:
                new.append((fx, fy, fw, y - fy))
            if y + h < fy + fh:
                new.append((fx, y + h, fw, fy + fh - y - h))

        def inside(a, b):
            return (a[0] >= b[0] and a[1] >= b[1] and
                    a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3])

        unique_new = []
        for i, a in enumerate(new):
            if any(inside(a, b) for b in keep):
                continue
            if any(inside(a, b) and (a != b or j < i) for j, b in enumerate(new) if j != i):
                continue
            unique_new.append(a)
        keep = [a for a in keep if not any(inside(a, b) for b in unique_new)]
        self.free = keep + unique_new


def _try_pack(chunks, size):
    order = sorted(range(len(chunks)),
                   key=lambda i: (max(chunks[i].aw, chunks[i].ah), chunks[i].aw * chunks[i].ah),
                   reverse=True)
    packer = _MaxRects(size)
    spots = [None] * len(chunks)
    for i in order:
        spot = packer.insert(chunks[i].aw, chunks[i].ah)
        if spot is None:
            return None
        spots[i] = spot
    return spots


def _pack(chunks, max_size, warn, log, lossless=False):
    """Place every chunk; returns the atlas size (smallest power of two that
    fits), or None in lossless mode when nothing up to ``max_size`` fits."""
    def place(spots):
        for chunk, (x, y) in zip(chunks, spots):
            chunk.ax, chunk.ay = x, y

    area = sum(c.aw * c.ah for c in chunks)
    side = max(max(c.aw, c.ah) for c in chunks)
    for size in _ATLAS_SIZES:
        if size > max_size:
            break
        if size * size < area or size < side:
            continue
        log(f"  trying {size} x {size}...")
        spots = _try_pack(chunks, size)
        if spots is not None:
            place(spots)
            return size
    if lossless:
        return None

    factor = min(1.0, math.sqrt(max_size * max_size / area))
    while True:
        factor *= 0.95
        for chunk in chunks:
            if chunk.flat is None:
                chunk.rescale(chunk.base_scale * factor)
        spots = _try_pack(chunks, max_size)
        if spots is not None:
            place(spots)
            warn(f"the textures need more room than {max_size} x {max_size}, so they were scaled "
                 f"to {factor:.0%}. Raise the maximum atlas size to keep full quality.")
            return max_size


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _png_bytes(pixels, alpha):
    """Encode (h, w, 4) uint8 pixels (bottom row first, like Blender) as PNG."""
    h, w, _ = pixels.shape
    c = 4 if alpha else 3
    rows = np.ascontiguousarray(pixels[::-1, :, :c]).reshape(h, w * c)
    filtered = np.empty((h, w * c + 1), np.uint8)
    filtered[:, 0] = 1  # "Sub" filter: store the difference to the texel on the left
    filtered[:, 1:c + 1] = rows[:, :c]
    np.subtract(rows[:, c:], rows[:, :-c], out=filtered[:, c + 1:])

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(filtered, 1)) + chunk(b"IEND", b""))


def _packed_image(name, png, is_color, path=None):
    """An image whose pixels live inside the .blend (packed PNG)."""
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    image = bpy.data.images.new(name, 1, 1)
    image.source = 'FILE'
    image.filepath = f"//{name}.png"
    if path:
        try:
            image.filepath = bpy.path.relpath(path) if bpy.data.filepath else path
        except ValueError:  # different drive on Windows
            image.filepath = path
    image.pack(data=png, data_len=len(png))
    image.buffers_free()  # drop the 1x1 placeholder; pixels now load from the packed PNG
    _set_colorspace(image, is_color)
    return image


def _master_material(name, images, has_alpha, emission_strength, displacement=None):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
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

    base = tex("BaseColor", 300)
    links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    if has_alpha:
        links.new(base.outputs["Alpha"], bsdf.inputs["Alpha"])
    links.new(tex("Metallic", 30).outputs["Color"], bsdf.inputs["Metallic"])
    links.new(tex("Roughness", -240).outputs["Color"], bsdf.inputs["Roughness"])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (-150, -520)
    links.new(tex("Normal", -520).outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    if "Emission" in images:
        links.new(tex("Emission", -800).outputs["Color"], _bsdf_input(bsdf, "Emission"))
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if "Height" in images and displacement is not None:
        disp = nodes.new('ShaderNodeDisplacement')
        disp.location = (100, -500)
        disp.inputs["Midlevel"].default_value, disp.inputs["Scale"].default_value = displacement
        links.new(tex("Height", -1080).outputs["Color"], disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    return mat


def _emission_peak(src, cache):
    if src.image is None:
        return max(src.offset[:3])
    tex = cache.get(src.image)
    top = float(tex.data[..., :3].max()) if tex.is_float else 1.0
    if src.ramp is not None:
        top = float(src.ramp[:, :3].max())
    return max(s * top + o for s, o in zip(src.scale[:3], src.offset[:3]))


def _room_report(groups, plan, size_limit):
    """Error text for lossless mode: which materials need how much room."""
    usage = []
    for g in groups:
        area = sum(c.w * c.h for c in g.chunks if c.flat is None)
        widest = max((max(c.w, c.h) for c in g.chunks), default=0)
        names = ", ".join(sorted({plan[s]["material"].name for s in g.slots
                                  if plan[s]["material"] is not None})) or "<empty slot>"
        usage.append((area, widest, names))
    usage.sort(reverse=True)
    total = sum(u[0] for u in usage)
    widest_all = max((u[1] for u in usage), default=0)
    needed = next((s for s in _ATLAS_SIZES + (32768, 65536)
                   if s * s >= total and s >= widest_all), 65536)
    lines = [f"Lossless: the textures do not fit in {size_limit} x {size_limit} "
             f"(they need at least {needed} x {needed}). Biggest users:"]
    for area, widest, names in usage[:5]:
        lines.append(f"  {names}: {area / 1e6:.1f} M texels (largest piece {widest} px)")
    lines.append("Turn off Lossless to shrink them to fit, raise Max Atlas Size, or reduce "
                 "texture tiling on those materials.")
    return "\n".join(lines)


def _group_details(groups, plan):
    """Per-material summary of what the packer did, for the build log."""
    lines = ["", "Per material:"]
    for g in groups:
        names = ", ".join(sorted({plan[s]["material"].name for s in g.slots
                                  if plan[s]["material"] is not None})) or "<empty slot>"
        lines.append(f"- {names}")
        if g.master is None:
            lines.append("    flat colour -> one small block")
            continue
        m = g.master
        lo, hi = g.loop_texel.min(axis=0), g.loop_texel.max(axis=0)
        span = (hi - lo) / (m.width, m.height)
        stored = sum(c.aw * c.ah for c in g.chunks)
        scales = [c.scale for c in g.chunks if c.flat is None]
        lines.append(f"    grid: '{m.texture_name}' {m.width}x{m.height}, UV map '{m.uv_layer}'")
        lines.append(f"    UVs cover {span[0]:.2f} x {span[1]:.2f} of the texture")
        lines.append(f"    {len(g.chunks)} piece(s), {sum(c.flat is not None for c in g.chunks)} flat, "
                     f"{stored / 1e6:.2f} M texels in the atlas")
        if scales and min(scales) < 1.0:
            lines.append(f"    SHRUNK: smallest piece scale {min(scales):.0%}")
        for c in sorted(g.chunks, key=lambda c: -c.aw * c.ah)[:3]:
            lines.append(f"      piece {c.w}x{c.h} texels at ({c.x0}, {c.y0}) -> "
                         f"{c.aw}x{c.ah} at atlas ({c.ax}, {c.ay})")
    return lines


def _write_text(name, lines):
    text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    text.clear()
    text.write("\n".join(lines))


def build_atlas(source, max_size=0, padding=8, output_dir=None, hide_source=True,
                lossless=True, raw_data=False, direct_normals=True, overrides=None, log=print):
    """Repack every material of ``source`` into one material with texture atlases.

    ``max_size`` 0 means Auto: the smallest power of two that holds every
    texel, up to AUTO_SIZE_LIMIT. With ``lossless`` nothing is ever scaled
    down; if the textures do not fit, a RuntimeError explains why. Roughness,
    Metallic, Alpha and Height maps tagged sRGB are gamma-decoded like Blender
    renders them, unless ``raw_data`` is set (game-engine style).

    Returns a dict with the new object, its images and statistics.
    """
    log_lines = []
    user_log = log

    def log(msg):
        log_lines.append(msg)
        user_log(msg)

    t_start = time.time()
    _check_source(source)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    out_dir = _output_dir(output_dir)
    if not max_size:
        max_size = AUTO_SIZE_LIMIT
        log("Atlas size: Auto (smallest size that keeps every texel)")

    warnings = []

    def warn(msg):
        if msg not in warnings:
            warnings.append(msg)
            log("  ! " + msg)

    plan = build_plan(source, direct_normals, overrides)
    for i, entry in enumerate(plan):
        for w in entry["warnings"]:
            warn(f"slot {i:02d}: {w}")

    cache = _TextureCache()
    loops = _MeshLoops(source.data)
    groups = _group_slots(plan, loops, cache)
    log(f"{len(plan)} material slot(s) -> {len(groups)} unique material(s)")

    peaks = [_emission_peak(g.sources["Emission"], cache) for g in groups]
    has_emission = any(p > 0.0 for p in peaks)
    emission_strength = max([1.0] + peaks)
    if emission_strength > 1.0:
        for g in groups:
            g.sources["Emission"].affine(1.0 / emission_strength)
    has_alpha = any(g.sources["Alpha"].image is not None or g.sources["Alpha"].offset[0] < 1.0
                    for g in groups)
    channels = ["BaseColor", "Roughness", "Metallic", "Normal"]
    channels += ["Alpha"] if has_alpha else []
    channels += ["Emission"] if has_emission else []

    # Displacement: one Displacement node drives the atlas. Each material moves
    # the surface by (height - midlevel) * scale; pick one (midlevel, scale)
    # whose 0..1 range covers every material, and re-express each height in it.
    # With identical settings everywhere this changes nothing.
    params = {g.displacement for g in groups if g.displacement is not None}
    ends = [e for m, sc in params for e in (-m * sc, (1.0 - m) * sc)]
    displacement = None
    if ends and max(ends) > min(ends):
        scale = max(ends) - min(ends)
        mid = -min(ends) / scale
        if len(params) == 1:
            mid, scale = next(iter(params))
        displacement = (mid, scale)
        for g in groups:
            if g.displacement is None:
                g.sources["Height"] = ChannelSource((mid, mid, mid, 1.0))
            elif g.displacement != displacement:
                m1, s1 = g.displacement
                k = s1 / scale
                g.sources["Height"].affine(k, mid - m1 * k)
        _BACKGROUND["Height"] = (int(round(min(max(mid, 0.0), 1.0) * 255)),) * 3
        channels.append("Height")

    ctx = _Context(cache, loops.default_uv, warn, raw_data)
    flat_size = max(4, 2 * padding)
    for g in groups:
        g.master = _pick_master(g, cache, loops.default_uv, warn)
        if g.master is None:  # no textures at all: one small flat block
            chunk = _Chunk(0, 0, flat_size, flat_size)
            chunk.make_flat(_flat_values(ctx, g, chunk, channels), flat_size)
            g.chunks = [chunk]
            g.loops = loops.loops_of(g.faces)[0]
            continue
        g.normal_rotation = _normal_rotation(g, loops.default_uv, warn)
        _cut_chunks(g, loops, padding)
        for chunk in g.chunks:
            if max(chunk.w, chunk.h) > max_size and not lossless:
                chunk.base_scale = max_size / max(chunk.w, chunk.h)
                chunk.rescale(chunk.base_scale)
                names = ", ".join(sorted({plan[s]["material"].name for s in g.slots
                                          if plan[s]["material"] is not None}))
                warn(f"'{names}' needs a {chunk.w} x {chunk.h} texel region (tiled texture?), "
                     f"larger than {max_size}; that part was scaled down to fit")
            if chunk.aw * chunk.ah > flat_size * flat_size:
                flat = _flat_values(ctx, g, chunk, channels)
                if flat is not None:
                    chunk.make_flat(flat, flat_size)

    chunks = [c for g in groups for c in g.chunks]
    log(f"Packing {len(chunks)} chunk(s)...")
    fits = all(max(c.aw, c.ah) <= max_size for c in chunks)
    size = _pack(chunks, max_size, warn, log, lossless) if fits else None
    if size is None:
        message = _room_report(groups, plan, max_size)
        _write_text(LOG_TEXT, log_lines + ["", message] + _group_details(groups, plan))
        raise RuntimeError(message)
    filled = sum(c.aw * c.ah for c in chunks) / float(size * size)

    # New UVs: every loop moves with its chunk; nothing is re-unwrapped.
    new_uv = np.zeros((len(source.data.loops), 2), np.float64)
    for g in groups:
        if g.master is None:
            c = g.chunks[0]
            new_uv[g.loops] = ((c.ax + c.aw / 2.0) / size, (c.ay + c.ah / 2.0) / size)
            continue
        p = np.array([(c.x0, c.y0, c.scale, c.ax, c.ay, c.aw, c.ah, c.flat is not None)
                      for c in g.chunks], np.float64)[g.loop_chunk]
        uv = ((g.loop_texel - p[:, 0:2]) * p[:, 2:3] + p[:, 3:5]) / size
        flat = p[:, 7] > 0
        uv[flat] = (p[flat, 3:5] + p[flat, 5:7] / 2.0) / size
        new_uv[g.loops] = uv

    target = _make_target(source)
    mesh = target.data
    for name in [l.name for l in mesh.uv_layers]:
        mesh.uv_layers.remove(mesh.uv_layers[name])
    layer = mesh.uv_layers.new(name=ATLAS_UV_NAME)
    layer.data.foreach_set("uv", new_uv.astype(np.float32).ravel())

    # Build each atlas and compress it to PNG on worker threads (zlib runs in
    # parallel) while the next atlas is being assembled.
    jobs = []
    workers = 2 if size >= 16384 else 4  # 16K atlases take 1 GB each while compressing
    with ThreadPoolExecutor(max_workers=min(workers, os.cpu_count() or 1)) as pool:
        for name, color, alpha in _OUTPUTS:
            if color not in channels:
                continue
            log(f"Writing {name} atlas ({size} x {size})...")
            atlas = np.empty((size, size, 4), np.uint8)
            atlas[..., :3] = _BACKGROUND[name]
            atlas[..., 3] = 255
            use_alpha = alpha is not None and has_alpha
            for g in groups:
                for c in g.chunks:
                    region = atlas[c.ay:c.ay + c.ah, c.ax:c.ax + c.aw]
                    region[..., :3] = _block(ctx, g, c, color)
                    if use_alpha:
                        region[..., 3] = _block(ctx, g, c, alpha)[..., 0]
            jobs.append((name, color, pool.submit(_png_bytes, atlas, use_alpha)))
            del atlas

        images, paths = {}, {}
        for name, color, job in jobs:
            png = job.result()
            path = os.path.join(out_dir, f"{target.name}_{name}_{size}.png") if out_dir else None
            if path:
                with open(path, "wb") as fh:
                    fh.write(png)
                paths[name] = path
                log(f"  saved {path}")
            images[name] = _packed_image(f"{target.name}_{name}_{size}", png,
                                         color in _COLOR_CHANNELS, path)

    for slot in target.material_slots:
        slot.link = 'DATA'
    mesh.materials.clear()
    mesh.materials.append(_master_material(f"M_{target.name}", images, has_alpha,
                                           emission_strength, displacement))
    mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), np.int32))
    mesh.update()

    if hide_source:
        source.hide_set(True)
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    target.hide_set(False)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target

    seconds = time.time() - t_start
    flat_count = sum(c.flat is not None for c in chunks)
    log(f"Done in {seconds:.1f} s: {size} x {size} atlas, {len(chunks)} chunk(s) "
        f"({flat_count} flat), {filled:.0%} filled -> '{target.name}'")
    _write_text(LOG_TEXT, log_lines + _group_details(groups, plan))
    return {"object": target, "images": images, "paths": paths, "size": size,
            "groups": len(groups), "chunks": len(chunks), "flat_chunks": flat_count,
            "filled": filled, "warnings": len(warnings), "seconds": seconds}


# ---------------------------------------------------------------------------
# Run from the Text Editor
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OBJECT_NAME = ""        # "" = the active object, e.g. "guitar.036"
    MAX_ATLAS_SIZE = 0      # 0 = Auto; else the smallest power of two that fits, up to this
    LOSSLESS = True         # never shrink textures; stop with an explanation instead
    RAW_DATA = False        # True = sRGB-tagged roughness/metallic maps keep raw values
    DIRECT_NORMALS = True   # use normal textures plugged in without a Normal Map node
    PADDING = 8             # texels of real texture kept around every UV island
    OUTPUT_DIR = None       # None = textures only inside the .blend;
                            # or a folder for PNG copies, e.g. "//atlas_textures/"
    DIAGNOSE_ONLY = False   # True = only write the report

    source_obj = bpy.data.objects.get(OBJECT_NAME) if OBJECT_NAME else bpy.context.active_object
    print(write_report(source_obj))
    if not DIAGNOSE_ONLY:
        build_atlas(source_obj, max_size=MAX_ATLAS_SIZE, padding=PADDING, output_dir=OUTPUT_DIR,
                    lossless=LOSSLESS, raw_data=RAW_DATA, direct_normals=DIRECT_NORMALS)
