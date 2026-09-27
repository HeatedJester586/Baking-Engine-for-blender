# SPDX-License-Identifier: GPL-3.0-or-later
"""PBR Atlas Baker - sidebar UI (View3D > Sidebar > Atlas)."""

import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import baker


class PBRAtlasSettings(bpy.types.PropertyGroup):
    method: EnumProperty(
        name="Method",
        items=(
            ('GPU', "GPU Rasterizer", "Copy every texture straight into the atlas on the GPU. "
                                      "No rays, so floating parts cannot cast noise"),
            ('CYCLES', "Cycles Surface Bake", "Native Cycles bake of each surface onto itself"),
        ),
        default='GPU',
    )
    resolution: EnumProperty(
        name="Resolution",
        items=[(str(r), f"{r} x {r}", "") for r in (1024, 2048, 4096, 8192)],
        default='4096',
    )
    margin: IntProperty(
        name="Edge Padding", subtype='PIXEL', default=16, min=0, max=128,
        description="Pixels to extend each UV island by, so mipmaps do not bleed background colour",
    )
    angle_limit: FloatProperty(
        name="Angle Limit", subtype='ANGLE', default=math.radians(66.0),
        min=math.radians(1.0), max=math.radians(89.0),
        description="Smart UV Project angle limit for the atlas UV map",
    )
    island_margin: FloatProperty(
        name="Island Margin", default=0.005, min=0.0, max=0.1, precision=4,
        description="Space between UV islands in the atlas (UV units)",
    )
    save_files: BoolProperty(
        name="Also Save PNG Files", default=False,
        description="The atlases are always stored inside the .blend. Enable this to also "
                    "write them to a folder as PNG files",
    )
    output_dir: StringProperty(
        name="Folder", subtype='DIR_PATH', default="//atlas_textures/",
        description="Where the PNG copies are written ('//' = next to the .blend file)",
    )
    hide_source: BoolProperty(
        name="Hide Original", default=True,
        description="Hide the original object after baking",
    )
    samples: IntProperty(
        name="Samples", default=8, min=1, max=1024,
        description="Cycles samples per pixel (anti-aliasing only; no lighting is baked)",
    )
    clear_custom_normals: BoolProperty(
        name="Clear Custom Normals", default=True,
        description="Remove custom split normals from the copy before the Cycles normal bake",
    )


def _active_mesh(context):
    obj = context.active_object
    return obj if obj is not None and obj.type == 'MESH' else None


class PBRATLAS_OT_diagnose(bpy.types.Operator):
    """Write a report of every material slot and where each PBR channel comes from"""

    bl_idname = "pbr_atlas.diagnose"
    bl_label = "Diagnose Materials"

    @classmethod
    def poll(cls, context):
        return _active_mesh(context) is not None

    def execute(self, context):
        obj = _active_mesh(context)
        report = baker.write_report(obj)
        print(report)
        warnings = sum(len(e["warnings"]) for e in baker.build_plan(obj))
        level = {'WARNING'} if warnings else {'INFO'}
        self.report(level, f"{warnings} warning(s). Full report in Text Editor: {baker.REPORT_TEXT}")
        return {'FINISHED'}


class PBRATLAS_OT_bake(bpy.types.Operator):
    """Merge all materials of the active mesh into one set of PBR atlases on a copy of it"""

    bl_idname = "pbr_atlas.bake"
    bl_label = "Bake Atlas"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_mesh(context) is not None and context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.pbr_atlas
        source = _active_mesh(context)
        if source.get(baker.MARKER):
            self.report({'ERROR'}, f"'{source.name}' is already a baked atlas; select the original object")
            return {'CANCELLED'}
        options = dict(
            resolution=int(settings.resolution),
            margin=settings.margin,
            angle_limit=settings.angle_limit,
            island_margin=settings.island_margin,
            output_dir=settings.output_dir if settings.save_files else None,
            hide_source=settings.hide_source,
            log=lambda msg: print("[PBR Atlas]", msg),
        )
        try:
            if settings.method == 'CYCLES':
                result = baker.bake_cycles_atlas(
                    source, samples=settings.samples,
                    clear_custom_normals=settings.clear_custom_normals, **options)
            else:
                result = baker.bake_gpu_atlas(source, **options)
        except Exception as exc:  # show the reason in the UI instead of a traceback
            self.report({'ERROR'}, str(exc))
            raise

        msg = f"Baked '{result['object'].name}' in {result['seconds']:.1f} s"
        if result["warnings"]:
            self.report({'WARNING'}, msg + f" with {result['warnings']} warning(s) - run Diagnose")
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class PBRATLAS_PT_panel(bpy.types.Panel):
    bl_label = "PBR Atlas Baker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Atlas"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.pbr_atlas
        obj = _active_mesh(context)

        box = layout.box()
        if obj is None:
            box.label(text="Select a mesh object", icon='ERROR')
        else:
            box.label(text=obj.name, icon='MESH_DATA')
            box.label(text=f"{len(obj.material_slots)} material slot(s)")

        layout.prop(settings, "method")
        layout.prop(settings, "resolution")
        layout.prop(settings, "margin")

        col = layout.column(heading="Atlas UVs")
        col.prop(settings, "angle_limit")
        col.prop(settings, "island_margin")

        if settings.method == 'CYCLES':
            col = layout.column(heading="Cycles")
            col.prop(settings, "samples")
            col.prop(settings, "clear_custom_normals")

        layout.prop(settings, "hide_source")
        layout.prop(settings, "save_files")
        if settings.save_files:
            layout.prop(settings, "output_dir")

        layout.separator()
        layout.operator(PBRATLAS_OT_diagnose.bl_idname, icon='VIEWZOOM')
        layout.operator(PBRATLAS_OT_bake.bl_idname, icon='RENDER_STILL')


_CLASSES = (PBRAtlasSettings, PBRATLAS_OT_diagnose, PBRATLAS_OT_bake, PBRATLAS_PT_panel)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pbr_atlas = PointerProperty(type=PBRAtlasSettings)


def unregister():
    del bpy.types.Scene.pbr_atlas
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
