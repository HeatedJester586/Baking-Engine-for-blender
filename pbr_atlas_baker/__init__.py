# SPDX-License-Identifier: GPL-3.0-or-later
"""PBR Atlas Baker - sidebar UI (View3D > Sidebar > Atlas)."""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty

from . import baker


class PBRAtlasSettings(bpy.types.PropertyGroup):
    max_size: EnumProperty(
        name="Max Atlas Size",
        description="The smallest power-of-two size that holds every texture at full "
                    "resolution is used, up to this size",
        items=[(str(s), f"{s} x {s}", "") for s in (2048, 4096, 8192, 16384)],
        default='8192',
    )
    padding: IntProperty(
        name="Padding", subtype='PIXEL', default=8, min=0, max=64,
        description="Texels of real texture kept around every UV island, so mipmaps do not bleed",
    )
    hide_source: BoolProperty(
        name="Hide Original", default=True,
        description="Hide the original object afterwards",
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


class PBRATLAS_OT_build(bpy.types.Operator):
    """Copy the active mesh and repack all of its materials into one material with texture atlases"""

    bl_idname = "pbr_atlas.build"
    bl_label = "Build Atlas"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_mesh(context) is not None

    def execute(self, context):
        settings = context.scene.pbr_atlas
        try:
            result = baker.build_atlas(
                _active_mesh(context),
                max_size=int(settings.max_size),
                padding=settings.padding,
                output_dir=settings.output_dir if settings.save_files else None,
                hide_source=settings.hide_source,
                log=lambda msg: print("[PBR Atlas]", msg),
            )
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        msg = (f"'{result['object'].name}': {result['size']} x {result['size']} atlas, "
               f"{result['groups']} material(s), {result['filled']:.0%} filled")
        if result["warnings"]:
            self.report({'WARNING'}, msg + f", {result['warnings']} warning(s) - see the console")
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

        layout.prop(settings, "max_size")
        layout.prop(settings, "padding")
        layout.prop(settings, "hide_source")
        layout.prop(settings, "save_files")
        if settings.save_files:
            layout.prop(settings, "output_dir")

        layout.separator()
        layout.operator(PBRATLAS_OT_diagnose.bl_idname, icon='VIEWZOOM')
        layout.operator(PBRATLAS_OT_build.bl_idname, icon='TEXTURE')


_CLASSES = (PBRAtlasSettings, PBRATLAS_OT_diagnose, PBRATLAS_OT_build, PBRATLAS_PT_panel)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pbr_atlas = PointerProperty(type=PBRAtlasSettings)


def unregister():
    del bpy.types.Scene.pbr_atlas
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
