# SPDX-License-Identifier: GPL-3.0-or-later
"""PBR Atlas Baker - sidebar UI (View3D > Sidebar > Atlas)."""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty

from . import baker


class PBRAtlasSettings(bpy.types.PropertyGroup):
    max_size: EnumProperty(
        name="Atlas Size",
        description="Auto picks the smallest power-of-two size that holds every texture at "
                    "full resolution. A number caps the size at that value",
        items=[('AUTO', "Auto", "Smallest size that keeps every texel (up to 16384)")]
              + [(str(s), f"Max {s} x {s}", "") for s in (2048, 4096, 8192, 16384)],
        default='AUTO',
    )
    lossless: BoolProperty(
        name="Lossless", default=True,
        description="Never shrink textures. If they do not fit, stop and explain which "
                    "materials need the room instead of lowering quality",
    )
    raw_data: BoolProperty(
        name="Raw Data Maps", default=False,
        description="Roughness / Metallic / Height textures tagged sRGB keep their raw values "
                    "(like a game engine). Off: they look exactly like in Blender",
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


# Last "Check Materials" result, shown in the Material Check panel.
_CHECK = {"object": None, "rows": []}


def _active_mesh(context):
    obj = context.active_object
    return obj if obj is not None and obj.type == 'MESH' else None


class PBRATLAS_OT_check(bpy.types.Operator):
    """List every material slot and what is plugged into each channel (also written to the Text Editor)"""

    bl_idname = "pbr_atlas.check"
    bl_label = "Check Materials"

    @classmethod
    def poll(cls, context):
        return _active_mesh(context) is not None

    def execute(self, context):
        obj = _active_mesh(context)
        rows = baker.check_materials(obj)
        _CHECK["object"], _CHECK["rows"] = obj.name, rows
        print(baker.write_report(obj))
        warnings = sum(len(r["warnings"]) for r in rows)
        level = {'WARNING'} if warnings else {'INFO'}
        self.report(level, f"{len(rows)} material slot(s), {warnings} warning(s). "
                           f"Full report in Text Editor: {baker.REPORT_TEXT}")
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
                max_size=0 if settings.max_size == 'AUTO' else int(settings.max_size),
                lossless=settings.lossless,
                raw_data=settings.raw_data,
                padding=settings.padding,
                output_dir=settings.output_dir if settings.save_files else None,
                hide_source=settings.hide_source,
                log=lambda msg: print("[PBR Atlas]", msg),
            )
        except RuntimeError as exc:
            print("[PBR Atlas]", exc)
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
        layout.prop(settings, "lossless")
        layout.prop(settings, "raw_data")
        layout.prop(settings, "padding")
        layout.prop(settings, "hide_source")
        layout.prop(settings, "save_files")
        if settings.save_files:
            layout.prop(settings, "output_dir")

        layout.separator()
        layout.operator(PBRATLAS_OT_check.bl_idname, icon='VIEWZOOM')
        layout.operator(PBRATLAS_OT_build.bl_idname, icon='TEXTURE')


class PBRATLAS_PT_check(bpy.types.Panel):
    bl_label = "Material Check"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Atlas"
    bl_parent_id = "PBRATLAS_PT_panel"

    def draw(self, context):
        layout = self.layout
        rows = _CHECK["rows"]
        if not rows:
            layout.label(text="Click Check Materials to list them", icon='INFO')
            return
        emitting = sum(r["emission"] is not None for r in rows)
        displaced = sum(r["displacement"] is not None for r in rows)
        warnings = sum(len(r["warnings"]) for r in rows)
        col = layout.column(align=True)
        col.label(text=f"'{_CHECK['object']}': {len(rows)} slot(s)", icon='OBJECT_DATA')
        col.label(text=f"{emitting} with emission, {displaced} with displacement", icon='LIGHT')
        if warnings:
            col.label(text=f"{warnings} warning(s)", icon='ERROR')

        for row in rows:
            box = layout.box()
            col = box.column(align=True)
            col.label(text=f"{row['name']}  ({row['faces']} faces)", icon='MATERIAL')
            for label, text in row["textures"]:
                col.label(text=f"{label}: {text}", icon='TEXTURE')
            if row["emission"] is not None:
                col.label(text=f"Emission: {row['emission']}", icon='LIGHT')
            if row["displacement"] is not None:
                col.label(text=f"Displacement: {row['displacement']}", icon='MOD_DISPLACE')
            if row["constants"]:
                text = ", ".join(f"{label} {value}" for label, value in row["constants"])
                col.label(text=f"Values: {text}", icon='COLOR')
            flags = []
            if row["emission"] is None:
                flags.append("no emission")
            if row["displacement"] is None:
                flags.append("no displacement")
            if flags:
                col.label(text=", ".join(flags).capitalize(), icon='BLANK1')
            for warning in row["warnings"]:
                sub = col.row()
                sub.alert = True
                sub.label(text=warning, icon='ERROR')


_CLASSES = (PBRAtlasSettings, PBRATLAS_OT_check, PBRATLAS_OT_build, PBRATLAS_PT_panel,
            PBRATLAS_PT_check)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pbr_atlas = PointerProperty(type=PBRAtlasSettings)


def unregister():
    del bpy.types.Scene.pbr_atlas
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
