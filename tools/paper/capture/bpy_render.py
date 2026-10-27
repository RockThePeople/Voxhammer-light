"""논문 캡처 렌더(Blender, EVAL_PY 의 bpy). cwd = Edit3D-Bench 소스(HDRI 상대경로).

Bench render.py 와 같은 설정을 쓴다: Cycles(64 샘플, 디노이즈, 확산·광택 반사 1회), HDRI `assets/brown_photostudio_02_1k.exr`,
카메라 반지름 1.8·렌즈 35 mm·센서 32 mm(원점 주시), bpyrenderer 의 불러오기·불투명 재질 처리. 다른 점: GPU(CUDA), 시점은 job 이 준다.
albedo: 재질마다 Principled BSDF 의 Base Color 입력(텍스처·정점색 연결 포함)을 Emission(세기 1) 로 바로 출력하고 월드 조명 0,
색 변환 Standard — 조명 없는 표면 색. shaded 는 Blender 기본 색 변환(Bench 와 같음).
출력은 투명 배경 RGBA PNG(흰 배경 합성은 호출 쪽).

    python bpy_render.py jobs.json
jobs.json = [{"glb": N 좌표 GLB, "res": 1024, "views": [{"elev": 도, "az": Bench 방위(도), "shaded": png, "albedo": png}]}]
"""
import json
import sys

import bpy
import numpy as np
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_env_map
from bpyrenderer.importer import load_file
from mathutils import Matrix


def gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                bpy.context.scene.cycles.device = "GPU"
                return kind
        except Exception:  # noqa: BLE001
            continue
    return "CPU"


def camera(elev, az):
    _, mats, _, _ = get_camera_positions_on_sphere(center=(0, 0, 0), radius=1.8, elevations=[elev], azimuths=[az - 90])
    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        bpy.context.scene.camera = bpy.context.object
    cam = bpy.context.scene.camera
    cam.data.type = "PERSP"
    cam.data.sensor_width = 32
    cam.data.lens = 35
    cam.matrix_world = Matrix(np.asarray(mats[0]))


def to_albedo():
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if out is None or bsdf is None:
            continue
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        bc = bsdf.inputs["Base Color"]
        if bc.is_linked:
            nt.links.new(bc.links[0].from_socket, em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = bc.default_value
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    w = bpy.context.scene.world
    if w and w.use_nodes:
        for n in w.node_tree.nodes:
            if n.type == "BACKGROUND":
                n.inputs["Strength"].default_value = 0.0
    bpy.context.scene.view_settings.view_transform = "Standard"


def main(jobs_path):
    jobs = json.load(open(jobs_path))
    init_render_engine("CYCLES")
    dev = gpu()
    print("[cap] device", dev, flush=True)
    sm = SceneManager()
    default_vt = bpy.context.scene.view_settings.view_transform
    for job in jobs:
        sm.clear(reset_keyframes=True)
        bpy.context.scene.camera = None
        load_file(job["glb"])
        sm.clear_normal_map()
        sm.set_material_transparency(False)
        sm.set_materials_opaque()
        set_env_map("assets/brown_photostudio_02_1k.exr")
        bpy.context.scene.view_settings.view_transform = default_vt
        sc = bpy.context.scene
        sc.render.resolution_x = sc.render.resolution_y = job.get("res", 1024)
        sc.render.resolution_percentage = 100
        sc.render.film_transparent = True
        sc.render.use_compositing = False      # SceneManager 가 켠 합성 노드는 Render Layers 연결이 없어 빈 화면이 된다
        sc.use_nodes = False
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        for v in job["views"]:
            camera(v["elev"], v["az"])
            if v.get("shaded"):
                sc.render.filepath = v["shaded"]
                bpy.ops.render.render(write_still=True)
        if any(v.get("albedo") for v in job["views"]):
            to_albedo()
            for v in job["views"]:
                if v.get("albedo"):
                    camera(v["elev"], v["az"])
                    sc.render.filepath = v["albedo"]
                    bpy.ops.render.render(write_still=True)
        print("[cap] done", job["glb"], flush=True)


if __name__ == "__main__":
    main(sys.argv[-1])
