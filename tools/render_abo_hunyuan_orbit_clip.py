"""Render and verify the first HunyuanVideo ABO turntable media sample."""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time


REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "tools/render_abo_fixed_camera_rgba.py"
REFERENCE_SHA = "47792466c588c851621943b399ac0a5437a26a5a67b449c7f016b5e15eab53c2"
LAYOUT = REPO / "results/bw1000/2026-11-27/specialized-hunyuanvideo-original-native-cache-9-frame-media-and-tensor-layout.json"
LAYOUT_SHA = "d77e2e1feaf37ce495f915a51922f2cebc977f425468b615af75d2787dc74c9b"
ASSET = "B07M6PHM8C"
PLAN = dict(frames=9, height=480, width=848)
AZIMUTHS = tuple(range(0, 121, 15))
FPS = 8
SAMPLES = 24
CAPTION = "A centered object in a neutral studio as the camera makes a smooth orbit around it under steady lighting."


def digest(raw):
    return sha256(raw).hexdigest()


def need(condition, error):
    if not condition:
        raise ValueError(error)


def exact_references():
    need(digest(REFERENCE.read_bytes()) == REFERENCE_SHA and digest(LAYOUT.read_bytes()) == LAYOUT_SHA,
         "Original renderer and measured training layout must both match")
    need(json.loads(LAYOUT.read_bytes())["original_media_dimensions"] == PLAN,
         "Original video input dimensions changed")
    return runpy.run_path(str(REFERENCE))


def frame_name(index):
    return f"frame-{index:02d}.private.png"


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def durable(stream, data):
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())


def render(glb, output):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector

    original = exact_references()
    need(bpy.app.background and bpy.app.version[:2] == (4, 5), "Blender 4.5 background is required")
    raw = original["read"](glb, original["GLB"]["bytes"])
    need(original["pin"](raw) == original["GLB"], "Publisher GLB changed")
    original["declared_material"](raw)
    fixed, reference = original["source_camera"]()
    axis_map = original["source_to_blender"]
    output.mkdir(mode=0o700)
    intent = dict(schema="ABO_HUNYUAN_TURNTABLE_MEDIA_V1", object=ASSET, source=original["GLB"],
                  original_cache_layout_SHA256=LAYOUT_SHA, plan=PLAN, azimuth_degrees=AZIMUTHS,
                  fps=FPS, frames_are_synthetic_renderings=True, dataset_split="training_candidate_only",
                  captions=[CAPTION], planned_media_examples=1, planned_native_cache_rows=0)
    with (output / "intent.json").open("xb") as stream:
        durable(stream, json_bytes(intent))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    need(bpy.ops.import_scene.gltf(filepath=str(glb.resolve())) == {"FINISHED"}, "Publisher glTF import failed")
    need(original["pin"](original["read"](glb, original["GLB"]["bytes"])) == original["GLB"],
         "Publisher GLB changed while importing")
    mesh, = [item for item in scene.objects if item.type == "MESH"]
    need(len(mesh.data.polygons) == 23592 and all(len(face.vertices) == 3 for face in mesh.data.polygons),
         "Publisher geometry differs")
    material, = list(mesh.data.materials)
    _, color = original["material_shader"](material)

    center = Vector(axis_map(reference["reference_center_m"]))
    extent = reference["reference_longest_extent_m"]
    start_eye = Vector(axis_map(fixed["eye"]))
    start_right = Vector(axis_map(fixed["right"]))
    start_up = Vector(axis_map(fixed["sensor_up"]))
    camera = bpy.data.objects.new("ABO orbit", bpy.data.cameras.new("ABO orbit orthographic"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.9 * extent
    camera.data.clip_start, camera.data.clip_end = extent * 0.01, extent * 10

    world = bpy.data.worlds.new("neutral studio")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs[0].default_value = (0.15, 0.15, 0.15, 1.0)
    background.inputs[1].default_value = 0.35
    scene.world = world
    for name, offset, watts, diameter in (("key", (1., -2., 2.), 25., 2.), ("fill", (-1.75, .4, .9), 12., 2.5)):
        light = bpy.data.lights.new(name, "AREA")
        fixture = bpy.data.objects.new(name, light)
        scene.collection.objects.link(fixture)
        fixture.location = center + extent * Vector(offset)
        fixture.rotation_euler = (center - fixture.location).to_track_quat("-Z", "Y").to_euler()
        light.energy, light.shape, light.size = watts, "DISK", extent * diameter

    settings = scene.render
    settings.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples, scene.cycles.seed = SAMPLES, 0
    scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
    settings.threads_mode, settings.threads = "FIXED", 4
    settings.resolution_x, settings.resolution_y = PLAN["width"], PLAN["height"]
    settings.resolution_percentage = 100
    settings.pixel_aspect_x = settings.pixel_aspect_y = 1
    settings.fps = FPS
    settings.film_transparent = False
    settings.image_settings.file_format = "PNG"
    settings.image_settings.color_mode = "RGB"
    settings.image_settings.color_depth = "8"

    artifacts = []
    began = time.perf_counter()
    with (output / "frames.jsonl").open("xb") as ledger:
        for index, degrees in enumerate(AZIMUTHS):
            turn = Matrix.Rotation(math.radians(degrees), 3, "Z")
            eye = center + extent * (turn @ start_eye)
            right, up = turn @ start_right, turn @ start_up
            backward = (eye - center).normalized()
            camera_axes = Matrix([right, up, backward]).transposed()
            need(abs(camera_axes.determinant() - 1) < 1e-6, "Camera basis is not a proper rotation")
            camera.matrix_world = Matrix.Translation(eye) @ camera_axes.to_4x4()
            bpy.context.view_layer.update()
            target = world_to_camera_view(scene, camera, center)
            need(abs(target.x - .5) < 1e-6 and abs(target.y - .5) < 1e-6, "Camera lost the reference center")
            frame = frame_name(index)
            settings.filepath = str((output / frame).resolve())
            need(bpy.ops.render.render(write_still=True) == {"FINISHED"}, "CPU frame render failed")
            frame_raw = original["read"](output / frame, 2 << 20)
            item = dict(index=index, azimuth_degrees=degrees, name=frame, pin=original["pin"](frame_raw),
                        camera_to_world=[list(row) for row in camera.matrix_world],
                        target_center_ndc=[target.x, target.y])
            durable(ledger, json_bytes(item))
            artifacts.append(item)
    result = dict(intent_SHA256=digest((output / "intent.json").read_bytes()), frame_count=len(artifacts),
                  ledger_SHA256=digest((output / "frames.jsonl").read_bytes()), source_albedo_SHA256=color,
                  elapsed_seconds=time.perf_counter() - began, renderer=bpy.app.version_string, engine=settings.engine,
                  device=scene.cycles.device, samples=SAMPLES, gpu_cards=0, native_cache_rows=0)
    with (output / "render.json").open("xb") as stream:
        durable(stream, json_bytes(result))
    print(json.dumps(result, sort_keys=True))


def verify(output):
    import io
    from PIL import Image

    original = exact_references()
    intent_raw = (output / "intent.json").read_bytes()
    manifest = json.loads(intent_raw)
    ledger_raw = (output / "frames.jsonl").read_bytes()
    rows = [json.loads(line) for line in ledger_raw.splitlines()]
    result = json.loads((output / "render.json").read_bytes())
    need(manifest["source"] == original["GLB"] and manifest["plan"] == PLAN
         and manifest["captions"] == [CAPTION] and manifest["dataset_split"] == "training_candidate_only"
         and manifest["fps"] == FPS and manifest["original_cache_layout_SHA256"] == LAYOUT_SHA
         and result["intent_SHA256"] == digest(intent_raw) and result["ledger_SHA256"] == digest(ledger_raw)
         and result["device"] == "CPU" and result["frame_count"] == len(rows) == PLAN["frames"],
         "Incomplete or inconsistent media manifest")
    rgb_hashes = []
    cameras = set()
    for i, row in enumerate(rows):
        need(row["index"] == i and row["azimuth_degrees"] == AZIMUTHS[i] and row["name"] == frame_name(i),
             "Frame order or angular spacing differs")
        matrix = row["camera_to_world"]
        cameras.add(tuple(column[3] for column in matrix[:3]))
        need(len(matrix) == 4 and matrix[3] == [0.0, 0.0, 0.0, 1.0]
             and all(abs(v - .5) <= 1e-6 for v in row["target_center_ndc"]), "Camera framing differs")
        raw = original["read"](output / row["name"], 2 << 20)
        need(original["pin"](raw) == row["pin"], "Recorded frame digest differs")
        with Image.open(io.BytesIO(raw), formats=("PNG",)) as frame:
            need(frame.mode == "RGB" and frame.size == (PLAN["width"], PLAN["height"]), "Frame dimensions or RGB differ")
            need(any(lo != hi for lo, hi in frame.getextrema()), "Frame has no visible spatial structure")
            rgb_hashes.append(digest(frame.tobytes()))
    need(len(set(rgb_hashes)) == len(cameras) == PLAN["frames"], "Repeated decoded image or camera position")
    print(json.dumps(dict(status="VERIFIED_DOMAIN_MEDIA_SAMPLE", media_examples=1, frames=len(rows),
                          width=PLAN["width"], height=PLAN["height"], chosen_fps=FPS, rgb_SHA256=rgb_hashes,
                          source_glb=original["GLB"], executed_renderer=result["renderer"],
                          actual_GPU=0, native_cache_rows=0, optimizer_updates=0), sort_keys=True))


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    create = commands.add_parser("render")
    create.add_argument("--glb", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    local = (REPO / "local-only").resolve()
    need(output.is_relative_to(local) and output != local, "Private output must be under local-only")
    if args.mode == "render":
        render(args.glb.resolve(), output)
    else:
        verify(output)


if __name__ == "__main__":
    main()
