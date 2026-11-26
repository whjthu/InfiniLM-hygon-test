"""Render one fixed ABO RGBA view in Blender 4.5 CPU and verify it locally."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys


REPO = Path(__file__).resolve().parents[1]
GLB = dict(bytes=2073360, SHA256="352d13e2a41bc68da722f745ef5f86713786b4fb28c4ad7d8a9acdd877bb7ed8")
CAMERA_DOC = REPO / "results/bw1000/2026-11-27/specialized-abo-fixed-camera-uv-alpha-diagnostic.json"
CAMERA_PIN = dict(bytes=4308, SHA256="1a7604a5c211231580757e190fff176ea9db2b278a2ae435c92e53b1c681efdc")
COLOR_PIN = "53d2c3f40ed127b2b1491206e260f9b72150a947644e47b937e9008dd478b7ed"
SIZE = SAMPLES = 64
PRIMARY = "fixed-camera.private.png"
CONTROL = "opaque-support-control.private.png"
LIMIT_PNG = 256 * 1024


def require(test, error):
    if not test:
        raise ValueError(error)


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def read(path, limit):
    require(path.is_file() and not path.is_symlink(), "REGULAR_FILE_REQUIRED")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit, "FILE_EXTENT")
    return raw


def write(path, result):
    body = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(body)
    return pin(body)


def source_to_blender(v):
    x, y, z = v
    return x, -z, y


def declared_material(raw):
    length, kind = struct.unpack_from("<I4s", raw, 12)
    require(struct.unpack_from("<4sII", raw) == (b"glTF", 2, len(raw)) and kind == b"JSON",
            "EXACT_GLB_HEADER")
    doc = json.loads(raw[20:20+length])
    mesh, = doc["meshes"]
    primitive, = mesh["primitives"]
    material = doc["materials"][primitive["material"]]
    params = material["pbrMetallicRoughness"]
    require(material["alphaMode"] == "BLEND" and material["doubleSided"] is False
            and "alphaCutoff" not in material and params["baseColorFactor"] == [1, 1, 1, 1],
            "PUBLISHER_BLEND_ADMISSION")
    color = doc["images"][doc["textures"][params["baseColorTexture"]["index"]]["source"]]
    view = doc["bufferViews"][color["bufferView"]]
    start = 28 + length + view["byteOffset"]
    require(pin(raw[start:start+view["byteLength"]])["SHA256"] == COLOR_PIN, "PUBLISHER_BASECOLOR_PIN")
    return str(color["name"])


def source_camera():
    raw = read(CAMERA_DOC, CAMERA_PIN["bytes"])
    require(pin(raw) == CAMERA_PIN, "SOURCE_CAMERA_PIN")
    doc = json.loads(raw)
    require(doc["glb"] == GLB, "SOURCE_CAMERA_WRONG_ASSET")
    return doc["camera"], doc["source_frame"]


def output_path(path):
    resolved = path.resolve()
    inside = (REPO / "local-only").resolve()
    require(resolved != inside and resolved.is_relative_to(inside), "PRIVATE_LOCAL_ONLY_OUTPUT_REQUIRED")
    return resolved


def material_shader(material):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    surface, = [n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    shader, = [n for n in nodes if n.type == "BSDF_PRINCIPLED"]
    color = shader.inputs["Base Color"].links
    alpha = shader.inputs["Alpha"].links
    old = surface.inputs["Surface"].links
    require(len(old) == len(color) == len(alpha) == 1 and old[0].from_node == shader
            and color[0].from_node == alpha[0].from_node and alpha[0].from_socket.name == "Alpha"
            and alpha[0].from_node.type == "TEX_IMAGE", "IMPORTED_NATIVE_COLOR_ALPHA_GRAPH")
    image = alpha[0].from_node.image
    require(image is not None and tuple(image.size) == (4096, 4096), "IMPORTED_NATIVE_BASECOLOR_IMAGE")
    packed = image.packed_file
    packed_sha = sha256(packed.data).hexdigest() if packed else None
    require(packed_sha is None or packed_sha == COLOR_PIN, "IMPORTED_PACKED_BASECOLOR_DIFFERS")
    # glTF single-sided visibility is made explicit on both the source graph and AA control.
    links.remove(old[0])
    geometry = nodes.new("ShaderNodeNewGeometry")
    mix = nodes.new("ShaderNodeMixShader")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    links.new(geometry.outputs["Backfacing"], mix.inputs[0])
    links.new(shader.outputs[0], mix.inputs[1])
    links.new(transparent.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], surface.inputs["Surface"])
    return shader, packed_sha


def render(glb, output):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector

    require(bpy.app.background and bpy.app.version[:2] == (4, 5), "BLENDER45_HEADLESS_REQUIRED")
    raw = read(glb, GLB["bytes"])
    require(pin(raw) == GLB, "EXACT_ORIGINAL_ABO_GLB")
    color_name = declared_material(raw)
    fixed, source = source_camera()
    output.mkdir(mode=0o700)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    require(bpy.ops.import_scene.gltf(filepath=str(glb.resolve())) == {"FINISHED"}, "NATIVE_GLTF_IMPORT_FAILED")
    require(pin(read(glb, GLB["bytes"])) == GLB, "GLB_CHANGED_DURING_IMPORT")
    mesh, = [item for item in scene.objects if item.type == "MESH"]
    bpy.context.view_layer.update()
    imported = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    actual_box = [[fn(v[i] for v in imported) for i in range(3)] for fn in (min, max)]
    lo = source["source_world_bbox_min_m"]
    hi = source["source_world_bbox_max_m"]
    box = [[lo[0], -hi[2], lo[1]], [hi[0], -lo[2], hi[1]]]
    require(len(mesh.data.polygons) == 23592 and all(len(p.vertices) == 3 for p in mesh.data.polygons)
            and max(abs(a-b) for left, right in zip(actual_box, box) for a,b in zip(left,right)) < 2e-6,
            "IMPORTED_GLTF_YUP_TO_ZUP_BOUNDS")
    material, = list(mesh.data.materials)
    require(material.use_nodes, "IMPORTED_GLTF_MATERIAL_GRAPH")
    shader, packed_sha = material_shader(material)

    # Importer converts geometry to Blender axes. Map the frozen camera explicitly into the same frame.
    center = Vector(source_to_blender(source["reference_center_m"]))
    extent = source["reference_longest_extent_m"]
    eye = center + extent * Vector(source_to_blender(fixed["eye"]))
    target = center + extent * Vector(source_to_blender(fixed["look_at"]))
    right = Vector(source_to_blender(fixed["right"]))
    up = Vector(source_to_blender(fixed["sensor_up"]))
    backward = (eye-target).normalized()
    rotation = Matrix([right, up, backward]).transposed()
    require(abs(rotation.determinant()-1) < 1e-6, "PROPER_CAMERA_AXES")
    camera = bpy.data.objects.new("fixed ABO orthographic camera", bpy.data.cameras.new("fixed ABO orthographic lens"))
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix.Translation(eye) @ rotation.to_4x4()
    camera.data.type = "ORTHO"
    width = extent * fixed["orthographic_width"]
    camera.data.ortho_scale = width
    camera.data.clip_start = extent * .01
    camera.data.clip_end = extent * 10
    scene.camera = camera

    world = bpy.data.worlds.new("neutral studio world")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs[0].default_value = (.15, .15, .15, 1)
    world.node_tree.nodes.get("Background").inputs[1].default_value = .35
    scene.world = world
    lights = []
    for label, offset, energy, size in (("key", (1., -2., 2.), 25., 2.),
                                         ("fill", (-1.75, .4, .9), 12., 2.5)):
        source_light = bpy.data.lights.new(label, "AREA")
        lamp = bpy.data.objects.new(label, source_light)
        scene.collection.objects.link(lamp)
        lamp.location = center + extent * Vector(offset)
        lamp.rotation_euler = (center-lamp.location).to_track_quat("-Z", "Y").to_euler()
        source_light.energy = energy
        source_light.shape = "DISK"
        source_light.size = extent * size
        lights.append(dict(label=label, location=list(lamp.location), watts=energy, diameter=source_light.size))

    r = scene.render
    r.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    r.threads_mode, r.threads = "FIXED", 4
    r.resolution_x = r.resolution_y = SIZE
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    bpy.context.view_layer.update()
    projection = [list(world_to_camera_view(scene, camera, p)[:2])
                  for p in (target, target+right*width/2, target+up*width/2)]
    require(max(abs(a-b) for p, q in zip(projection, ((.5,.5),(1,.5),(.5,1))) for a,b in zip(p,q)) < 2e-6,
            "INDEPENDENT_BLENDER_CAMERA_PROJECTION")
    artifacts = {}
    for file in (PRIMARY, CONTROL):
        if file == CONTROL:
            alpha = shader.inputs["Alpha"]
            shader.id_data.links.remove(alpha.links[0])
            alpha.default_value = 1
        r.filepath = str(output / file)
        require(bpy.ops.render.render(write_still=True) == {"FINISHED"}, "CYCLES_CPU_RENDER_FAILED")
        artifacts[file] = pin(read(output / file, LIMIT_PNG))
    receipt = dict(schema="ABO_FIXED_CAMERA_CPU_RGBA_RENDER_V1", assets=artifacts,
        camera_source=CAMERA_PIN, glb=GLB, publisher_alpha_mode="BLEND", alpha_cutoff=None,
        original_base_color_alpha_link_verified=True, packed_base_color_SHA256=packed_sha, source_image_name=color_name,
        camera=dict(blender_zup_from_gltf_yup=[[1,0,0],[0,0,-1],[0,1,0]], world_to_camera=[list(row) for row in camera.matrix_world.inverted()],
            camera_to_world=[list(row) for row in camera.matrix_world], eye_world=list(eye), target_world=list(target),
            orthographic_width_m=width, resolution=[SIZE,SIZE], pixel_center_convention="0.5-based top-left",
            px_per_meter=SIZE/width, principal_center_px=[SIZE/2,SIZE/2], projection_landmarks=projection,
            clip_m=[camera.data.clip_start,camera.data.clip_end]),
        actual_qualified_renderer=1, actual_control_renders=1, actual_GPU=0, engine=r.engine,
        render_device=scene.cycles.device, blender_version=bpy.app.version_string, threads=r.threads, samples=SAMPLES,
        transparent_film=True, control_overrides_only_native_input_alpha=True, lights=lights)
    result = write(output / "render.json", receipt)
    print(json.dumps(dict(render_receipt=result, planned_primary=1, actual_primary_written=1)))


def alpha_metrics(alpha, opaque, size):
    require(len(alpha) == len(opaque) == size ** 2, "ALPHA_EXTENT")
    partial = {i for i,a in enumerate(alpha) if 0 < a < 255}
    # Require a 3x3 all-opaque neighborhood in the control. Border pixels and narrow
    # geometry cannot prove material opacity on their own.
    interior = {y*size+x for y in range(1,size-1) for x in range(1,size-1)
                if all(opaque[(y+dy)*size+x+dx] == 255 for dx in (-1,0,1) for dy in (-1,0,1))}
    attributable = {i for i in partial & interior if opaque[i] - alpha[i] >= 2}
    return dict(total=len(alpha), transparent=alpha.count(0), opaque=alpha.count(255), partial=len(partial),
                control_alpha_partial=sum(0 < a < 255 for a in opaque), opaque_control_interior=len(interior),
                partial_in_eroded_opaque_control=len(partial & interior), alpha_deficit_at_least_two=len(attributable),
                same_partial_on_control_edge=sum(i not in interior and alpha[i] == opaque[i] for i in partial),
                lower_half_attributed_partial=sum(i//size >= size//2 for i in attributable),
                material_partial_evidence="WITNESSED_IN_CONTROL_INTERIOR" if attributable else "UNKNOWN_EDGE_AA_NOT_EXCLUDED")


def verify_camera(actual):
    fixed, frame = source_camera()
    center = source_to_blender(frame["reference_center_m"])
    extent = frame["reference_longest_extent_m"]
    width = extent * fixed["orthographic_width"]
    expected_eye = [c + extent*x for c, x in zip(center, source_to_blender(fixed["eye"]))]
    expected_right = source_to_blender(fixed["right"])
    expected_up = source_to_blender(fixed["sensor_up"])
    c2w, w2c = actual["camera_to_world"], actual["world_to_camera"]
    errors = [abs(a-b) for left,right in ((actual["eye_world"],expected_eye), (actual["target_world"],center),
                   ([r[3] for r in c2w[:3]],expected_eye), ([r[0] for r in c2w[:3]],expected_right),
                   ([r[1] for r in c2w[:3]],expected_up)) for a,b in zip(left,right,strict=True)]
    errors.extend(abs(sum(w2c[i][k]*c2w[k][j] for k in range(4))-(i==j)) for i in range(4) for j in range(4))
    errors.extend(abs(a-b) for actual_pair, expected_pair in zip(actual["projection_landmarks"],
                  ((.5,.5),(1,.5),(.5,1)), strict=True) for a,b in zip(actual_pair,expected_pair,strict=True))
    require(actual["blender_zup_from_gltf_yup"] == [[1,0,0],[0,0,-1],[0,1,0]]
            and actual["resolution"] == [SIZE,SIZE] and actual["principal_center_px"] == [SIZE/2,SIZE/2]
            and abs(actual["orthographic_width_m"]-width) < 2e-6
            and abs(actual["px_per_meter"]-SIZE/width) < 2e-6 and max(errors) < 2e-6,
            "RENDERED_CAMERA_CALIBRATION")


def verify(output):
    import io
    from PIL import Image

    raw = read(output / "render.json", 32 * 1024)
    data = json.loads(raw)
    require(data["schema"] == "ABO_FIXED_CAMERA_CPU_RGBA_RENDER_V1" and data["glb"] == GLB
            and data["camera_source"] == CAMERA_PIN and data["engine"] == "CYCLES"
            and data["render_device"] == "CPU" and data["samples"] == SAMPLES
            and data["actual_GPU"] == 0 and data["publisher_alpha_mode"] == "BLEND"
            and data["control_overrides_only_native_input_alpha"] is True
            and set(data["assets"]) == {PRIMARY,CONTROL}, "EXACT_RENDER_RECEIPT")
    verify_camera(data["camera"])
    arrays = []
    for name in (PRIMARY,CONTROL):
        body = read(output / name, LIMIT_PNG)
        require(pin(body) == data["assets"][name], "PNG_HASH")
        with Image.open(io.BytesIO(body), formats=("PNG",)) as image:
            require(image.size == (SIZE,SIZE) and image.mode == "RGBA", "RGBA_PNG_EXTENT_MODE")
            if name == PRIMARY:
                pixels = list(image.getdata())
                require(any(a and max(red,green,blue) for red,green,blue,a in pixels), "VISIBLE_COLOR_REQUIRED")
            arrays.append(image.getchannel("A").tobytes())
    counts = alpha_metrics(*arrays,SIZE)
    require(0 < counts["transparent"] < SIZE**2 and counts["opaque_control_interior"] > 0,
            "TRANSPARENT_FILM_AND_OBJECT_REQUIRED")
    result = dict(status="PASS_RGBA", render=pin(raw), assets=data["assets"], alpha=counts,
                  verified_calibrated_renders=1, technical_control_renders=1, actual_GPU=0,
                  formal_training_rows=0)
    print(json.dumps(result, sort_keys=True))


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("render")
    run.add_argument("--glb", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = output_path(args.output)
    if args.mode == "render":
        render(args.glb,output)
    else:
        verify(output)


if __name__ == "__main__":
    main()
