"""Render the opt-in SF3D atlas route as an overlay on the frozen common framework."""

from hashlib import sha256
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FRAMEWORK = "a68952cb2285d6593b665147249d14ec1f8cecbc"
FROZEN = REPO / "code/bw1000/training-qwen-target-unified-framework-a68952c-v1"
ARCHIVE = REPO / "code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz"
CLI = "generation/src/gencuda/cli.py"
NATIVE = "generation/src/gencuda/backends/sf3d_training.py"
DERIVATIVE = "generation/src/gencuda/backends/sf3d_atlas_training.py"
INPUT = "generation/src/gencuda/backends/atlas_inputs.py"


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def checked(raw, sha):
    if pin(raw)["SHA256"] != sha:
        raise ValueError("unexpected original framework bytes")
    return raw


def original_sources():
    manifest = json.loads(checked((FROZEN / "FRAMEWORK_FILES.json").read_bytes(),
                                  "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"))
    if manifest["framework_commit"] != FRAMEWORK:
        raise ValueError("unexpected original framework")
    cli = checked((FROZEN / "deltas" / CLI).read_bytes(), manifest["files"][CLI]["SHA256"])
    checked(ARCHIVE.read_bytes(), "85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f")
    with tarfile.open(ARCHIVE, mode="r:gz") as tar:
        backend = checked(tar.extractfile("framework/" + NATIVE).read(), manifest["files"][NATIVE]["SHA256"])
    return cli.decode("utf-8"), backend.decode("utf-8")


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("native integration site is not unique")
    return source.replace(old, new, 1)


def render():
    cli, native = original_sources()
    route = '    module, function, source = routes[spec.name]\n'
    cli = once(cli, route, route + ('    if command == "train" and spec.name == "stable-fast-3d" and "sf3d_atlas" in config:\n'
                                   '        module = "sf3d_atlas_training"\n'))
    with_atlas = ('    from io import BytesIO\n'
                  '    from .atlas_inputs import atlas_config_sha256, load_atlas\n\n'
                  '    atlas = load_atlas(config)\n'
                  '    os.environ.setdefault("SF3D_OFFLINE_INIT", "1")\n')
    native = once(native, '    os.environ.setdefault("SF3D_OFFLINE_INIT", "1")\n', with_atlas)
    native = once(native, '        config_sha256=_sf3d_training_config_sha256(config),\n'
                          '        input_sha256=_sf3d_paths_sha256(input_paths, include_paths=False),\n',
                          '        config_sha256=atlas_config_sha256(_sf3d_training_config_sha256(config), atlas),\n'
                          '        input_sha256=atlas["input_sha256"],\n')
    native = once(native, '        images = [Image.open(path).convert("RGBA") for path in input_paths]\n'
                          '        targets = [foreground_material_targets(np.asarray(image)) for image in images]\n',
                          '        images = [Image.open(BytesIO(raw)).convert("RGBA") for raw in atlas["images"]]\n'
                          '        targets = atlas["targets"]\n')
    original = '        "input_count": len(input_paths),\n'
    native = once(native, original, original + ('        "domain_training_manifest_sha256": atlas["manifest_sha256"],\n'
                                                '        "domain_training_labels_sha256": atlas["labels_sha256"],\n'))
    return {CLI: cli.encode("utf-8"), DERIVATIVE: native.encode("utf-8"),
            INPUT: (HERE / "atlas_inputs.py").read_bytes()}


def write_overlay(output):
    outputs = render()
    output.mkdir(mode=0o700)
    for name, raw in outputs.items():
        path = output / name
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("xb") as f:
            f.write(raw)
    return dict(base_framework=FRAMEWORK, overlay_files={name: pin(raw) for name, raw in outputs.items()})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(write_overlay(parser.parse_args().output)))
