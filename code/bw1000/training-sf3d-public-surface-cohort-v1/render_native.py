"""Add the completed surface-label cohort to the frozen shared SF3D entry."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / 'code/bw1000/training-sf3d-public-atlas-native-route-v1'
INPUT = 'generation/src/gencuda/backends/surface_inputs.py'
BACKEND = 'generation/src/gencuda/backends/sf3d_surface_training.py'
IDENTITY = '8567c9ce3dc0ee8382055c044aa2f837bc2f612000b77210acb8a8b32f902039'
RECIPE = 'sf3d-public-surface-area-nearest-PBR4096-v1'
TRAIN = (
    ('WaterBottle',
     'training-sf3d-public-pbr-atlas2-v1-original/WaterBottle/glTF/WaterBottle_baseColor.png',
     (2165850, '62d0ad9b4e2c75e9bc7d67644a50d697563000ffa586d2e185ea38f5000e0a63'),
     'training-sf3d-public-surface-material-targets-v1-original/WaterBottle-surface-material-targets-private.npz',
     (129348, '53fa04fac557161b4b269a72a39684145122054faeb47360069af0384e8f9952')),
    ('Corset',
     'training-sf3d-public-surface-cohort-v1-original/Corset_baseColor.png',
     (4501246, '76a4c45762a26e3bc8a62ea72cf791bd376d310b7bb0856daecac2278c00e626'),
     'training-sf3d-public-surface-cohort-v1-original/Corset-surface-material-targets-private.npz',
     (134135, 'ee3143988a6d48c319ae0bafdf616d3d0c15268ab33101687af1daae11ac69e6')),
)


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def exact(path, wanted):
    size, digest = wanted
    with path.open('rb') as stream:
        raw = stream.read(size + 1)
    if pin(raw) != dict(bytes=size, SHA256=digest):
        raise ValueError('bound input differs: ' + path.name)
    return raw


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('expected one frozen integration anchor')
    return source.replace(old, new, 1)


def render():
    renderer = BASE / 'render.py'
    raw = exact(renderer, (4304, 'f8141b4c68b5aa4837e0408a18eeb596179d7ebf7166f85a9744fb02e06dcf8d'))
    namespace = {'__name__': 'pinned_sf3d_atlas', '__file__': str(renderer)}
    exec(compile(raw, str(renderer), 'exec'), namespace)
    overlay = namespace['render']()
    cli, atlas_backend, atlas_input = (namespace[k] for k in ('CLI', 'DERIVATIVE', 'INPUT'))
    if pin(overlay[atlas_input])['SHA256'] != '52f996b408e5a5774efb78a197c9fb7429b08cc8a2c9f2c7233fb02b3110d14a':
        raise ValueError('original atlas loader differs')
    atlas_route = '        module = "sf3d_atlas_training"\n'
    surface_route = ('    if command == "train" and spec.name == "stable-fast-3d" and "sf3d_surface" in config:\n'
                     '        if "sf3d_atlas" in config:\n'
                     '            raise ValueError("select exactly one SF3D material recipe")\n'
                     '        module = "sf3d_surface_training"\n')
    overlay[cli] = once(overlay[cli].decode(), atlas_route, atlas_route + surface_route).encode()
    loader = overlay[atlas_input].decode().replace('load_atlas', 'load_surface').replace(
        'atlas_config_sha256', 'surface_config_sha256')
    loader = once(loader, 'sf3d-public-pbr-UV-atlas-v1', RECIPE)
    loader = once(loader, 'config.get("sf3d_atlas")', 'config.get("sf3d_surface")')
    loader = once(loader, '{"recipe", "train"}:', '{"recipe", "train", "surface_train_identity_SHA256"}:')
    loader = once(loader, '    sources, targets, expected_paths, identity = [], [], [], []',
                  f'    if plan["surface_train_identity_SHA256"] != "{IDENTITY}":\n'
                  '        raise ValueError("frozen surface cohort identity differs")\n'
                  '    sources, targets, expected_paths, identity = [], [], [], []')
    loader = once(loader, 'row["target_pin"], 16384)', 'row["target_pin"], 256 * 1024)')
    expected = {
        group: {'image_pin': dict(bytes=image_pin[0], SHA256=image_pin[1]),
                 'target_pin': dict(bytes=label_pin[0], SHA256=label_pin[1])}
        for group, _, image_pin, _, label_pin in TRAIN
    }
    loader = once(loader, '    for row in plan["train"]:\n',
                  f'    expected_pins = {expected!r}\n'
                  '    for row in plan["train"]:\n'
                  '        if any(row.get(key) != pin for key, pin in expected_pins[row["group"]].items()):\n'
                  '            raise ValueError("frozen surface image or label pin differs")\n')
    loader = once(loader, 'schema="SF3D_PUBLIC_ATLAS_TRAIN_LABEL_INPUT_ID_V1", recipe=RECIPE,',
                  'schema="SF3D_PUBLIC_SURFACE_TRAIN_LABEL_INPUT_ID_V1", recipe=RECIPE,\n'
                  '                           surface_train_identity_SHA256=plan["surface_train_identity_SHA256"],')
    loader = once(loader, 'SF3D_ATLAS_NATIVE_CONFIG_V1', 'SF3D_SURFACE_NATIVE_CONFIG_V1')
    overlay[INPUT] = loader.encode()
    backend = overlay[atlas_backend].decode().replace('.atlas_inputs', '.surface_inputs').replace(
        'load_atlas', 'load_surface').replace('atlas_config_sha256', 'surface_config_sha256')
    overlay[BACKEND] = backend.encode()
    return overlay


def prepare_train(output):
    output.mkdir()
    rows = []
    for group, image, image_pin, label, label_pin in TRAIN:
        row = dict(group=group)
        for role, name, wanted, suffix in (('image', image, image_pin, '.png'),
                                           ('target', label, label_pin, '.npz')):
            original = REPO / 'local-only' / name
            if original.resolve(strict=True) != original:
                raise ValueError('noncanonical private source')
            raw = exact(original, wanted)
            file = group + suffix
            with (output / file).open('xb') as stream:
                stream.write(raw)
            row.update({role: file, role+'_pin': pin(raw)})
        rows.append(row)
    payload = dict(recipe=RECIPE, train=rows, surface_train_identity_SHA256=IDENTITY)
    manifest = (json.dumps(payload, sort_keys=True) + '\n').encode()
    path = output / 'train-manifest.json'
    with path.open('xb') as stream:
        stream.write(manifest)
    return dict(sf3d_surface=dict(manifest=str(path), **pin(manifest)),
                input_images=[str(output / row['image']) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    sources = render()
    for name, raw in sources.items():
        path = args.output / 'overlay' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    config = prepare_train(args.output.resolve(strict=True) / 'train')
    (args.output / 'private-config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(source_overlay={name: pin(raw) for name, raw in sources.items()},
                          train_manifest={key: config['sf3d_surface'][key] for key in ('bytes', 'SHA256')},
                          train_objects=len(config['input_images']), actual_CUDA_calls=0)))


if __name__ == '__main__':
    main()
