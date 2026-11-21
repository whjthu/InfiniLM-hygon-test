"""Complete the existing surface-PBR cohort with the publisher's exact Corset assets."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import time
import urllib.request


REPO = Path(__file__).resolve().parents[3]
LOCAL = REPO / 'local-only'
OUTPUT = LOCAL / 'training-sf3d-public-surface-cohort-v1-original'
PREVIOUS = LOCAL / 'training-sf3d-public-surface-material-targets-v1-original'
PREVIOUS_RESULT = 'cd621ac0e0701c7391f57d126c2735154616eca7b8fd39ab75ec74b70ba2dca8'
BASE = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Corset/glTF/'
LICENSE = 'https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Corset/LICENSE.md'
ASSETS = {
    'Corset.gltf': (2487, '1aaa7fa54731ca833c3a0f919ba82c1af2a69cc7f8de36ba76bfb5038de8bd38'),
    'Corset.bin': (662184, 'de7730fb4201466603410dc1a7128bc1339e945a945dbb852f6dc39f3c53ec22'),
    'Corset_baseColor.png': (4501246, '76a4c45762a26e3bc8a62ea72cf791bd376d310b7bb0856daecac2278c00e626'),
    'Corset_occlusionRoughnessMetallic.png': (3443675, 'c3b41efc3a39fd2b3b9d6f0380057a826192632c4a099f4e3fa0452b8dfce0dc'),
}
SURFACE = REPO / 'code/bw1000/training-sf3d-public-surface-material-targets-v1'
MESH = REPO / 'code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py'
PBR = REPO / 'code/bw1000/training-sf3d-public-pbr-atlas2-v1/pbr.py'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(raw):
    return {'bytes': len(raw), 'SHA256': sha256(raw).hexdigest()}


def exact(path, pair):
    size, digest = pair
    need(path.resolve(strict=True) == path and path.is_file(), 'expected canonical regular file')
    with path.open('rb') as stream:
        data = stream.read(size + 1)
    need(pin(data) == {'bytes': size, 'SHA256': digest}, 'frozen input differs: ' + path.name)
    return data


def save(name, data):
    with (OUTPUT / name).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def receive(opener, name, expected, deadline):
    size, checksum = expected
    url = BASE + name
    left = deadline - time.monotonic()
    need(left > 1, 'publisher receive budget expired')
    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'SF3D-cohort-audit'})
    with opener.open(request, timeout=min(left, 25)) as response:
        need(response.geturl() == url and response.status == 200, 'publisher response identity differs')
        lengths = response.headers.get_all('Content-Length')
        need((lengths is None or lengths == [str(size)]) and
             response.headers.get('Content-Encoding', 'identity') == 'identity', 'publisher payload envelope differs')
        data = bytearray()
        while len(data) <= size:
            need(time.monotonic() < deadline, 'publisher receive deadline expired')
            chunk = response.read(min(256 * 1024, size + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        need(time.monotonic() < deadline, 'publisher receive deadline expired')
    need(pin(data) == {'bytes': size, 'SHA256': checksum}, 'publisher digest or byte count differs')
    with (OUTPUT / name).open('xb') as stream:
        stream.write(data)
    return dict(name=name, url=url, status=200, **pin(data))


def acquire():
    OUTPUT.mkdir(mode=0o700)
    deadline = time.monotonic() + 120
    opener = urllib.request.build_opener(NoRedirect)
    report = dict(publisher='KhronosGroup/glTF-Sample-Assets', license='CC0-1.0', license_source=LICENSE,
                  requested_files=list(ASSETS), files=[], passed=False, original_local_path_read=False)
    try:
        for name, expected in ASSETS.items():
            report['files'].append(receive(opener, name, expected, deadline))
        report['passed'] = True
    except (OSError, ValueError) as error:
        report['error_type'] = type(error).__name__
        report['error'] = str(error)[:240]
    finally:
        save('publisher-receipt.json', report)
    return report


def derive():
    import importlib.util
    import numpy as np
    from PIL import Image

    ast_source = exact(SURFACE / 'surface_targets.py', (5992,
                           '35f8e149aebaae562b8e5d42737c8a886a16e25fc3dd46ef73534a464bd8c857'))
    spec = importlib.util.spec_from_file_location('surface_pbr', SURFACE / 'surface_targets.py')
    surface = importlib.util.module_from_spec(spec)
    exec(compile(ast_source, str(SURFACE / 'surface_targets.py'), 'exec'), surface.__dict__)
    mesh = surface.source_module('gltf_geometry', MESH,
                                   {'bytes': 5502, 'SHA256': '8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a'})
    pbr = surface.source_module('pbr_pixel', PBR,
                                 {'bytes': 3808, 'SHA256': '3bc2279874fefbe2ebdb36ffa74824ee83661c63ea39359099dc266f7437d1e9'})
    receipt_raw = (OUTPUT / 'publisher-receipt.json').read_bytes()
    receipt = json.loads(receipt_raw)
    need(receipt['passed'] is True and [row['name'] for row in receipt['files']] == list(ASSETS)
         and receipt['license'] == 'CC0-1.0', 'exact publisher receipt required')
    assets = {name: exact(OUTPUT / name, expected) for name, expected in ASSETS.items()}
    for row in receipt['files']:
        name = row['name']
        need(row['url'] == BASE + name and row['status'] == 200
             and pin(assets[name]) == {field: row[field] for field in ('bytes', 'SHA256')},
             'publisher receipt and source bytes differ')
    graph = json.loads(assets['Corset.gltf'])
    blob = assets['Corset.bin']
    need([row['uri'] for row in graph['buffers']] == ['Corset.bin'], 'frozen mesh buffer URI differs')
    positions, faces, _ = mesh.read_geometry(graph, blob)
    need(faces.dtype.kind == 'u' and len(positions) == 11505 and len(faces) == 18324,
         'frozen Corset geometry count or indices differ')
    primitive, = graph['meshes'][0]['primitives']
    material = graph['materials'][primitive['material']]['pbrMetallicRoughness']
    images = []
    for key, name, mode in (('baseColorTexture', 'Corset_baseColor.png', 'RGBA'),
                            ('metallicRoughnessTexture', 'Corset_occlusionRoughnessMetallic.png', 'RGB')):
        texture = graph['textures'][material[key]['index']]
        need(graph['images'][texture['source']]['uri'] == name, 'frozen material image URI differs')
        with Image.open(io.BytesIO(assets[name])) as image:
            need(image.format == 'PNG' and image.mode in ('RGB', 'RGBA') and max(image.size) <= 4096,
                 'frozen PNG envelope differs')
            images.append(np.array(image.convert(mode)))
    fields, geometry = surface.surface_samples(graph, blob, mesh.accessor, *images, pbr.decode_pixels, 'Corset')
    corset = OUTPUT / 'Corset-surface-material-targets-private.npz'
    with corset.open('xb') as stream:
        np.savez_compressed(stream, **fields)
    new_pin = pin(corset.read_bytes())
    prior = json.loads(exact(PREVIOUS / 'result.json', (6965, PREVIOUS_RESULT)))
    need(prior['recipe'] == surface.RECIPE and prior['completed_objects'] == 2
         and prior['objects']['Corset']['completed'] is False, 'previous partial result differs')
    previous = prior['objects']
    objects = [previous['WaterBottle'], dict(group='Corset', split='train', completed=True,
        mesh=pin(blob), gltf=pin(assets['Corset.gltf']),
        textures={'base': pin(assets['Corset_baseColor.png']), 'orm': pin(assets['Corset_occlusionRoughnessMetallic.png'])},
        target=new_pin, material_samples=surface.N, source_surface=geometry), previous['BoomBox']]
    expected_splits = [('WaterBottle', 'train'), ('Corset', 'train'), ('BoomBox', 'dev')]
    need([(row['group'], row['split']) for row in objects] == expected_splits, 'frozen train/dev cohort differs')
    sources = []
    for row in objects:
        path = corset if row['group'] == 'Corset' else PREVIOUS / (row['group']+'-surface-material-targets-private.npz')
        fields = surface.load_native_targets(path, row['target'], surface.RECIPE)
        target = surface.to_original_target_object(fields)
        need(target['albedo'].shape == (3,) and type(target['roughness']) is float
             and type(target['metallic']) is float, 'native material target ABI differs')
        sources.append(dict(group=row['group'], split=row['split'], path=path.relative_to(REPO).as_posix(),
                            pin=row['target'], material_samples=row['material_samples']))
    identity = surface.training_identity(objects)
    alternate_dev = dict(objects[-1], target={'bytes': 1, 'SHA256': '0'*64})
    need(surface.training_identity([*objects[:2], alternate_dev]) == identity, 'dev influences training identity')
    save('cohort-private.json', dict(recipe=surface.RECIPE, ordered_labels=sources, train_identity_SHA256=identity))
    result = dict(recipe=surface.RECIPE, completed_objects=3, train_objects=2, dev_objects=1, blind_objects=0,
                  completed_surface_slots=3*surface.N, newly_computed_slots=surface.N, reused_label_objects=2,
                  verified_native_target_objects=3, train_identity_SHA256=identity, dev_label_excluded=True,
                  corset_geometry=geometry, corset_target=new_pin, publisher_receipt=pin(receipt_raw),
                  supersedes_partial_result_SHA256=PREVIOUS_RESULT, local_cohort_complete=True,
                  native_caller_integrated=False, actual_optimizer_updates=0, actual_Slurm_steps=0,
                  formal17_GO=False, training_ready_added=0)
    save('result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('acquire', 'derive'))
    args = parser.parse_args()
    result = acquire() if args.mode == 'acquire' else derive()
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(args.mode == 'acquire' and not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
