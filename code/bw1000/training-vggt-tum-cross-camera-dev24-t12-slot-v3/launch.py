"""Bind the frozen VGGT development scorer to an unoccupied T12 allocation."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
LABEL = PACKAGE.name
PARENT, NODE = '1103606', 'f13r4n09'
REQUEST = 'b51243f49f82496cb6ec80029de1f84d'
ORIGINAL_INDEX = 'coordination/training/vggt-tum-cross-camera-controller-files-v1.json'
ORIGINAL_INDEX_SHA = 'f224f502dc9b5755182f458591e47cf5214f7484dc0c0480d4028d5dade0a910'
CONTROLLER = 'tools/control_vggt_tum_cross_camera_dev.py'
STAGE_SHA = 'd221478fe51e14ccdf8ca858632c60aee67993e693c58f29b2c9357d373dc903'
INTENT = f'local-only/training80-user-all-nodes-live-first/{LABEL}/intent.json'


def read_pinned(publication, relative, wanted=None):
    raw = (REPO / relative).read_bytes()
    if wanted is not None and sha256(raw).hexdigest() != wanted:
        raise ValueError(f'original source changed: {relative}')
    command = ['git', '--no-optional-locks', '-c', 'safe.directory=' + REPO.as_posix(),
               '-C', str(REPO), 'cat-file', 'blob', publication + ':' + relative]
    if subprocess.check_output(command, timeout=10) != raw:
        raise ValueError(f'publication source changed: {relative}')
    return raw


def controller_sources(publication):
    original = json.loads(read_pinned(publication, ORIGINAL_INDEX, ORIGINAL_INDEX_SHA))
    sources = {name: read_pinned(publication, name, pin) for name, pin in original.items()}
    source = sources[CONTROLLER]
    for old, new in (
        (b"PARENT, NODE = '1087656', 'f11r4n19'", f"PARENT, NODE = '{PARENT}', '{NODE}'".encode()),
        (b"LABEL = 'training-vggt-tum-cross-camera-dev24-v1'", f"LABEL = '{LABEL}'".encode()),
    ):
        if source.count(old) != 1:
            raise ValueError('original controller binding changed')
        source = source.replace(old, new)
    compile(source, CONTROLLER, 'exec')
    sources[CONTROLLER] = source

    scorer_path = 'tools/score_vggt_tum_cross_camera_dev.py'
    scorer = sources[scorer_path]
    substitutions = (
        (b'import numpy as np', b'import importlib.metadata\nimport numpy as np'),
        (b'torch.__version__ != TORCH_VERSION', b'importlib.metadata.version("torch") != TORCH_VERSION\n            or str(torch.version.__version__) != "2.10.0"\n            or str(torch.__version__) not in ("2.10.0", TORCH_VERSION)'),
    )
    for old, new in substitutions:
        if scorer.count(old) != 1:
            raise ValueError('original scoreboard provider check changed')
        scorer = scorer.replace(old, new)
    compile(scorer, scorer_path, 'exec')
    sources[scorer_path] = scorer
    stage_path = 'tools/stage_vggt_tum_cross_camera_dev.py'
    original_staged = b'training-vggt-tum-cross-camera-dev24-v1-staged'
    if sources[stage_path].count(original_staged) != 1:
        raise ValueError('original scoring stage identity changed')
    sources[stage_path] = sources[stage_path].replace(
        original_staged, b'training-vggt-tum-cross-camera-dev24-v3-runtime-corrected')
    index = {name: sha256(data).hexdigest() for name, data in sources.items()}
    sources[ORIGINAL_INDEX] = (json.dumps(index, sort_keys=True, allow_nan=False) + '\n').encode()
    return sources


def main(publication, transport):
    """The coordinator reserves a fresh card and injects its authenticated transport."""
    if not callable(transport):
        raise ValueError('existing authenticated transport is required')
    sources = controller_sources(publication)
    rel = PACKAGE.relative_to(REPO).as_posix()
    receiver = read_pinned(publication, rel + '/receive.py')
    raw_claim = read_pinned(publication, rel + '/CLAIM.json')
    claim = json.loads(raw_claim)
    if (claim['request'], claim['parent'], claim['node'], claim['stage_SHA256']) != (REQUEST, PARENT, NODE, STAGE_SHA):
        raise ValueError('frozen T12 claim differs')
    if sha256(sources[ORIGINAL_INDEX]).hexdigest() != claim['remapped_source_index_SHA256']:
        raise ValueError('remapped controller source differs from publication')
    context = dict(request=REQUEST, publication=publication, label=LABEL, stage_SHA256=STAGE_SHA,
                   source_index_SHA256=sha256(sources[ORIGINAL_INDEX]).hexdigest(),
                   claim_SHA256=sha256(raw_claim).hexdigest())
    payload = ('INPUT=' + repr(dict(context=context, sources=sources)) + '\n').encode() + receiver
    path = REPO / INTENT
    path.parent.mkdir(mode=0o700, parents=True)
    intent = dict(context, body_SHA256=sha256(payload).hexdigest(), epoch=time.time(),
                  parent=PARENT, node=NODE, replay_allowed=False)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(intent, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return transport(payload, intent)
