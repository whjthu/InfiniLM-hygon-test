"""Frozen recording split; no Torch, network, model or answer access."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from view_schedule import VIEW_BATCHES

LABEL = 'training-vggt-tum-near7-bidirectional14-native25-v1'
REQUEST = '3bd3a976cc5348148e982f7e3d8fe0b6'
PARENT, NODE = '1103606', 'f13r4n09'
TRAIN = 'freiburg3_structure_texture_near'
HOLDOUT = ('freiburg3_nostructure_texture_far', 'freiburg3_sitting_xyz')
BUNDLE_SHA = '647b78e286687cc1d55670f64472fbe0f395c1b81cbeb928999fcf415f6a59f7'
OLD_SHA = 'fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c'
EXTRA_SHA = '9a39e38c2acf5da7aacfc08729c60307cc048ea2293be81c83b40a69f6695cfa'
BACKEND_SHA = '762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998'
PLANNED_UPDATES, PLANNED_PREDICTIONS = 25, 4


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def read(path, expected, cap=1<<20):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'missing/aliased pinned member')
    with path.open('rb') as stream:
        raw = stream.read(cap+1)
    need(0 < len(raw) <= cap and sha(raw) == expected, 'bounded member SHA differs')
    return raw


def child(root, name):
    relative = PurePosixPath(name)
    need(relative.as_posix() == name and not relative.is_absolute() and
         '..' not in relative.parts and '\\' not in name, 'unsafe relative member')
    return Path(root).joinpath(*relative.parts)


def check_clock(epoch, deadline, now):
    need(all(type(x) in (int, float) and math.isfinite(x) for x in (epoch, deadline, now)) and
         0 < epoch <= now < deadline-30 and 0 < deadline-epoch <= 1800,
         'invalid/exhausted original epoch')


def partition(descriptors):
    need(len(descriptors) == 9, 'nine source pair denominator differs')
    known = set()
    train, held = [], []
    for item in descriptors:
        pins = item['pair']['files_SHA256']
        media = {pin for name, pin in pins.items() if name.endswith('.png')}
        need(len(pins) == 7 and len(media) == 4 and not known.intersection(media),
             'duplicate/invalid raw media across recording split')
        known.update(media)
        sequence = item['pair']['sequence']
        if sequence == TRAIN:
            train.append(item)
        else:
            need(sequence in HOLDOUT, 'unregistered recording')
            held.append(item)
    need(len(train) == 7 and len(held) == 2 and
         {x['pair']['sequence'] for x in held} == set(HOLDOUT), 'recording quota differs')
    return train, held


def descriptors(root):
    """Only bundle/pair declarations here; no pose, depth or image bytes."""
    root = Path(root)
    read(root/'bundle.json', BUNDLE_SHA, 2048)
    original = json.loads(read(root/'original/bundle.json', OLD_SHA, 8192))
    from gencuda.data_contract import PAIRS
    need(len(original['pairs']) == len(PAIRS) == 8, 'original descriptor denominator differs')
    result = []
    for (key, sequence, pin), row in zip(PAIRS, original['pairs']):
        need(row == dict(root=key, manifest=key+'/pair.json', manifest_SHA256=pin, sequence=sequence),
             'original frozen roster differs')
        pair = json.loads(read(child(root/'original', row['manifest']), pin, 4096))
        need(pair['sequence'] == sequence, 'pair sequence differs')
        result.append(dict(root=str(root/'original'/key), pin=pin, pair=pair))
    pair = json.loads(read(root/'sitting_xyz/pair-manifest.json', EXTRA_SHA, 4096))
    need(pair['sequence'] == HOLDOUT[1], 'addition sequence differs')
    result.append(dict(root=str(root/'sitting_xyz'), pin=EXTRA_SHA, pair=pair))
    train, held = partition(result)
    identity = dict(train=[x['pin'] for x in train], holdout=[x['pin'] for x in held],
                    grouping='whole-recording', source=BUNDLE_SHA)
    return train, held, dict(identity, selection_SHA256=sha(encoded(identity)),
        independent_train_pairs=7, independent_train_recordings=1,
        heldout_pairs=2, heldout_recordings=2, global_fresh_blind=False,
        prior_927_used_heldout_recordings=True)


def verify_steps(rows):
    need([row['global_optimizer_step'] for row in rows] == list(range(1, 26)),
         'all 25 actual updates required')
    need(all(row['batch_cursor'] == (row['global_optimizer_step']-1) % VIEW_BATCHES for row in rows),
         'fourteen-orientation training cursor differs')
    for row in rows:
        need(all(type(row[key]) in (int, float) and math.isfinite(row[key])
                 for key in ('loss', 'grad_norm', 'camera_head_grad_norm_before_clip')),
             'nonfinite native step')
    need(any(row['camera_head_grad_norm_before_clip'] > 0 for row in rows),
         'no actual nonzero gradient')
