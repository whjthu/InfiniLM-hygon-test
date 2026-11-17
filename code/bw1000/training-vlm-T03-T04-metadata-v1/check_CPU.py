"""Small actual pure-stdlib boundary checks; no endpoint or prior-test execution."""
import ast
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import zlib

D = Path(__file__).resolve().parent
tree = ast.parse((D/'common.py').read_bytes())
ns = dict(gzip=gzip, hashlib=hashlib, io=None, json=json, os=os, Path=Path,
          PurePosixPath=__import__('pathlib').PurePosixPath, re=re, stat=__import__('stat'),
          struct=struct, zlib=zlib)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]),
             'actual-common-functions', 'exec'), ns)
positive, rejected = [], []


def reject(name, fn):
    try:
        fn()
    except (ValueError, FileExistsError):
        rejected.append(name)
    else:
        raise AssertionError(name+' was accepted')


plan = dict(max_file_bytes=4096, max_index_bytes=4096, max_decompressed_bytes=8192,
            max_compressed_bytes=8192)
context = dict(epoch=1, deadline=301)
files = {'internvl/model/config.json': b'{"example":1}\n'}
blob = ns['pack'](files, plan, context)
assert ns['unpack'](blob, plan, context) == files
positive.append('actual_pack_unpack_roundtrip')
reject('truncated_gzip', lambda: ns['unpack'](blob[:-1], plan, context))
reject('trailing_gzip_stream', lambda: ns['unpack'](blob+gzip.compress(b'foreign'), plan, context))
reject('path_traversal', lambda: ns['pack']({'../foreign': b'x'}, plan, context))
reject('backslash_member', lambda: ns['pack']({'internvl\\model': b'x'}, plan, context))
reject('wrong_context', lambda: ns['unpack'](blob, plan, dict(epoch=2, deadline=302)))
small = dict(plan, max_decompressed_bytes=20)
reject('decompression_limit', lambda: ns['unpack'](blob, small, context))
raw = bytearray(gzip.decompress(blob)); raw[-2] ^= 1
reject('payload_sha_differs', lambda: ns['unpack'](gzip.compress(raw), plan, context))
row = dict(name='same', bytes=1, SHA256=hashlib.sha256(b'x').hexdigest())
index = ns['canonical'](dict(context=context, files=[row, row]))
duplicate = gzip.compress(struct.pack('<Q', len(index))+index+b'xx')
reject('duplicate_member', lambda: ns['unpack'](duplicate, plan, context))
with tempfile.TemporaryDirectory() as temp:
    existing = Path(temp)/'owned'
    existing.mkdir()
    reject('existing_receiver_namespace', lambda: ns['own'](existing, context))
native = dict(rank=0, world_size=8, global_step=25, step_records=[dict(step=i) for i in range(1, 26)],
              initial_trainable_sha256='1'*64)
receipt = dict(rank=0, world_size=8, completed_global_step=2, optimizer_updates=2, micro_forwards=8,
               clip_calls=2, files=[dict(name='initial.pt', bytes=123, SHA256='2'*64)] +
               [dict(name='other-%02d'%i) for i in range(24)])
args = [dict(run_root='/example/original'), 0, native, receipt, dict(bytes=10, SHA256='3'*64),
        dict(bytes=10, SHA256='4'*64)]
value = ns['initial_row'](*args)
assert value['initial_tensor_read'] is False and value['initial_file_bytes'] == 123
positive.append('actual_initial_row_without_tensor_read')
bad = copy.deepcopy(args); bad[2]['initial_trainable_sha256'] = '[string omitted]'
reject('omitted_initial_digest', lambda: ns['initial_row'](*bad))
bad = copy.deepcopy(args); bad[3]['files'][1]['name'] = 'initial.pt'
reject('duplicate_initial_member', lambda: ns['initial_row'](*bad))
bad = copy.deepcopy(args); bad[3]['optimizer_updates'] = 1
reject('wrong_capture_update_boundary', lambda: ns['initial_row'](*bad))
bad = copy.deepcopy(args); bad[2]['rank'] = 7
reject('foreign_rank', lambda: ns['initial_row'](*bad))
print(json.dumps(dict(passed=True, positive=positive, rejected=rejected,
                     pure_CPU_only=True, remote_calls=0, actual_metadata_read=False)))
