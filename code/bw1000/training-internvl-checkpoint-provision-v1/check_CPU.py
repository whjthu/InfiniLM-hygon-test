"""Small real stdlib protocol/CLI boundaries; no SSH, weights or target endpoint."""
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import subprocess
import sys
import time

D = Path(__file__).resolve().parent


def extracted(filename, names, namespace):
    tree = ast.parse((D/filename).read_bytes())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def main():
    passed, rejected = [], []
    context = dict(deadline=time.time()+30)
    ns = dict(PLAN=dict(chunk_bytes=1048576), CONTEXT=context, MONO_END=time.monotonic()+30,
              time=time, PurePosixPath=PurePosixPath, struct=struct, json=json,
              H=lambda raw:hashlib.sha256(raw).hexdigest())
    extracted('protocol.py', ['require', 'member_name', 'exact', 'header_parts', 'tick'], ns)

    def bad(name, function):
        try:
            function()
        except (ValueError, EOFError):
            rejected.append(name)
        else:
            raise AssertionError('bad boundary admitted: '+name)

    class OneByte(io.BytesIO):
        def read(self, count=-1):
            return super().read(min(1,count))

    assert ns['exact'](OneByte(b'abcdef'),6) == b'abcdef'
    passed.append('actual_exact_short_reads')
    bad('stream_truncation', lambda:ns['exact'](io.BytesIO(b'ab'),3))
    bad('chunk_over_1MiB', lambda:ns['exact'](io.BytesIO(),1048577))
    header = b'{"a":{},"__metadata__":{}}'
    pin = dict(header_bytes=len(header), header_SHA256=hashlib.sha256(header).hexdigest(), tensors=1)
    blob = struct.pack('<Q', len(header))+header
    assert ns['header_parts'](OneByte(blob),pin) == blob
    passed.append('actual_header_prefix_hash_count')
    bad('header_length',lambda:ns['header_parts'](io.BytesIO(struct.pack('<Q',0)+header),pin))
    bad('header_SHA',lambda:ns['header_parts'](io.BytesIO(blob[:-1]+b' '),pin))
    bad('header_truncation',lambda:ns['header_parts'](io.BytesIO(blob[:-1]),pin))
    bad('header_tensor_count',lambda:ns['header_parts'](io.BytesIO(blob),dict(pin,tensors=2)))
    assert ns['member_name']('model/config.json').as_posix() == 'model/config.json'
    passed.append('canonical_member')
    for value in ('../a','/root/a','a\\b','a/../b','a//b','.'):
        bad('unsafe_member_'+repr(value),lambda value=value:ns['member_name'](value))
    context['deadline'] = time.time()-1
    bad('expired_original_epoch',ns['tick'])
    rns = extracted('relay_once.py',['transport_state','envelope'],dict(struct=struct))
    assert rns['transport_state'](ready=True,forced=False,returncode=0,expired=False) == 'RETURNED'
    for row in (dict(ready=True,forced=True,returncode=-9,expired=False),
                dict(ready=True,forced=False,returncode=-9,expired=True),
                dict(ready=False,forced=False,returncode=1,expired=False),
                dict(ready=True,forced=False,returncode=255,expired=False)):
        assert rns['transport_state'](**row) == 'UNKNOWN'
    passed.append('actual_transport_forced_expired_unack_255_UNKNOWN')
    observation = extracted('reconcile_endpoint.py',['process_observation'],{})['process_observation']
    class ForbiddenProc:
        def exists(self):
            raise AssertionError('cross-host PID was queried')
    row = observation(dict(host='original-login01',boot_id='original-boot',pid=123,startticks=456),
                      'different-login02','different-boot',ForbiddenProc())
    assert row['same_host'] is False and row['original_process_live'] is None
    passed.append('cross_host_different_boot_process_UNKNOWN_no_PID_query')
    relaytree = ast.parse((D/'relay_once.py').read_bytes())
    bootnode = next(node for node in relaytree.body if isinstance(node,ast.Assign)
                    and any(isinstance(x,ast.Name) and x.id=='BOOTSTRAP' for x in node.targets))
    bns = {}
    exec(compile(ast.Module(body=[bootnode],type_ignores=[]),'relay_once.py','exec'), bns)
    script = b"import sys,hashlib,json\nx=sys.stdin.buffer.read()\nprint(json.dumps([len(x),hashlib.sha256(x).hexdigest()]))\n"
    suffix = b'binary\x00suffix\xff'
    result = subprocess.run([sys.executable,'-I','-S','-B','-c',bns['BOOTSTRAP']],
                            input=rns['envelope'](script)+suffix,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)
    assert result.returncode == 0 and json.loads(result.stdout) == [len(suffix),hashlib.sha256(suffix).hexdigest()]
    passed.append('actual_child_binary_bootstrap_preserves_suffix')
    for name,payload in (('short_length',b'a'),('short_script',struct.pack('<Q',4)+b'x'),
                         ('oversized_script',struct.pack('<Q',262145))):
        row = subprocess.run([sys.executable,'-I','-S','-B','-c',bns['BOOTSTRAP']],input=payload,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)
        assert row.returncode != 0
        rejected.append('actual_child_'+name)
    bad('empty_script',lambda:rns['envelope'](b''))
    bad('oversized_script_envelope',lambda:rns['envelope'](b'x'*262145))
    value = dict(passed=True, valid=passed, rejected=rejected, target_invocations=0, SSH=0, weights=0,
                 GPU=0, description='Extracted actual helper functions and real local child bootstrap; not endpoint/transfer qualification')
    print(json.dumps(value,sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
