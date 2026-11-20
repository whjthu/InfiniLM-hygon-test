"""Compile all five exact I128 sources against a local CUDA-launch recorder."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import tempfile


D = Path(__file__).resolve().parent
REPO = D.parents[3]
ROOT = REPO/'code/bw1000/GLM952-global-TP16-component-r470/full-native-source-v1'
ORIGINAL = {
    'routed_i128.cu': '4adb7a1c1b11b6a4434c93b8e9787966bec6a71787fa4837a5ed13e3afc128b6',
    'routed_i128.hpp': '95264c6f5afeaf73280c4d6fb63e3ef6b0152468930cf789eb965508afa5ea67',
}
AUDIT_SHA = '5db40da457f7b1449b30764bb9c224a5cd4f50824555b2fa866294aedca5df46'
LAUNCH = re.compile(rb'\b(validate|gate_up|down)<<<([^\n]+?)>>>\(')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'frozen source drift: {path.name}')
    return data


def to_host(source):
    names = []
    # Only transform triple-chevron launching into a typed recording call.
    # The three kernel bodies, wrapper parameters, predicates and return paths are the source bytes.
    def replace(match):
        kernel, launch = match.groups()
        names.append(kernel)
        return b'record_launch('+kernel+b','+launch+b','
    new = LAUNCH.sub(replace, source)
    if names != [b'validate',b'gate_up',b'down'] or b'<<<' in new:
        raise ValueError('exact three routed launch sites required')
    return b'#include "routed_i128.hpp"\n'+new


def run(cxx):
    audit = D.parent/'routed-i128-cpu-artifact-audit-v1/audit_once.py'
    pinned(audit,AUDIT_SHA)
    api = runpy.run_path(str(audit))
    inventory = api['inventory'](REPO)
    source, header = (pinned(ROOT/name, pin) for name,pin in ORIGINAL.items())
    worlds = [('original',4,8,source)]
    anchors = {}
    for arm in inventory:
        plan, rows, _, folder = api['load'](arm,REPO)
        prof=plan['actual_profile']
        call=plan['actual_caller']
        original_profile = json.loads(pinned(REPO/prof['path'],prof['SHA256']))
        original_caller = pinned(REPO/call['path'],call['SHA256'])
        window = 'ffn.routed_gateup' if arm=='gate' else 'ffn.routed_down'
        stream = {layer: detail['windows'][window]['stream']
                  for layer,detail in original_profile['detailed_layers'].items()
                  if window in detail['windows']}
        published = prof['gateup_stream_ms' if arm=='gate' else 'down_stream_ms']
        if stream != published or len(stream) != 6 or any(row['samples'] != 16 for row in stream.values()):
            raise ValueError('measured layer or per-rank profile scope drift')
        if f'tp_==16?glm_r470_i128_{arm}('.encode() not in original_caller:
            raise ValueError('frozen caller no longer reaches the measured TP16 leaf')
        anchors[arm]=dict(original_step=prof['step'],profile_SHA256=prof['SHA256'],
                          caller_SHA256=call['SHA256'],measured_layer_count=len(stream),
                          stream_median_us=[1000*min(x['median_ms'] for x in stream.values()),
                                            1000*max(x['median_ms'] for x in stream.values())])
        for row in rows:
            variants = (row,8) if arm=='gate' else (4,row)
            raw = pinned(folder/f'rows{row}.cu',plan['variants'][str(row)]['source_SHA256'])
            worlds.append((f'{arm}_rows{row}',*variants,raw))
    out = []
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        (work/'routed_i128.hpp').write_bytes(header)
        for label,gate,down,source in worlds:
            path=work/(label+'.cpp')
            path.write_bytes(to_host(source))
            for branch in ('hip', 'cuda'):
                exe=work/(label+'-'+branch+'.exe')
                cmd=[cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas',
                     f'-DGATE_ROWS={gate}',f'-DDOWN_ROWS={down}', '-I'+str(work),'-I'+str(D)]
                if branch=='hip':
                    cmd.append('-D__HIPCC__=1')
                subprocess.run(cmd+[str(path),str(D/'check_wrappers.cpp'),'-o',str(exe)],check=True,timeout=30)
                raw=subprocess.check_output([str(exe)],timeout=8)
                counts=json.loads(raw)
                if counts != dict(valid_launch_checks=72,rejected_argument_checks=12,compiled_validate_kernel_checks=16):
                    raise ValueError('incomplete local fixture denominator')
                out.append(dict(label=label,shuffle_preprocessor_branch=branch,source_SHA256=sha(source),**counts))
    return dict(qualification='LOCAL_HOST_WRAPPER_ABI_FIXTURE',
                compiler_version=subprocess.check_output([cxx,'--version'],text=True,timeout=5).splitlines()[0],
                full_native_source_files=len(worlds),actual_local_host_compiles=len(out),actual_local_host_runs=len(out),
                exact_frozen_header_SHA256=sha(header),real_profile_and_caller=anchors,
                results=out,actual_target_compiles=0,actual_GPU=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cxx',default='g++')
    args=p.parse_args()
    print(json.dumps(run(args.cxx),indent=2))
