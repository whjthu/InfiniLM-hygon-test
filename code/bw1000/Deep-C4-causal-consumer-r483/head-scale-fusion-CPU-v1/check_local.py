"""Build and run the actual head-scale arithmetic with a local launch recorder."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


D = Path(__file__).resolve().parent
REPO = D.parents[3]
CALLER = REPO/'code/bw1000/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/native-src/deep_c4_constant_slot_metadata_r482.cpp'
PROFILE = REPO/'analysis/bw1000/Deep-r467-post1498-main-phase-decision.json'
CALLER_SHA256 = 'd7d035960c9c05c9b849417fe743f300901503def01426e900d230e7dd19a5da'
PROFILE_SHA256 = '7ffef1122f09ecaaf319da3f55c9e7b76d21ceaf4e086a99106a9016691c2da4'
LAUNCH = b'merge<Items><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run():
    caller = CALLER.read_bytes()
    if sha(caller) != CALLER_SHA256:
        raise ValueError('Deep Native calling translation unit changed')
    markers = (
        b'heads_=64/size_t(tp_)',
        b'float scale=float(1/std::sqrt(8192.))',
        b'auto head=work.alloc({tokens,heads_},rt::DataType::kBFloat16)',
    )
    # The three operator calls use exact signatures from the pinned producer.
    triplet = re.search(rb'auto head_float=work.alloc[^\n]*',caller)
    if any(x not in caller for x in markers) or triplet is None:
        raise ValueError('pinned Deep operator line absent')
    line = triplet[0]
    ordered = (b'),head,head_float);',b'),head_float,score_scale_,scaled_head);',
               b'),scaled_head,pending.query_scales,pending.merged_weights);')
    if not all(x in line for x in ordered) or [line.index(x) for x in ordered] != sorted(line.index(x) for x in ordered):
        raise ValueError('Deep Cast/Mul/Mul order changed')
    profile = PROFILE.read_bytes()
    if sha(profile) != PROFILE_SHA256:
        raise ValueError('Deep profile evidence changed')
    window = json.loads(profile)['ranked_decision'][0]['existing_finer_data']
    windows = {}
    for w in window:
        row = w['host_phase_ms']['indexer_query_features']
        windows[str(w['past'])] = row
    if set(windows) != {'8499','8500'} or any(len(row) != 3 or sorted(row) != row for row in windows.values()):
        raise ValueError('measured Deep phase identity changed')
    raw=(D/'head_scale.cu').read_bytes()
    if raw.count(LAUNCH) != 1 or raw.count(b'#include <cuda_runtime.h>\n') != 1:
        raise ValueError('exact CUDA launch replacement required')
    local=raw.replace(LAUNCH,b'cpu_launch(merge<Items>, grid, threads, 0, stream,\n',1)
    local=local.replace(b'#include <cuda_runtime.h>\n',b'#include "cpu_fixture_runtime.hpp"\n',1)
    if b'<<<' in local:
        raise ValueError('unhandled CUDA launch')
    with tempfile.TemporaryDirectory() as scratch:
        scratch=Path(scratch)
        candidate=scratch/'head_scale.cpp'
        exe=scratch/'head_scale.exe'
        candidate.write_bytes(local)
        subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas',
                        '-ffp-contract=off','-I'+str(D),str(candidate),str(D/'check_cpu.cpp'),'-o',str(exe)],
                       check=True,timeout=35)
        result=json.loads(subprocess.check_output([str(exe)],timeout=12))
    if ({key: result[key] for key in ('cases','exact_values','rejected_launch_inputs')}
            != {'cases':20,'exact_values':152064,'rejected_launch_inputs':16}
            or result['reassociation_would_change_values']<=0):
        raise ValueError('incomplete fixture')
    return dict(verified_source=sha(raw),caller_source=sha(caller),profile_SHA256=sha(profile),
                profile_indexer_query_feature_host_ms_min_median_max=windows,
                compiler=subprocess.check_output(['g++','--version'],text=True,timeout=5).splitlines()[0],
                actual_local_host_compiles=1,actual_local_runs=1,actual_target_compiles=0,
                actual_GPU=0,**result)


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
