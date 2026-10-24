#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
BASE=/refactor/candidates/Kimi-Down-component-build-r8653
mkdir "$BASE"
cp /refactor/incoming/kimi_down_component_r8653.cpp "$BASE/"
ABI=1 # ActualCPU188/183 compiler macro; no torch import or CPUdlopen
g++ -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
 -I/opt/dtk/cuda/cuda/include -I"$COMMON/csrc" -I"$SDK/include" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" \
 -MMD -MF "$BASE/component.d" "$BASE/kimi_down_component_r8653.cpp" \
 -Wl,--no-as-needed /refactor/candidates/Kimi-CCL512-r8606/libKimi_CCL512_r8606.so "$COMMON/python/_infinilm.cpython-310-x86_64-linux-gnu.so" -Wl,--as-needed -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -linfiniccl \
 -Wl,-rpath,"$COMMON/python/infinilm/_lib:$SDK/lib" -o "$BASE/libKimi_down_component_r8653.so"
python - <<'PY'
import pathlib,json,hashlib,subprocess,datetime,os
base=pathlib.Path('/refactor/candidates/Kimi-Down-component-build-r8653');lib=base/'libKimi_down_component_r8653.so';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
deps=(base/'component.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
selected={p:sha(pathlib.Path(p))for p in deps if pathlib.Path(p).is_file()and p.startswith('/refactor/')}
assert all('/incoming/'not in p for p in selected)
assert all('/Common/Kimi/InfiniLM/'in p for p in selected if "/csrc/"in p)
native=pathlib.Path('/refactor/candidates/Kimi-CCL512-r8606/libKimi_CCL512_r8606.so');assert sha(native)=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
actualdynamic=[native,pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so'),pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so'),*[pathlib.Path('/refactor/install/hygon-kimi-5f91ad7-r3672/lib')/n for n in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so']]]
out=dict(actual_dynamic_dependency_SHA256={str(p):sha(p)for p in actualdynamic},source_count=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_dlopen=False,source_SHA256=sha(base/'kimi_down_component_r8653.cpp'),library=str(lib),library_SHA256=sha(lib),headers=selected,undefined=subprocess.check_output(['nm','-D','-C','--undefined-only',str(lib)],text=True),needed=subprocess.check_output(['readelf','-d',str(lib)],text=True))
p=pathlib.Path('/refactor/results/Kimi-Down-component-CPU-r8653.json');assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
PY
