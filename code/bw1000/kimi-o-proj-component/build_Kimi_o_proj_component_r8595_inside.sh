#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
BASE=/refactor/candidates/Kimi-o-proj-component-build-r8595
mkdir "$BASE"
cp /refactor/incoming/kimi_o_proj_component_r8594.cpp "$BASE/"
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
g++ -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
 -I/opt/dtk/cuda/cuda/include -I"$COMMON/csrc" -I"$SDK/include" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" \
 -MMD -MF "$BASE/component.d" "$BASE/kimi_o_proj_component_r8594.cpp" \
 -Wl,--no-as-needed "$COMMON/python/_infinilm.cpython-310-x86_64-linux-gnu.so" -Wl,--as-needed -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -linfiniccl \
 -Wl,-rpath,"$COMMON/python/infinilm/_lib:$SDK/lib" -o "$BASE/libKimi_o_proj_component_r8595.so"
python - <<'PY'
import pathlib,json,hashlib,subprocess,datetime,os
base=pathlib.Path('/refactor/candidates/Kimi-o-proj-component-build-r8595');lib=base/'libKimi_o_proj_component_r8595.so';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
deps=(base/'component.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
selected={p:sha(pathlib.Path(p))for p in deps if pathlib.Path(p).is_file()and p.startswith('/refactor/')}
assert all('/incoming/'not in p for p in selected)
assert all('/Common/Kimi/InfiniLM/'in p for p in selected if "/csrc/"in p)
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_dlopen=False,source_SHA256=sha(base/'kimi_o_proj_component_r8594.cpp'),library=str(lib),library_SHA256=sha(lib),headers=selected,undefined=subprocess.check_output(['nm','-D','-C','--undefined-only',str(lib)],text=True),needed=subprocess.check_output(['readelf','-d',str(lib)],text=True))
p=pathlib.Path('/refactor/results/Kimi-o-proj-component-CPU-r8595.json');assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
PY
