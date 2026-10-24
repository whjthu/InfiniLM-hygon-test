#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/Kimi-provider-PMC-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/install/hygon-kimi-5f91ad7-r3672/lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
unset LD_PRELOAD INFINILM_KIMI_PACKED_W2_R6122 INFINILM_KIMI_PACKED_AUTO_R6136
stem="/refactor/results/Kimi-provider-PMC-r8669g-${SLURM_JOB_ID}.${SLURM_STEP_ID}"
for kernel in groupedW134531 staticW2R326U; do
 dir="$stem-$kernel-raw";mkdir -m700 "$dir"
 timeout --kill-after=5s 180 /opt/dtk/hip/bin/hipprof --pmc --pmc-type 2 --kernel-name "$kernel" -d "$dir" -o "$dir/pmc" python /refactor/incoming/Kimi_provider_PMC_r8669g.py /refactor/incoming/Kimi-provider-PMC-manifest-r8669g.json "$stem-$kernel.json" "$kernel"
done
python - "$stem" <<'PY'
import pathlib,json,sys
p=pathlib.Path(sys.argv[1]);rows=[json.loads(pathlib.Path(str(p)+'-'+k+'.json').read_text())for k in['groupedW134531','staticW2R326U']];assert all(z['completed']and z['normal_CUDA_cleanup']for z in rows);pathlib.Path(str(p)+'.json').write_text(json.dumps({'completed':True,'normal_CUDA_cleanup':True,'processes':rows,'scope':'FilteredPMCdiagnostic only; noordinaryW13time/newModelTPS'},indent=2))
PY
