#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import os,socket,json
v=sorted(os.sched_getaffinity(0));q={'check':'8owner_CPUmask_beforeload','hostname':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'allowed_CPUs':v,'required':32,'passed':len(v)>=32,'GPU_initialized':False};print(json.dumps(q));assert q['passed']
PY
