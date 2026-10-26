#!/bin/bash
set -euo pipefail
python3 -c 'import os,json;v=sorted(os.sched_getaffinity(0));print(json.dumps({"node":os.uname().nodename,"step":os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],"allowed_CPUs":v,"required":32,"passed":len(v)>=32,"GPU_initialized":False}));assert len(v)>=32'
