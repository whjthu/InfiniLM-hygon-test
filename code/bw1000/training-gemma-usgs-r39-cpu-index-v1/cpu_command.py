"""Build one private zero-GRES CPU-index argv. No submissions."""

from pathlib import PurePosixPath
import re
import shlex

from prepare import NEW,REQUEST,ROOT,need

R26_NATIVE="training-gemma-usgs-r26-native-image-CPU-v1/537844da9e9b470993209b74ad45c7f7/data"


def cpu_argv(epoch,lineage_sha):
    need(re.fullmatch(r"[0-9a-f]{64}",lineage_sha) and type(epoch) in (int,float),"R39_EXACT_CPU_INDEX_SCOPE")
    r=PurePosixPath(ROOT.as_posix())
    new=PurePosixPath(NEW.as_posix())
    src=new/"r39-source"
    old=r/"training-gemma-usgs72-fresh-initial-domain25-v1"
    media=r/"training-gemma-usgs-render72-native-CPU-v1/data"
    r26=r/R26_NATIVE
    processor=r/"training-gemma-processor-assets-v1/models/gemma-3-4b-it"
    argv=["srun","--jobid=1103606","--job-name=training-gemma-usgs-r39-two-updates-v1-index-"+REQUEST[:12],
          "--exact","--immediate=5","-N1","-n1","--nodelist=f13r1n12","--cpus-per-task=4","--mem=8192M",
          "--gres=none","--time=00:04:00","/public/software/apps/apptainer/1.3.4/bin/apptainer",
          "exec","--contain","--no-home","--pwd","/"]
    for source,mode in ((new,"rw"),(src,"ro"),(old,"ro"),(media,"ro"),(r26,"ro"),(processor,"ro")):
        argv+=["--bind",f"{source}:{source}:{mode}"]
    for source,alias in ((r/"training-vlm-swift-offline-runtime-v1","/swift-runtime"),
                         (r/"training-minicpm-hygon-processor-r2/cpu/runtime-site","/minicpm-transformers"),("/usr/local/hyhal","/opt/hyhal")):
        argv+=["--bind",f"{source}:{alias}:ro"]
    env={"CUDA_VISIBLE_DEVICES":"-1","HIP_VISIBLE_DEVICES":"-1","ROCR_VISIBLE_DEVICES":"-1","HF_HUB_OFFLINE":"1",
         "TRANSFORMERS_OFFLINE":"1","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","OMP_NUM_THREADS":"2",
         "PYTHONPATH":"/minicpm-transformers:/swift-runtime/site:"+str(old/"framework")+":"+str(src)}
    python=["python3","-B","-u",str(src/"cpu_index.py"),"--epoch",str(epoch),"--lineage-sha256",lineage_sha]
    shell="set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; export "
    shell+=" ".join(f"{k}={shlex.quote(v)}" for k,v in env.items())+"; exec "+shlex.join(python)
    return argv+[str(r.parent/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",shell]
