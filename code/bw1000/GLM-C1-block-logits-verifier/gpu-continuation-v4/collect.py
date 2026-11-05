"""Read the 16 original owner reports after the GLM continuation step exits."""
from hashlib import sha256
import json
import math
from pathlib import Path

from support import D,OUTPUT,NODES

class TerminalNotVisible(RuntimeError):
    pass

def collect(output,step,epoch):
    output=Path(output)
    manifest=json.loads((D/"GPU_MANIFEST.json").read_bytes())
    identity=dict(run=manifest["run"],actual_step=step,
        source_manifest_SHA256=manifest["source_manifest_SHA256"],
        provider_config_SHA256=manifest["provider_config_SHA256"],
        gpu_start_epoch=epoch,Native_seconds=manifest["Native_seconds"])
    def read(name):
        path=output/name
        value=json.loads(path.read_bytes())
        assert value["identity"]==identity,name
        return value
    expected_pids={}
    values=[]
    for stage,node in enumerate(NODES):
        stage_done=read("block-stage-terminal-%d.json"%stage)
        assert stage_done["normal"] and stage_done["local_processes_exited"]==8 and stage_done["all16_closed"]
        expected_pids[node]=[read("block-stage-%d.json"%stage)["PID"],*read("block-actors-%d.json"%stage)["PIDs"]]
        assert len(set(expected_pids[node]))==9
        for rank in range(8):
            prefix="prefix-continuation-%d-%d.json"%(stage,rank)
            q=read(prefix)
            assert (q["stage"],q["rank"],q["Native_SHA256"],q["private_plan_SHA256"])==(
                stage,rank,manifest["Native_SHA256"],manifest["private_plan_SHA256"])
            assert q["normal_shutdown"] and q["minimum_driver_free_bytes"]>=512*1024**2
            closed=read("closed-%d-%d.json"%(stage,rank))
            assert closed["normal"] and closed["post_close_Torch_liveness"]
            life=json.loads((output/("lifecycle-%d-%d.json"%(stage,rank))).read_bytes())
            assert life["normal"] and life["completed"] and life["destroy_attempts"]==life["shutdown_attempts"]==1
            assert life["destroy_RC"]==life["shutdown_RC"]==life["resident_reset_RC"]==0
            assert life["primary_error"] is None and not life["cleanup_errors"]
            assert len(q["cases"])==3
            cases=[]
            for row,(geometry,width,accepted,past) in zip(q["cases"],
                    (("below",4,0,1935),("above",8,8,2949),("crossing_prompt_slice",8,4,2044)),strict=True):
                assert row["geometry"]==geometry and (row["width"],row["accepted"],row["prefix_tokens"])==(width,accepted,past)
                assert row["continuation_tokens"]==2 and row["first_continuation_past"]==past+accepted and row["final_reset"]
                assert len(row["tail_continuation_logits"])==(2 if stage==1 else 0)
                assert len(row["serial_continuation_CAPI_s"])==len(row["rollback_continuation_CAPI_s"])==2
                assert all(math.isfinite(x) and x>=0 for key in ("serial_continuation_CAPI_s","rollback_continuation_CAPI_s") for x in row[key])
                assert all(math.isfinite(x[k]) and x[k]>=0 for x in row["tail_continuation_logits"]
                           for k in ("max_abs","rmse","reference_rms"))
                cases.append({k:row[k] for k in ("geometry","width","accepted","first_continuation_past",
                    "all_tail_continuation_top1_equal","tail_continuation_logits","serial_continuation_CAPI_s","rollback_continuation_CAPI_s","block")})
            values.append(dict(stage=stage,rank=rank,report_SHA256=sha256((output/prefix).read_bytes()).hexdigest(),
                               minimum_driver_free_bytes=q["minimum_driver_free_bytes"],cases=cases))
    return dict(normal16owner_exit=True,normal_Native_shutdown=True,owners=16,cases=3,
        expected_node_PIDs=expected_pids,owner_results=values,HTTP=0,quality_qualification=False,
        measured_acceptance=False,clock_scope="Owner CAPI and two-vote block commit; fixed diagnostic continuation.")
