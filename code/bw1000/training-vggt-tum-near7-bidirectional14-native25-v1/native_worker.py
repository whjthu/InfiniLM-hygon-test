"""Run exactly one fresh VGGT continuous or checkpoint-resume process. No Slurm calls."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time

from split_contract import BUNDLE_SHA, BACKEND_SHA, TRAIN, descriptors, verify_steps, need
from checkpoint_io import checkpoint_sha256
from view_schedule import VIEW_BATCHES, validate_observed_views, view_order_sha256

ARMS = {"continuous":0}
WEIGHT_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"


def save(path,value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")


def process_id():
    fields=Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()
    return dict(pid=os.getpid(),startticks=int(fields[19]),boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def check_clock(epoch,deadline,now):
    if (not all(math.isfinite(x) for x in (epoch,deadline,now))
            or not 0<deadline-epoch<=1800 or not epoch<=now<deadline-30):
        raise ValueError("Invalid or exhausted original experiment epoch")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=ARMS,required=True)
    for key in ("data","weights","upstream","output_root"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    parser.add_argument("--bundle-sha256",required=True)
    parser.add_argument("--overlay-sha256",required=True)
    args=parser.parse_args()
    epoch=float(os.environ["VGGT_DOMAIN_EPOCH"])
    deadline=float(os.environ["VGGT_DOMAIN_DEADLINE"])
    check_clock(epoch,deadline,time.time())
    def expired(*_):
        raise TimeoutError("Original VGGT domain epoch exhausted")
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    for path in (args.data,args.weights,args.upstream,args.output_root):
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("Expected fixed absolute, unaliased worker roots")
    root=args.output_root/args.arm
    root.mkdir()
    invocation=dict(arm=args.arm,epoch=epoch,deadline=deadline,process=process_id())
    save(root/"invocation.json",invocation)
    from gencuda import domain_overlay
    from gencuda.backends import vggt_training as backend
    backend_path=Path(backend.__file__)
    backend_source=backend_path.read_bytes()
    if (backend_path!=Path("/sealed/framework/generation/src/gencuda/backends/vggt_training.py")
            or hashlib.sha256(backend_source).hexdigest()!=args.overlay_sha256):
        raise ValueError("Deployed VGGT domain backend changed")
    if args.bundle_sha256!=BUNDLE_SHA or args.data!=Path("/sealed/data"):
        raise ValueError("expected original complete source mount")
    config=dict(domain_overlay.SETTINGS,dataset_root=str(args.data),annotation_root=str(args.data),
                dataset_manifest=str(args.data/"bundle.json"),dataset_sha=args.bundle_sha256)
    completed=ARMS[args.arm]
    resume_sha=None
    claim=json.loads(Path(__file__).with_name('CLAIM.json').read_bytes())
    overlay_path=Path(domain_overlay.__file__)
    need(overlay_path==Path('/sealed/framework/generation/src/gencuda/domain_overlay.py') and
         hashlib.sha256(overlay_path.read_bytes()).hexdigest()==claim['runtime_overlay_SHA256'],
         'scoped collector file-bind identity differs')
    need(domain_overlay.SETTINGS['cached_batches']==VIEW_BATCHES and domain_overlay.SETTINGS['learning_rate']==2e-5,
         'frozen fourteen-orientation schedule differs')
    _,_,declaration=descriptors(args.data)
    save(root/"config.json",config)
    import torch
    if (torch.cuda.device_count()!=1 or torch.version.hip!="6.3.26113"
            or importlib.metadata.version("torch")!="2.10.0+das.opt1.dtk2604.2606161421.g91a580"):
        raise ValueError("Expected the qualified single-device HIP runtime")
    result=backend.run_vggt_training(args.weights,config,root/"train",args.upstream)
    m=result["metrics"]
    for key,expected in (("global_optimizer_step",25),("completed_global_step_before_resume",completed),
                         ("executed_optimizer_steps",25-completed),("measurement_steps",20-max(0,completed-5)),
                         ("cached_valid_batches",VIEW_BATCHES),("sequence_count",1),("rejected_invalid_depth_batches",0)):
        if m[key]!=expected:
            raise ValueError("Incorrect native completion: "+key)
    for key in ("camera_head_weights_changed","frozen_aggregator_unchanged","main_forward_device",
                "nonzero_finite_camera_head_gradients","resume_boundary"):
        if result["checks"][key] is not True:
            raise ValueError("Native qualification failed: "+key)
    if (m["checkpoint_identity_evidence"]["frozen_model"]["weights_model_pt"]!=WEIGHT_SHA256
            or m["resume_source_sha256"]!=resume_sha or m["formal_CO3D_eligible"] is not False):
        raise ValueError("Original model, resume identity or domain provenance changed")
    if (m["training_scope"]!="camera_head" or m["frozen_aggregator_feature_cache"] is not False
            or m["trainable_backbone_parameters"]!=0 or m["frames_per_step"]!=2
            or m["trainable_parameters"]!=m["trainable_camera_head_parameters"] or m["trainable_parameters"]<=0):
        raise ValueError("Native trainable parameters, frozen aggregator or frame scope differ")
    cohort=m['domain_geometry']
    need(cohort['verified_pairs']==7 and cohort['physical_recordings']=={TRAIN:7} and
         cohort['derived_views']==VIEW_BATCHES and cohort['view_order_SHA256']==claim['view_order_SHA256']==view_order_sha256() and
         cohort['split']==declaration and cohort['heldout_geometry_loaded_by_training'] is False,
         'train/holdout closure or native fourteen-orientation consumption differs')
    validate_observed_views(cohort['rows'])
    native_view_order = [row["sequence"] + ":" + ",".join(map(str, row["input_frame_ids"]))
                         for row in cohort["rows"]]
    checkpoints={}
    for number in (10,20,25):
        path=root/"train"/f"checkpoint-{number}.pt"
        if number<=completed:
            continue
        payload=torch.load(path,map_location="cpu",weights_only=False)
        metadata=payload["metadata"]
        if (metadata!=backend._vggt_boundary_metadata(metadata["contract"],number)
                or metadata["contract"]["view_order"]!=native_view_order
                or metadata["next_batch_cursor"]!=number % VIEW_BATCHES
                or metadata["next_global_optimizer_step"]!=number+1):
            raise ValueError("Checkpoint boundary metadata differs")
        unsigned={key:value for key,value in payload.items() if key!="integrity_sha256"}
        if payload["integrity_sha256"]!=backend.resume_common.tree_sha256(unsigned):
            raise ValueError("Checkpoint integrity differs")
        state=payload["state"]
        groups=state["optimizer"]["param_groups"]
        if any(group["lr"]!=2e-5 for group in groups):
            raise ValueError("Constant optimizer learning rate changed")
        checkpoints[str(number)]=dict(
            file_SHA256=checkpoint_sha256(path),global_step=number,
            next_batch_cursor=metadata["next_batch_cursor"],next_global_step=metadata["next_global_optimizer_step"],
            model_SHA256=backend.resume_common.tree_sha256(state["camera_head"]),
            optimizer_SHA256=backend.resume_common.tree_sha256(state["optimizer"]),
            placements_SHA256=backend.resume_common.tree_sha256(state["optimizer_placements"]),
            rng_SHA256=backend.resume_common.tree_sha256(payload["rng"]),
            full_boundary_SHA256=payload["integrity_sha256"],optimizer_group_learning_rates=[g["lr"] for g in groups])
        del payload, metadata, unsigned, state, groups
    records=[json.loads(line) for line in (root/"train/steps.ndjson").read_bytes().splitlines()]
    if [row["global_optimizer_step"] for row in records]!=list(range(completed+1,26)):
        raise ValueError("Incomplete native step denominator")
    verify_steps(records)
    save(root/"native-result.json",result)
    receipt=dict(arm=args.arm,process=invocation["process"],epoch=epoch,deadline=deadline,
                 bundle_SHA256=args.bundle_sha256,backend_SHA256=args.overlay_sha256,
                 result_SHA256=hashlib.sha256((root/"native-result.json").read_bytes()).hexdigest(),
                 checkpoints=checkpoints,final_state_SHA256=m["final_state_sha256"],
                 checkpoint_identity=m["checkpoint_identity"],resume_source_SHA256=resume_sha,
                 input_order_SHA256=cohort["input_order_SHA256"],selection_SHA256=declaration["selection_SHA256"],
                 unique_training_pairs=7,derived_training_views=VIEW_BATCHES,view_order_SHA256=view_order_sha256(),
                 steps=records,executed_optimizer_steps=25-completed,formal_CO3D_eligible=False,
                 scheduler=dict(type="constant_learning_rate_in_optimizer_param_groups",mutable_scheduler=False,
                                learning_rate=2e-5),gradient_accumulation=1)
    save(root/"receipt.json",receipt)
    print(json.dumps(dict(arm=args.arm,actual_updates=25-completed,final_step=25,
                         receipt_SHA256=hashlib.sha256((root/"receipt.json").read_bytes()).hexdigest())))
    from evaluate import run
    head=root/'train/camera_head.pt'
    matches=[a for a in result['artifacts'] if a['path']==str(head)]
    need(len(matches)==1, 'new native head artifact identity absent')
    metric=run(args.data,args.weights,args.upstream,head,matches[0]['sha256'],
               m['camera_head_sha_after'],root/'evaluation')
    save(root/'effect-summary.json',metric)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
