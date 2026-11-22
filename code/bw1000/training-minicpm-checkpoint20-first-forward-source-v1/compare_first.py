"""Compare the new single-forward witnesses with the original checkpoint-20 metadata."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from probe import require

ORIGINAL_PUBLIC="816e3561638b1625aa18038b8dfb50fc2e9c343570501cbde6a7c29839d5be86"
ORIGINAL_COMPARISON="bf37e76945e669d7727dfa01ca436c49dd006db8e249d0c0d290ed037160c5d9"
FIELDS=("RNG_after_native_restore","RNG_before_compute_loss","root_before_forward","inner_linear",
        "language_before_forward","RNG_after_forward","ddp","compute_loss_source","operators")


def read(path,limit,digest=None):
    with path.open("rb") as stream:
        raw=stream.read(limit+1)
    require(0<len(raw)<=limit and (digest is None or sha256(raw).hexdigest()==digest),
            "bounded original metadata identity differs: "+path.name)
    return json.loads(raw)


def original_rows(root,public):
    proof=read(public,3271,ORIGINAL_PUBLIC)
    require(proof["actual_accounting"]["gpu"][0]=="1103606.1171"
            and proof["original_comparison_SHA256"]==ORIGINAL_COMPARISON
            and proof["independent_exact_recomputation"] is True,"not the consumed native 25+5 run")
    original=read(root/"paired-recovery.json",2*1024**2,ORIGINAL_COMPARISON)
    pins=original["evidence_SHA256"]
    require(original["ranks"]==8 and original["paired_remaining_microbatches"]==160 and len(pins)==56,
            "original comparator denominator differs")
    rows={}
    for rank in range(8):
        def artifact(arm,name):
            rel=Path(arm)/"train/minicpm-usgs-witness"/f"rank-{rank:02d}"/name
            return read(root/rel,8*1024**2,pins[rel.as_posix()])
        cont=artifact("continuous","result.json")
        resumed=artifact("resumed","result.json")
        saved=artifact("continuous","checkpoint-20-rng.json")
        original_input=cont["inputs"][80]
        resume_input=resumed["inputs"][0]
        require(cont["rank"]==resumed["rank"]==saved["rank"]==rank
                and original_input["ordinal"]==81 and resume_input["ordinal"]==1
                and original_input["global_step"]==resume_input["global_step"]==20
                and original_input["full_input"]==resume_input["full_input"]
                and cont["steps"][19]["global_step"]==20
                and resumed["RNG_loads"][0]["saved_equal"] is True and resumed["RNG_loads"][0]["actual"]==saved["RNG"],
                "original first microbatch, checkpoint or saved RNG differs")
        rows[rank]=dict(input=resume_input["full_input"],state=cont["steps"][19]["state"],RNG=saved["RNG"],
                        continuous_loss=cont["outputs"][80],resumed_loss=resumed["outputs"][0])
    return rows


def probe_rows(train):
    rows={}
    for rank in range(8):
        folder=train/"minicpm-usgs-witness"/f"rank-{rank:02d}"
        row=read(folder/"first-forward-state.json",2*1024**2)
        stopped=read(folder/"first-forward-stop.json",4096)
        require(row["rank"]==rank and row["world_size"]==8 and row["checkpoint_step"]==row["global_step"]==20
                and row["state"]=="STOP_BEFORE_BACKWARD" and row["planned_forwards"]==row["completed_forwards"]==1
                and row["backward_calls"]==row["optimizer_updates"]==0
                and stopped==dict(rank=rank,state="ALL_RANKS_STOPPED",global_step=20,backward_calls=0,optimizer_updates=0),
                "incomplete eight-rank stop-before-backward receipt")
        rows[rank]=row
    return rows


def compare(records,original,repeat=None):
    require(set(records)==set(original)==set(range(8)) and (repeat is None or set(repeat)==set(records)),
            "requires exactly eight pinned original ranks")
    rows=[]
    for rank,row in sorted(records.items()):
        historical=original[rank]
        require(row["actual_input"]["full_input"]==historical["input"] and row["restored_state"]==historical["state"]
                and row["RNG_after_native_restore"]==historical["RNG"],"first runtime input or restore identity differs")
        loss=row["actual_loss"]
        require(loss["ordinal"]==1 and type(loss["loss_scalar"]) is float and math.isfinite(loss["loss_scalar"]),
                "invalid first native loss")
        root=row["root_before_forward"]
        info=dict(rank=rank,RNG_diff_after_native_restore_to_compute=row["RNG_before_compute_loss"]!=historical["RNG"],
                  RNG_diff_during_wrapper_to_root=root["RNG"]!=row["RNG_before_compute_loss"],
                  RNG_diff_root_to_loss=row["RNG_after_forward"]!=root["RNG"],
                  root_autocast=root["context"]["autocast_enabled"],
                  language_autocast=row["language_before_forward"]["context"]["autocast_enabled"],
                  first_linear_autocast=row["inner_linear"]["context"]["autocast_enabled"],
                  SDPA_route_observed=row["operators"]["SDPA_route_observed"],
                  history_has_boundary_context=False)
        for arm in ("continuous","resumed"):
            previous=historical[arm+"_loss"]
            info[arm+"_loss_match"]=loss["loss"]==previous["loss"] and loss["loss_scalar"]==previous["loss_scalar"]
            info[arm+"_absolute_loss_delta"]=abs(loss["loss_scalar"]-previous["loss_scalar"])
        if repeat is not None:
            second=repeat[rank]
            require(second["actual_input"]["full_input"]==historical["input"]
                    and second["restored_state"]==historical["state"],"repeat input/state differs")
            info["repeat_differences"]=[key for key in (*FIELDS,"actual_loss") if row[key]!=second[key]]
        rows.append(info)
    return dict(ranks_planned=8,ranks_compared=len(rows),micro_forwards_per_rank=1,backward_calls=0,optimizer_updates=0,
                compared_new_repeat=repeat is not None,rows=rows,continuous_context_recovered=False,
                rank1_and_rank6_later_first_difference_covered=False,formal_recovery_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for label in ("original-root","original-public","probe-train"):
        parser.add_argument("--"+label,type=Path,required=True)
    parser.add_argument("--repeat-probe-train",type=Path)
    args=parser.parse_args()
    second=probe_rows(args.repeat_probe_train) if args.repeat_probe_train else None
    print(json.dumps(compare(probe_rows(args.probe_train),original_rows(args.original_root,args.original_public),second),
                     sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
