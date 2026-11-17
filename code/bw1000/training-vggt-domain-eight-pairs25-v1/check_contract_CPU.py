"""Verify real two-pair CPU geometry and canonical source/cursor contracts for eight fixed pairs without Torch."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from data_contract import PAIRS, sha, verify_bundle
from domain_overlay import SETTINGS, check_config, render
from native_worker import check_clock
from compare_resume import compare


def rejects(call):
    try:
        call()
    except (ValueError,KeyError,TypeError):
        return True
    raise AssertionError("Bad contract unexpectedly admitted")


def run(source_root, data):
    canonical=source_root/"generation/src/gencuda/backends/vggt_training.py"
    raw=canonical.read_bytes()
    code=render(raw)
    tree=ast.parse(code)
    # The runtime update, checkpoint implementation and control flow are retained byte for byte.
    original=raw.decode()
    candidate=code.decode()
    start=original.index("    raw_state = torch.load(weights")
    until=original.index('        "sequence_count": len(dataset.sequence_list),\n')
    assert original[start:until]==candidate[candidate.index("    raw_state = torch.load(weights"):candidate.index('        "sequence_count": len({batch["seq_name"] for batch in batches}),\n')]
    after=original.index('        "cached_valid_batches": len(batches),\n')
    assert original[after:]==candidate[candidate.index('        "cached_valid_batches": len(batches),\n'):]
    names={"VGGTCheckpointValidationError","_vggt_require","_vggt_contract","_validate_vggt_contract",
           "_vggt_boundary_metadata","_vggt_next_coordinate"}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    constants={"VGGT_CHECKPOINT_SCHEMA","_VGGT_IDENTITY_FIELDS"}
    selected += [node for node in tree.body if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets)]
    namespace={"Any":Any}
    exec(compile(ast.Module(selected,type_ignores=[]),str(canonical),"exec",flags=__import__("__future__").annotations.compiler_flag),namespace)
    base=namespace["_vggt_contract"](identity={k:"a"*64 for k in namespace["_VGGT_IDENTITY_FIELDS"]},
                                     seed=20260911,warmup_steps=5,max_steps=20,
                                     view_order=["pair:"+str(i) for i in range(8)],device_ids=[0])
    boundaries={}
    for number in (10,20,25):
        meta=namespace["_vggt_boundary_metadata"](base,number)
        assert meta["next_batch_cursor"]==number%8 and meta["next_global_optimizer_step"]==number+1
        boundaries[str(number)]=dict(metadata=meta,next=namespace["_vggt_next_coordinate"](meta))
    assert boundaries["10"]["next"]=={"global_optimizer_step":11,"batch_cursor":2}
    assert boundaries["20"]["next"]=={"global_optimizer_step":21,"batch_cursor":4}
    assert boundaries["25"]["next"] is None
    negative=0
    for invalid in (0,5,26,True):
        negative+=rejects(lambda:namespace["_vggt_boundary_metadata"](base,invalid))
    bundle_pin=sha((data/"bundle.json").read_bytes())
    manifest=verify_bundle(data,bundle_pin)
    geometry=[]
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"training-domain-pilot-v1"))
    from tum_vggt_pair import prepare_pair
    for row in manifest["pairs"]:
        pair=json.loads((data/row["manifest"]).read_bytes())
        _,observed=prepare_pair(data/row["root"],pair)
        geometry.append(observed)
    config=dict(SETTINGS,dataset_root=str(data.resolve()),annotation_root=str(data.resolve()),
                dataset_manifest=str(data.resolve()/"bundle.json"),dataset_sha=bundle_pin)
    assert check_config(config)==data.resolve()
    for key,value in (("max_steps",25),("warmup_steps",0),("cached_batches",2),("cards",[False]),
                      ("validate_recovery",True),("cache_frozen_aggregator_features",True),("scheduler","cosine")):
        bad=dict(config)
        bad[key]=value
        negative+=rejects(lambda:check_config(bad))
    changed=raw.replace(b'    max_steps = int(config.get("max_steps", 20))',b'    max_steps = int(config.get("max_steps", 25))')
    negative+=rejects(lambda:render(changed))
    for epoch,deadline,now in ((0,0,0),(100,200,201),(100,7400,150),(100,200,float("nan"))):
        negative+=rejects(lambda:check_clock(epoch,deadline,now))
    check_clock(100,300,150)
    # Comparator fixtures are deliberately synthetic and contain no model state.
    steps=[dict(global_optimizer_step=i,batch_cursor=(i-1)%8,loss=1/(i+1),
                grad_norm=1.,camera_head_grad_norm_before_clip=1.) for i in range(1,26)]
    def checkpoint(number):
        return dict(global_step=number,next_global_step=number+1,next_batch_cursor=number%8,
                    **{key:"b"*64 for key in ("file_SHA256","model_SHA256","optimizer_SHA256",
                                             "placements_SHA256","rng_SHA256","full_boundary_SHA256")})
    fixtures={}
    for index,(arm,first) in enumerate((("continuous",0),("resume20",20))):
        fixtures[arm]=dict(arm=arm,executed_optimizer_steps=25-first,bundle_SHA256=bundle_pin,
                          backend_SHA256=sha(code),checkpoint_identity=base["identity"],epoch=1.,deadline=1801.,
                          process=dict(boot_id="fixture",pid=100+index,startticks=50),
                          steps=deepcopy(steps[first:]),checkpoints={str(i):checkpoint(i) for i in (10,20,25) if i>first},
                          final_state_SHA256="c"*64,resume_source_SHA256="b"*64 if first else None,
                          scheduler=dict(type="constant_learning_rate_in_optimizer_param_groups",
                                         mutable_scheduler=False,learning_rate=2e-5),gradient_accumulation=1)
    assert compare(fixtures)["comparison_passed"]
    for edit in ("source","cursor","process","denominator","epoch"):
        broken=deepcopy(fixtures)
        if edit=="source":
            broken["resume20"]["resume_source_SHA256"]="d"*64
        elif edit=="cursor":
            broken["resume20"]["steps"][0]["batch_cursor"]=1
        elif edit=="process":
            broken["resume20"]["process"]=broken["continuous"]["process"]
        elif edit=="denominator":
            broken["resume20"]["steps"].pop()
        else:
            broken["resume20"]["epoch"]=2.
        negative+=rejects(lambda:compare(broken))
    wrong=deepcopy(fixtures)
    wrong["resume20"]["checkpoints"]["25"]["optimizer_SHA256"]="e"*64
    assert compare(wrong)["comparison_passed"] is False
    wrong=deepcopy(fixtures)
    wrong["resume20"]["steps"][0]["loss"]+=0.001
    assert compare(wrong)["comparison_passed"] is False
    for path in Path(__file__).resolve().parent.glob("*.py"):
        ast.parse(path.read_bytes())
    return dict(state="LOCAL_CPU_CONTRACT_PASS",bundle_SHA256=bundle_pin,overlay_SHA256=sha(code),
                canonical_update_and_checkpoint_bytes_unchanged=True,real_eight_pair_geometry=geometry,
                canonical_checkpoint_metadata_boundaries=boundaries,negative_contracts_rejected=negative,
                comparator_fixture_pass=True,comparator_optimizer_mismatch_false=True,comparator_loss_mismatch_false=True,
                native_Torch_or_checkpoint_state_exercised=False,actual_DCU=0,training_ready=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root",type=Path)
    parser.add_argument("data",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.source_root,args.data)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2)
        stream.write("\n")
    print(json.dumps({key:value for key,value in result.items() if key not in
                      ("real_eight_pair_geometry","canonical_checkpoint_metadata_boundaries")},sort_keys=True))
