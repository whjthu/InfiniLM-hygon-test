"""Read only the exact original SDXL .596/.598 tensors and attribute the update gap."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import ModuleType


REQUEST = "a93d41a0d31140f1ad99bb105a024e99"
LABEL = "training-sdxl-sdpa-adjoint-fp32-v1"
PUBLIC = "afdb55bbed267512e07428f36600cbeef25819cbb87084188465d30ff0b0a15f"
REFERENCE = "c59a71095924edfdfb9ad7041b8ef32f747e7825557a313f72ad09a3b0b53036"
REFERENCE_BYTES = 74257307
OBSERVED = "bda1a100837fa939137c925f14b1f60bffddde0bc38aa5af94514f0e51f5d37c"
OBSERVED_BYTES = 73890269
SUMMARY = "928964cffb6729df06787508b854b48fc649d840735cf7f17387adb7395abe68"
SUMMARY_BYTES = 935
COMPARISON_BYTES = 1783928
POLICY = "075e86e54a3eeefb8d5aad219a23f16959d14bd42b22103cd5fe5302c823f6b5"
POLICY_BYTES = 2677
HELPER = "68f7a24c0bd448f36eba7defd4c1c2e0779ef074fd7de1396fbafc2769b19518"
REPLAY = "e8f0e5d333b87ee11a3cc01ae61c37cac7543d32892ddf3129c803b450767142"
COMPARATOR = "f8e2ad7c11a3b236306a69bc6c7bd429b95de623de5568fe54fa7f66e9c4e3a2"
TENSORS, ELEMENTS = 1120, 5806080
TERMS = ("preclip_gradient","clipping_increment","platform_residual")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hex_digest(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value)


def pinned(path, digest, limit, size=None):
    with path.open("rb") as handle:
        raw = handle.read(limit+1)
    require(hex_digest(digest) and 0 < len(raw) <= limit and (size is None or len(raw)==size)
            and hashlib.sha256(raw).hexdigest()==digest, "identity mismatch: "+path.name)
    return raw


def close_evidence(run, public_path, reference_dir):
    proof = json.loads(pinned(public_path,PUBLIC,3739,3739))
    require(proof["request"]==REQUEST and proof["post_clean"] is True and proof["numeric_passed"] is False
            and proof["measured_candidate_actual_update"]==1
            and proof["native_summary_sha256"]==SUMMARY
            and all(n==TENSORS for n in proof["parameter_counts"].values()), "original public decision differs")
    terminal = json.loads(pinned(run/"terminal.json",proof["terminal_SHA256"],65536))
    require(terminal["request"]==REQUEST and terminal["step"]=="1087656.596"
            and terminal["comparison_complete"] is True and terminal["gpu_passed"] is True
            and terminal["clean_post"] is True, "original terminal or comparison absent")
    rows={row[0]:row for event in terminal["runs"] if (row:=event["terminal"]["row"])}
    require(rows["1087656.596"][1:4]==[LABEL,"COMPLETED","0:0"]
            and rows["1087656.598"][1:4]==[LABEL+"-compare","COMPLETED","0:0"], "wrong comparison identity")
    comparison = json.loads(pinned(run/"comparison.json",proof["full_private_comparison_SHA256"],
                                   COMPARISON_BYTES,COMPARISON_BYTES))
    require(comparison["stage_id"]=="sdxl-sft" and comparison["reference_sha256"]==REFERENCE
            and comparison["policy_sha256"]==POLICY and comparison["comparator_sha256"]==COMPARATOR
            and comparison["accelerator"]=="hip" and comparison["boundary"]=="fixed_realized_inputs"
            and comparison["replay_receipt_sha256"]==SUMMARY and comparison["observed_sha256"]==OBSERVED
            and comparison["passed"] is False and comparison["checks"]==proof["checks"]
            and "structural_failure" not in comparison, "comparison identity or gate differs")
    for name in ("gradients_before_clip","gradients_after_clip","parameter_update"):
        metric=comparison["metrics"][name]
        require(len(metric["per_tensor"])==TENSORS
                and all(metric[k]==v for k,v in proof["metrics"][name].items()), "comparison denominator differs")
    replay = run/"candidate/replay"
    summary = json.loads(pinned(replay/"summary.json",SUMMARY,SUMMARY_BYTES,SUMMARY_BYTES))
    require(summary["observed_sha256"]==OBSERVED
            and summary["replay_source_sha256"]==REPLAY and summary["runtime"]["accelerator"]=="hip"
            and summary["runtime"]["cuda_count"]==1 and summary["runtime"]["cuda_available"] is True
            and all(summary[k]==comparison[k] for k in ("stage_id","reference_sha256","policy_sha256","boundary")),
            "one-device original replay receipt differs")
    policy = json.loads(pinned(reference_dir/"policy.json",POLICY,POLICY_BYTES,POLICY_BYTES))
    gate = policy["metrics"]["parameter_update_relative_l2_max"]
    require(policy["stage_id"]=="sdxl-sft" and policy["reference_payload_sha256"]==REFERENCE and gate==0.05,
            "original policy threshold differs")
    return replay/"observed.pt",comparison["metrics"],gate


def archive(torch,path,size,digest):
    require(type(size) is int and 0<size<=128*1024**2, "exact archive size is outside the bound")
    def identity():
        stat = path.stat()
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns
    observed = identity()
    require(observed[2]==size and hex_digest(digest), "exact archive size missing or different")
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for raw in iter(lambda:handle.read(1024**2),b""):
            value.update(raw)
    require(value.hexdigest()==digest and identity()==observed, "archive bytes differ: "+path.name)
    data=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    require(identity()==observed,"archive changed across load: "+path.name)
    return data


def helper_module(path):
    raw = pinned(path,HELPER,32768)
    helper = ModuleType("actual_original398_qualified_cpu_primitives")
    exec(compile(raw,str(path),"exec"),helper.__dict__)
    return helper


def zeros():
    return {"native":0.0,"reference_native":0.0,"sign_CPU":0.0,"opposite_native":0.0,
            "sign_high_margin_native":0.0,"opposite":0,"opposite_ref_above_16eps":0,
            "gram":[[0.0]*len(TERMS) for _ in TERMS]}


def finalize(energy,limit):
    gram=energy["gram"]
    base=energy["reference_native"]
    native=energy["native"]
    total=math.fsum(x for row in gram for x in row)
    require(base>0 and native>0 and all(math.isfinite(x) for x in (base,native,total))
            and math.isclose(total,native,rel_tol=1e-10,abs_tol=1e-18), "three-term squared error does not close")
    return dict(reference_native_squared_l2=base,native_error_squared_l2=native,
                native_relative_l2=math.sqrt(native/base),gate=limit,
                squared_error_reduction_required=max(0.0,1-limit**2*base/native),
                component_names=TERMS,component_gram_over_reference_squared=[[x/base for x in row] for row in gram],
                clipped_gradient_only_relative_l2=math.sqrt((gram[0][0]+gram[1][1]+2*gram[0][1])/base),
                cpu_sign_only_relative_l2=math.sqrt(energy["sign_CPU"]/base),
                opposite_sign_count=energy["opposite"],opposite_sign_ref_above_16eps_count=energy["opposite_ref_above_16eps"],
                native_error_squared_share_opposite_sign=energy["opposite_native"]/native,
                native_error_squared_share_opposite_sign_ref_above_16eps=energy["sign_high_margin_native"]/native)


def compare(torch, helper, ref, hip, expected, threshold):
    names=list(ref["initial_adapter"])
    tables=("initial_adapter","final_adapter","gradients_before_clip","gradients_after_clip")
    require(len(names)==TENSORS and all(list(book[k])==names for book in (ref,hip) for k in tables),
            "ordered population differs from all 1120 trainables")
    options=helper.group(ref,len(names))
    require(helper.group(hip,len(names))==options and
            (options["lr"],options["eps"],options["weight_decay"],tuple(options["betas"]))==(1e-4,1e-8,.01,(.9,.999)),
            "real empty-state AdamW options differ")
    require(all(options.get(k) is None for k in ("foreach","fused")), "original default optimizer mode differs")
    for book in (ref,hip):
        state=book["final_optimizer"]["state"]
        require(list(state)==list(range(TENSORS)) and all(state[n]["step"].device.type=="cpu"
                and state[n]["step"].numel()==1 and float(state[n]["step"])==1 for n in state),
                "archive does not describe exactly one AdamW update")
    initial = [helper.checked_tensor(ref["initial_adapter"],name,ref["initial_adapter"][name].shape) for name in names]
    require(sum(t.numel() for t in initial)==ELEMENTS, "actual full parameter element denominator differs")
    require(all(torch.equal(value,helper.checked_tensor(hip["initial_adapter"],name,value.shape))
                for name,value in zip(names,initial)), "actual initial adapters differ")
    values=[
        {k:[helper.checked_tensor(book[k],name,p.shape) for name,p in zip(names,initial)] for k in tables[1:]}
        for book in (ref,hip)
    ]
    pre, post = ([value[k] for value in values] for k in ("gradients_before_clip","gradients_after_clip"))
    changed=[
        sum(int((x.contiguous().view(torch.int16)!=y.contiguous().view(torch.int16)).count_nonzero())
            for x,y in zip(before,after)) for before,after in zip(pre,post)
    ]
    limits=ref["arguments"]["max_grad_norm"]
    raw_norms=[book["preclip_norm"] for book in (ref,hip)]
    require(all(t.device.type=="cpu" and t.numel()==1 and t.is_floating_point() for t in raw_norms),
            "archived preclip norm changed scalar schema")
    norms=[float(t) for t in raw_norms]
    require(type(limits) in (int,float) and math.isfinite(limits) and limits>0
            and all(math.isfinite(n) and n>0 for n in norms), "true preclip norm or configured threshold invalid")
    cpu_post=[helper.cpu_replay(initial,gradient,options)[0] for gradient in post]
    cpu_pre=[helper.cpu_replay(initial,pre[i],options)[0] if changed[i] else cpu_post[i] for i in range(2)]
    flipped=[(a!=0)&(b!=0)&(a.sign()!=b.sign()) for a,b in zip(*post)]
    sign_only=[torch.where(mask,-b,b) for mask,b in zip(flipped,post[1])]
    cpu_sign=helper.cpu_replay(initial,sign_only,options)[0]
    energy=zeros()
    for i in range(TENSORS):
        start=initial[i].double()
        ref_native=values[0]["final_adapter"][i].double()-start
        native=values[1]["final_adapter"][i].double()-values[0]["final_adapter"][i].double()
        pre_effect=cpu_pre[1][i].double()-cpu_pre[0][i].double()
        post_effect=cpu_post[1][i].double()-cpu_post[0][i].double()
        components=[pre_effect,post_effect-pre_effect,native-post_effect]
        for j,left in enumerate(components):
            for k,right in enumerate(components):
                energy["gram"][j][k]+=float((left*right).sum())
        sign_difference=cpu_sign[i].double()-cpu_post[0][i].double()
        opposite=flipped[i]
        margin=opposite & (post[0][i].double().abs()>16*options["eps"])
        for key, delta in (("reference_native",ref_native),("native",native),("sign_CPU",sign_difference),
                            ("opposite_native",native[opposite]),("sign_high_margin_native",native[margin])):
            energy[key]+=float(delta.square().sum())
        energy["opposite"]+=int(opposite.count_nonzero())
        energy["opposite_ref_above_16eps"]+=int(margin.count_nonzero())
    metrics=finalize(energy,threshold)
    require(math.isclose(metrics["native_relative_l2"],expected["parameter_update"]["relative_l2"],rel_tol=1e-9)
            and math.isclose(math.sqrt(energy["reference_native"]),expected["parameter_update"]["reference_l2"],rel_tol=1e-9),
            "native update and denominator did not reproduce the original comparison")
    return dict(metrics,tensors=TENSORS,elements=ELEMENTS,postclip_changed_elements=changed,
                training_clip_threshold=limits,training_saved_preclip_norms=norms,
                cpu_post_replays=2,cpu_pre_additional_replays=sum(v>0 for v in changed),cpu_sign_replays=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","public","reference-dir","optimizer-helper"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    observed,metrics,threshold=close_evidence(args.run,args.public,args.reference_dir)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized() and torch.cuda.device_count()==0,"CPU scope requires zero visible accelerators")
    reference=archive(torch,args.reference_dir/"reference.pt",REFERENCE_BYTES,REFERENCE)
    candidate=archive(torch,observed,OBSERVED_BYTES,OBSERVED)
    result=compare(torch,helper_module(args.optimizer_helper),reference,candidate,metrics,threshold)
    require(not torch.cuda.is_initialized(),"a CUDA runtime was initialized")
    print(json.dumps(dict(result,actual_gpu_step="1087656.596",actual_comparator_step="1087656.598",
                          observed_archive_SHA256=OBSERVED,observed_archive_bytes=OBSERVED_BYTES,
                          reference_archive_SHA256=REFERENCE,reference_archive_bytes=REFERENCE_BYTES,
                          GPU_cards=0,elapsed_seconds=time.monotonic()-started,training_qualified=False),
                     sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
