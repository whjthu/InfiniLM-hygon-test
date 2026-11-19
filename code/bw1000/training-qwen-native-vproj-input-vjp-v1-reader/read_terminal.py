"""Validate Slurm, the eight training receipts, and the measured real-input replay."""
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

LABEL = "training-qwen-native-vproj-input-vjp-v1"
REQUEST = "4e306ff0a6a047f0977ae72b4b4035f7"
PUBLICATION = "60ed8f371de372b5c82cae1cbcde2de01928182e"
REF = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ROOT = REF / LABEL
CONTROL = REF / "training-qwen-native-lora-weight-vjp-v1"
SITE = "decoder_06.v_proj"
PRIOR = {'outcome.json': {'SHA256': '08110e8e7a5c5609abd6d8a475bf1313c888db8b66b78de2317d2be6217e477e', 'bytes': 22824}, 'capture/rank-00/receipt.json': {'SHA256': '46b2e210a9284d5e6d82787b09d811943ed4588b503e1c17b368eb59c5ea102e', 'bytes': 4294}, 'capture/rank-01/receipt.json': {'SHA256': '9991e8bcdb6d9dbe6237243a23dbb6f26511802a352db42e11177db54417d3db', 'bytes': 4293}, 'capture/rank-02/receipt.json': {'SHA256': '72b263d6a97378b39da6fb4911694799851396e4e439d4df327a11248d1a41f0', 'bytes': 4293}, 'capture/rank-03/receipt.json': {'SHA256': '13d2a177649dcbaaa00b70f0274b7d697c9d8182868b9096ce3e824229a14012', 'bytes': 4297}, 'capture/rank-04/receipt.json': {'SHA256': 'ac2ed3db99095fcb1f4f1d997a63a613d7280b0eb6a49209d01a0dffc3126720', 'bytes': 4292}, 'capture/rank-05/receipt.json': {'SHA256': '1a782986e82e361084d2e83cff44a8e903237fa14b67a1b7b05f63a083548794', 'bytes': 4293}, 'capture/rank-06/receipt.json': {'SHA256': '99786e93d6be46e61d3f54820c97667ae15be378c06466c3fccd6d81c9abf071', 'bytes': 4293}, 'capture/rank-07/receipt.json': {'SHA256': '99049bf0c5cef6b442eb93d4f184803769d42d264a240429f34eeb992046496d', 'bytes': 4294}}
HEX = re.compile(r"[0-9a-f]{64}")

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def read(path, pins, field=None, limit=512 << 10):
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    require(0 < len(raw) <= limit, "RECEIPT_SIZE")
    p = {"bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}
    pins[field or path.relative_to(ROOT).as_posix()] = p
    return json.loads(raw)

def accounting():
    columns = "JobIDRaw,JobName%100,State%35,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"
    cmd = ["sacct", "-n", "-P", "-j", "1103606", "--format="+columns]
    data = subprocess.run(cmd, text=True, capture_output=True, timeout=8, check=True).stdout
    rows = [r.split("|")[:7] for r in data.splitlines() if "|" + LABEL in r]
    return columns.split(","), rows

def phase_pins(info, rank):
    require(info["rank"] == rank and info["optimizer_updates"] == info["clip_calls"]
            == info["completed_global_step"] == 2 and info["micro_forwards"] == 8
            and info["state"] == "TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED", "CAPTURE_DENOMINATOR")
    items = {r["name"]: (r["bytes"],r["SHA256"]) for r in info["files"]}
    require(len(items)==len(info["files"])==25
            and all(Path(name).name == name and n > 0 and HEX.fullmatch(s) for name,(n,s) in items.items()),
            "CAPTURE_FILES_IDENTITY")
    return items

def run(prior):
    columns, rows = accounting()
    pins = {}
    data = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), request=REQUEST,
                source_commit=PUBLICATION, accounting_columns=columns, steps=rows, pins=pins,
                complete=False, planned_training_ranks=8, planned_microbatches=64,
                planned_optimizer_updates=2)
    if not (ROOT/"outcome.json").is_file():
        return data
    intent=read(ROOT/"intent.json",pins)
    keeper=read(ROOT/"outcome.json",pins)
    require(intent["request_id"] == REQUEST and intent["claim_commit"] == keeper["claim_commit"] == PUBLICATION
            and intent["epoch"] == keeper["epoch"] and intent["deadline"] == keeper["deadline"] == intent["epoch"] + 1720,
            "ORIGINAL_CONTRACT")
    data["keeper_passed"] = keeper["keeper_passed"]
    data["native_intent_present"] = keeper["native_intent_present"]
    data["GPU_step_requested"] = keeper["GPU_step_requested"]
    before, after = keeper.get("before"), keeper.get("after")
    data["clean_post"] = bool(before and after and after["pci"] == before["pci"]
                              and after["boot_id"] == before["boot_id"] and after["kfd_holder_count"] == 0
                              and not after["owned_processes_remaining"] and after["master_port_available"])
    if not keeper["keeper_passed"]:
        data["error_stage"] = (keeper.get("error") or {}).get("stage")
        data["error_type"] = (keeper.get("error") or {}).get("type")
        files=["native.log", LABEL+".log"]
        for filename in files:
            p=ROOT/filename
            if p.is_file():
                with p.open("rb") as f:
                    f.seek(max(0,p.stat().st_size-8192))
                    rows_tail=f.read(8192).decode("utf-8","replace").splitlines()
                allowed=re.compile(r"(?:ValueError|RuntimeError|TypeError|AssertionError|KeyError): (?:[A-Z][A-Z0-9_]+)?")
                data[filename+"_selected_errors"]=[m.group() for line in rows_tail if (m:=allowed.search(line))][-12:]
        return data
    by_name={r[1]:r for r in rows}
    require(len(rows)==3 and set(by_name)=={LABEL,LABEL+"-pre",LABEL+"-post"} and data["clean_post"]
            and all(r[2:4]==["COMPLETED","0:0"] and r[5]=="f13r1n10" for r in rows)
            and "gres/dcu=8" in by_name[LABEL][6].split(","), "ACTUAL_SLURM_TERMINAL_AND_CLEANUP")
    assigned=read(ROOT/"assigned.json",pins)
    require(assigned["step"]==by_name[LABEL][0] and len(assigned["PCI_by_index"])==8
            and sorted(assigned["PCI_by_index"])==before["pci"]==after["pci"],"ORIGINAL_PHYSICAL_DEVICE_SET")
    native=read(ROOT/"native-outcome.json",pins)
    require(native["normal_native_exit"] and native["rank_record_counts"]=={"completed":8}
            and native["native_returncode"]==0,"ACTUAL_NATIVE_TWO_UPDATES")
    old=read(CONTROL/"outcome.json",pins,"control/outcome.json")
    require(old["keeper_passed"] and pins["control/outcome.json"]==prior["outcome.json"],
            "CONTROL_ACTUAL_PINNED_TERMINAL")
    metadata={}
    same_inputs=0
    totals=Counter()
    for rank in range(8):
        rel=Path("capture")/f"rank-{rank:02d}"/"receipt.json"
        earlier=read(CONTROL/rel,pins,"control/"+rel.as_posix())
        require(pins["control/"+rel.as_posix()]==prior[rel.as_posix()],"CONTROL_ACTUAL_RANK_PIN")
        original=phase_pins(earlier,rank)
        current=phase_pins(read(ROOT/rel,pins),rank)
        require(set(current)==set(original),"CAPTURE_PHASE_NAMES_CHANGED")
        policy=read(ROOT/f"lt-policy-rank-{rank:02d}.json",pins)
        records=policy["forwards"]
        updates=policy["updates"]
        require(policy["rank"]==rank and policy["vproj_input_replay_completed"] is (rank==1)
                and [v["ordinal"] for v in records]==list(range(1,9))
                and all(v["base_calls"]==252 for v in records)
                and [v["global_step"] for v in updates]==[1,2],"ACTUAL_POLICY_LIFECYCLE")
        padded=[v["padded_vproj_calls"] for v in records]
        require(all(v in (0,1) for v in padded) and padded[0]==int(rank==1)
                and (rank==1 or not any(padded)),"EXACT_RANK_SCOPE")
        for j in (0,1):
            require(updates[j]["padded_vproj_forwards"]==sum(padded[4*j:4*j+4])
                    and updates[j]["padded_vproj_recomputes"]>=0
                    and (rank==1 or updates[j]["padded_vproj_recomputes"]==0),"EXACT_RECOMPUTE_SCOPE")
        inputs=[original[f"forward-{i:02d}-input.pt"]==current[f"forward-{i:02d}-input.pt"] for i in range(1,9)]
        outputs=[original[f"forward-{i:02d}-output.pt"]==current[f"forward-{i:02d}-output.pt"] for i in range(1,9)]
        same_inputs+=sum(inputs)
        phase_equal={k:original[k]==current[k] for k in sorted(current) if not k.startswith("forward-")}
        totals.update(k for k,value in phase_equal.items() if value)
        metadata[str(rank)]=dict(input_pins_match=inputs,output_pins_match=outputs,padded_forward_occurrences=sum(padded),
                                 padded_recompute_occurrences=sum(v["padded_vproj_recomputes"] for v in updates),
                                 other_phase_manifest_matches=phase_equal)
    x=read(ROOT/"vproj-input-vjp-rank-01.json",pins)
    require((x["rank"],x["global_step"],x["native_microbatch"])==(1,1,1),"FIRST_ACTUAL_BACKWARD")
    replay=x["replay"]
    require((replay["site"],replay["padded_rows"],replay["weight_shape"])==(SITE,192,[1024,4096]),
            "EXACT_NATIVE_VPROJ_GEOMETRY")
    shapes={"operand":[1,186,4096],"upstream":[1,186,1024],"native_dx":[1,186,4096]}
    require(all(replay[k]["shape"]==v and replay[k]["dtype"]=="torch.bfloat16"
                and HEX.fullmatch(replay[k]["SHA256"]) for k,v in shapes.items()),"REAL_VPROJ_OPERAND_PINS")
    r=replay["rows"]
    require([v["arm"] for v in r]==["native_rows","aligned_rows","aligned_rows","native_rows"] and all(
           v["calls"]==16 and 0<v["median_vjp_event_ms"]<float("inf")
           and all(math.isfinite(v[k]) and v[k]>=0 for k in ("max_abs_vs_rounded_fp32","relative_l2_vs_rounded_fp32"))
           and all(type(v[k])==int and 0<=v[k]<=16*186*4096 for k in ("changed_from_native_padded_dx","changed_from_arm_anchor"))
           for v in r),"ACTUAL_BOUNDED_REPLAY")
    times={arm:statistics.median(v["median_vjp_event_ms"] for v in r if v["arm"]==arm)
           for arm in ("native_rows","aligned_rows")}
    data.update(complete=True, actual_GPU_step=by_name[LABEL][0], ranks=8, actual_optimizer_updates=2,
                physical_cards=8, actual_microbatches=64, same_serialized_input_manifests=same_inputs,
                serialized_other_phase_equal_rank_counts=dict(totals),manifest_comparison_by_rank=metadata,
                real_vproj_gradient=replay,diagnostic_vjp_calls=sum(v["calls"] for v in r),
                component_median_ms=times,aligned_over_unpadded_component_vjp=times["aligned_rows"]/times["native_rows"],
                a100_value_comparison=False,recovery_gate_passed=False,formal_performance=False,
                formal_acceptance=False)
    return data

try:
    observed=run(PRIOR)
    print(json.dumps(observed,allow_nan=False))
except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc:
    print(json.dumps(dict(read_error=type(exc).__name__, reason=str(exc)[:150])))
