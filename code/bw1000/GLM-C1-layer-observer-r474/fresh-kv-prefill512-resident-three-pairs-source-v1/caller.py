"""Six complete original prompts on one fully loaded TP16 resident context."""

from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import re
import sys
from time import monotonic_ns


HERE = Path(__file__).resolve().parent
BASE_SHA = "962798930aaf96ec5ad7dd0489b99819f9fd9366c348e503726cf0b508a14613"
PRIOR_RESULT_SHA = "8049e7da21d476287aaae41b752dfaf11f97363313872f0513c3a9886d082f3f"
ORDERS = (("OFF","ON"),("ON","OFF"),("OFF","ON"))
WINDOWS = ("first_forward_ns","sum_three_forward_ns","host_scope_ns")
PLAN = [{"pair":p,"position":i,"arm":arm} for p,order in enumerate(ORDERS,1) for i,arm in enumerate(order,1)]


def need(ok,message):
    if not ok:
        raise ValueError(message)


def load_base():
    base = HERE/"base"
    if not base.is_dir():
        base = HERE.parent/"fresh-kv-prefill512-resident-pair-source-v2"
    raw = (base/"SOURCE_FILES.json").read_bytes()
    need(sha256(raw).hexdigest()==BASE_SHA,"original resident caller manifest differs")
    pins = json.loads(raw)["files"]
    need(set(pins)=={"README.zh-CN.md","SCOPE.json","aggregate.py","bindings.py","check_local.py","run_pair.py"},
         "original resident caller members differ")
    for name,pin in pins.items():
        data = (base/name).read_bytes()
        need(pin == {"SHA256":sha256(data).hexdigest(),"bytes":len(data)},"original resident member differs: "+name)
    sys.path.insert(0,str(base))
    try:
        packages = [import_module(name) for name in ("bindings","run_pair")]
    finally:
        sys.path.pop(0)
    need(all(Path(mod.__file__).resolve()==(base/(mod.__name__+".py")).resolve() for mod in packages),
         "an unrelated original resident caller is already imported")
    return packages


binding, prior = load_base()
CHUNKS = prior.CHUNKS
PROVIDERS = {"recipe_SHA256":binding.RECIPE_SHA,"confirmed_providers":71}


def phase(arm, event):
    return f"prefill_pair_{arm['pair']}_position_{arm['position']}_{event}"


PHASES = ("prefill_pair_created","prefill_pair_loaded",
          *(phase(arm,event) for arm in PLAN for event in ("ready","complete")),
          "prefill_pair_reset_complete","prefill_pair_closed")


def input_hashes(ids):
    need(len(ids)==1312 and prior.canonical_hash(ids)==prior.PROMPT_SHA,
         "requires complete original ordinal 1 input")
    return {past:prior.canonical_hash(ids[past:past+count]) for past,count in CHUNKS}


def progress_base(rank, pci):
    return dict(rank=rank,expected_PCI=pci,state="BOUND",plans=PLAN,forwards_planned=18,
                attempted_forwards=0,successful_forwards=0,reset_calls=0,device_input_audits=0,
                arms=[],normal_close=False,Model_GO=False,model_speed_win=False)


def run_arm(ctx, io, handle, pins, input_sha, arm, result, first, rank, *, memory, deadline, emit, clock):
    for past,count in CHUNKS:
        need(ctx.ctx==handle and ctx.past==past and io.pointers(past,count)==pins[past],
             "original loaded context, causal positions or resident tensors changed")
        opt_in = arm["arm"]=="ON" and past==0
        deadline("before measured forward")
        if rank==0:
            io.poison_root_logits()
        row = dict(past=past,tokens=count,opt_in=opt_in,IDs_SHA256=input_sha[past],state="ENTERED")
        arm["fragments"].append(row)
        result["attempted_forwards"]+=1
        emit(result)
        start = clock()
        try:
            routes = ctx.forward(count,past,*pins[past],phase="prefill",prefill512=opt_in)
        except Exception:
            if opt_in and hasattr(ctx,"last_prefill512"):
                row["observed_route_counts"]=ctx.last_prefill512
            raise
        end = clock()
        result["successful_forwards"]+=1
        row["state"]="RETURNED"
        if opt_in:
            row["observed_route_counts"]=routes
        need(ctx.ctx==handle and ctx.past==past+count and type(start) is int and type(end) is int and end>start,
             "actual forward completion or monotonic interval differs")
        row.update(forward_ns=end-start,memory_after=prior.memory_sample(memory,"after forward"))
        if opt_in:
            row["route_counts"]=prior.counts(routes)
        else:
            need(routes is None,"nontrial fragment reported route counters")
        if rank==0:
            raw = io.read_root_logits()
            row["root_BF16_logits_SHA256"]=prior.checked_logits(raw)
            if arm["position"]==1:
                first[past]=raw
            else:
                row["bitwise_equal_to_pair_first"]=raw==first[past]
        emit(result)


def run_loaded(ctx, io, ids, load, *, rank, native_sha, expected_pci, providers, gate, memory, deadline, emit,
               clock=monotonic_ns):
    result = progress_base(rank,expected_pci)
    emit(result)
    try:
        prior.ready_geometry(ctx,rank)
        handle = ctx.ctx
        input_sha = input_hashes(ids)
        need(handle and ctx.rank==rank and ctx.past==0 and ctx.ready is True and native_sha==prior.NATIVE_SHA
             and providers==PROVIDERS and load["context_creations"]==load["checkpoint_loads"]==1
             and load["loaded_parameters"]==load["checkpoint_key_count"]==prior.LOAD_KEYS
             and load["all78layers_loaded"] is True and load["actual_layers"]==list(range(prior.LAYERS)),
             "complete loaded parent, provider or checkpoint evidence differs")
        pci = io.physical_pci()
        need(re.fullmatch(prior.PCI_PATTERN,expected_pci) and pci==expected_pci,"fresh physical device differs")
        pins = {past:io.pointers(past,count) for past,count in CHUNKS}
        need(all(len(p)==4 and all(type(a) is int and a>0 for a in p) for p in pins.values())
             and len({(p[2],p[3]) for p in pins.values()})==1
             and all(p[0]==pins[0][0]+8*past and p[1]==pins[0][1]+8*past for past,p in pins.items()),
             "unchanged contiguous inputs, output and owner producer stream required")
        result.update(state="STARTED",context_creations=1,checkpoint_loads=1,parameters=prior.LOAD_KEYS,
                      layers=prior.LAYERS,Native_SHA256=native_sha,full_input_SHA256=prior.PROMPT_SHA,
                      PCI=pci,providers=providers)
        io.verify_inputs(ids)
        result["device_input_audits"]=1
        emit(result)
        gate("prefill_pair_loaded")
        first={}
        for item in PLAN:
            if item["position"]==1:
                first={}
            deadline("before arm reset")
            ctx.reset()
            result["reset_calls"]+=1
            need(ctx.ctx==handle and ctx.past==0,"reset must preserve the same loaded context")
            facts = dict(item,memory_before=prior.memory_sample(memory,"after arm reset"),fragments=[])
            result["arms"].append(facts)
            emit(result)
            gate(phase(item,"ready"))
            started = clock()
            run_arm(ctx,io,handle,pins,input_sha,facts,result,first,rank,memory=memory,deadline=deadline,emit=emit,clock=clock)
            facts.update(first_forward_ns=facts["fragments"][0]["forward_ns"],
                         sum_three_forward_ns=sum(row["forward_ns"] for row in facts["fragments"]),
                         host_scope_ns=clock()-started,memory_after=prior.memory_sample(memory,"after full prompt"))
            need(ctx.ctx==handle and ctx.past==1312,"each arm must consume the whole original prompt")
            io.verify_inputs(ids)
            result["device_input_audits"]+=1
            emit(result)
            gate(phase(item,"complete"))
        deadline("before final reset")
        ctx.reset()
        result["reset_calls"]+=1
        need(ctx.ctx==handle and ctx.past==0,"final reset differs")
        result["final_memory"]=prior.memory_sample(memory,"after final reset")
        gate("prefill_pair_reset_complete")
        result["state"]="COMPLETED"
        emit(result)
        return result
    except Exception as error:
        result.update(state="FAILED",error_type=type(error).__name__,error=str(error)[:512],
                      direct_reset_on_error=False,direct_close_on_error=False)
        emit(result)
        raise


def run_owner(*, node_index, device, expected_pci, world_uid, tp_uid, torch, safe_open, native, providers,
              input_binding, loader_path, gate, memory, memory_summary, deadline, emit, mark, clock=monotonic_ns):
    """Call the original full checkpoint setup once and close only on completion."""
    rank=node_index*8+device
    progress=progress_base(rank,expected_pci)
    load=dict(context_create_attempts=0,context_creations=0,checkpoint_load_attempts=0,checkpoint_loads=0)

    def publish(report):
        nonlocal progress
        progress=report
        progress["load_progress"]=load
        emit(progress)

    publish(progress)
    try:
        deadline("bind original prompt and Native")
        ids = binding.load_input(input_binding)
        need(providers==PROVIDERS and re.fullmatch(prior.PCI_PATTERN,expected_pci)
             and prior.physical_pci(device)==expected_pci,"provider or physical device differs before create")
        ctx = prior.load_owner(native,node_index=node_index,device=device,world_uid=world_uid,tp_uid=tp_uid,
                               torch=torch,safe_open=safe_open,loader_path=loader_path,memory=memory,
                               deadline=deadline,mark=mark,gate=gate,load_report=load)
        deadline("allocate full resident input")
        io = prior.DeviceIO(torch,device,ids)
        report = run_loaded(ctx,io,ids,load,rank=rank,native_sha=native.SHA256,expected_pci=expected_pci,
                            providers=providers,gate=gate,memory=memory,deadline=deadline,emit=publish,clock=clock)
        deadline("before normal close")
        ctx.close()
        need(ctx.ctx is None and ctx.ready is False,"normal close did not release the typed context")
        report.update(normal_close=True,memory_after_close=prior.memory_sample(memory,"after original close"),
                      driver_memory=prior.driver_summary(memory_summary))
        gate("prefill_pair_closed")
        report["state"]="CLOSED"
        publish(report)
        return report
    except Exception as error:
        progress.update(state="FAILED",error_type=type(error).__name__,error=str(error)[:512])
        publish(progress)
        raise
