"""Execute the first-chunk experiment through a fully loaded TP16 owner."""

import ctypes as C
from hashlib import sha256
import os
import re
import struct
from time import monotonic_ns

from bindings import (LOAD_KEYS, MODEL, NATIVE_SHA, PROMPT_SHA, VOCAB, canonical_hash,
                      load_checkpoint_function, load_input, model_configuration)


FLOOR = 8 * 1024**3
LAYERS = 78
CHUNKS = ((0, 512), (512, 512), (1024, 288))
COUNTERS = ("attempted", "packed", "rotary_layout", "tensor_layout", "address_span")
ARMS = ("OFF", "ON")
PLAN = [
    dict(arm=arm, past=past, tokens=count, opt_in=(arm == "ON" and past == 0))
    for arm in ARMS for past, count in CHUNKS
]
REQUIRED_ENV = {
    "GLM_GLOBAL_TP16_R470": "1",
    "GLM_GLOBAL_TP16_MODEL_R470": "1",
    "GLM_V_ON_DEMAND_R403": "1",
    "GLM_BOUNDED_ROPE_R4279": "1",
    "GLM_SHARED_CONTROL_R4279": "1",
    "GLM_VOCAB_PARALLEL_EMBEDDING_R471": "1",
}
OFF_SWITCHES = ("GLM_HISTORY_VALUE_ONLY_V1", "GLM_SELECTED_VALUE_PACKED_LAYER0", "INFINILM_GLM474_FRESH_KV_PACK",
                "GLM_C1_LAYER_OBSERVER_R474")
PCI_PATTERN = r"[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def check_environment(environment):
    need(all(environment.get(name) == value for name, value in REQUIRED_ENV.items())
         and all(environment.get(name) in (None, "", "0") for name in OFF_SWITCHES),
         "exact TP16/NoV/RoPE/embedding entry or other caller opt-in differs")


def ready_geometry(ctx, rank):
    ctx.owned()
    report = ctx.geometry()
    expected = dict(TP=16, PP=1, rows=1, capacity_per_row=32768, world_rank=rank, tp_rank=rank,
                    root_output_rank=0, local_attention_heads=4, local_index_heads=2, loaded_parameters=LOAD_KEYS,
                    Model_READY=True, weights_loaded=True, Model_constructed=True, cache_allocated=True,
                    poisoned=False, replicated_embedding_BF16_bytes=0, embedding_BF16_local_bytes=118947840,
                    embedding_vocab_begin=rank*9680, embedding_vocab_end=(rank+1)*9680)
    need(all(type(report.get(key)) is type(value) and report[key] == value for key, value in expected.items()),
         "actual fully loaded TP16 C1/32K owner differs")
    return report


def load_owner(native, *, node_index, device, world_uid, tp_uid, torch, safe_open, loader_path,
               memory, deadline, mark, gate, load_report):
    check_environment(os.environ)
    rank = node_index * 8 + device
    need(native.SHA256 == NATIVE_SHA and native.verified_cabi_count == 14, "actual .1071 CABI build differs")
    need(torch.cuda.device_count() == 8 and torch.cuda.current_device() == device,
         "one original local owner per assigned device is required")
    cfg = model_configuration(MODEL)
    checkpoint = load_checkpoint_function(loader_path)
    deadline("before construct")
    memory_sample(memory, "before construct")
    load_report["context_create_attempts"] += 1
    ctx = native.binding.GlobalTP16Context(native.library, cfg, node_index, device, world_uid, tp_uid)
    load_report["context_creations"] += 1
    created = ctx.geometry()
    need(created["Model_constructed"] is True and created["cache_allocated"] is True
         and created["weights_loaded"] is False and created["Model_READY"] is False,
         "initial resident construction differs")
    memory_sample(memory, "after original full constructor")
    gate("prefill_pair_created")
    load_report["checkpoint_load_attempts"] += 1
    checkpoint(ctx, model_path=MODEL, torch=torch, safe_open=safe_open, report=load_report, deadline=deadline,
               memory=lambda label: memory_sample(memory, label), mark=mark, progress=lambda *args, **kwargs: None)
    ready_geometry(ctx, rank)
    need(load_report["loaded_parameters"] == load_report["checkpoint_key_count"] == LOAD_KEYS
         and load_report["all78layers_loaded"] is True,
         "original full-checkpoint denominator differs")
    load_report["checkpoint_loads"] += 1
    return ctx


def memory_sample(memory, label):
    free, capacity = memory(label)
    need(type(free) is int and type(capacity) is int and FLOOR <= free <= capacity,
         "original per-owner 8GiB floor failed")
    return {"free": free, "total": capacity}


def driver_summary(read):
    summary = read()
    need(type(summary["samples"]) is int and summary["samples"] > 0
         and type(summary["min_observed"]) is int and summary["min_observed"] >= FLOOR and summary["errors"] == [],
         "continuous original driver sampling or 8GiB floor failed")
    return summary


class DeviceIO:
    """Keep complete IDs, positions, logits, and a real producer stream resident."""

    def __init__(self, torch, device, token_ids):
        self.torch = torch
        self.device = device
        need(torch.cuda.current_device() == device and len(token_ids) == 1312, "original device or full prompt differs")
        self.stream = torch.cuda.Stream(device=device)
        self.pointer = int(self.stream.cuda_stream)
        need(self.pointer > 0, "explicit caller requires a real non-default producer stream handle")
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            placement = torch.device("cuda", device)
            self.full_ids = torch.tensor(token_ids, dtype=torch.int64, device=placement)
            self.full_positions = torch.arange(len(token_ids), dtype=torch.int64, device=placement)
            self.logits = torch.empty((1, VOCAB), dtype=torch.bfloat16, device=placement)
            self.slice_views = {
                (past, count): (self.full_ids.narrow(0, past, count), self.full_positions.narrow(0, past, count))
                for past,count in CHUNKS
            }
        self.stream.synchronize()

    def physical_pci(self):
        return physical_pci(self.device)

    def verify_inputs(self, token_ids):
        self.stream.synchronize()
        need(tuple(self.full_ids.detach().cpu().tolist()) == tuple(token_ids)
             and self.full_positions.detach().cpu().tolist() == list(range(len(token_ids))),
             "device ID or position payload changed")

    def pointers(self, past, count):
        ids, pos = self.slice_views[(past, count)]
        need(ids.is_contiguous() and pos.is_contiguous() and ids.numel() == pos.numel() == count
             and ids.dtype == self.torch.int64 and pos.dtype == self.torch.int64
             and tuple(self.logits.shape) == (1,VOCAB) and self.logits.dtype == self.torch.bfloat16,
             "resident fragment pointers or logits shape differ")
        return int(ids.data_ptr()), int(pos.data_ptr()), int(self.logits.data_ptr()), self.pointer

    def poison_root_logits(self):
        with self.torch.cuda.stream(self.stream):
            self.logits.fill_(float("nan"))

    def read_root_logits(self):
        return self.logits.detach().cpu().view(self.torch.uint8).numpy().tobytes()


def physical_pci(device):
    lib = C.CDLL("/opt/dtk/cuda/cuda/lib64/libcudart.so.12")
    get_bus = lib.cudaDeviceGetPCIBusId
    get_bus.argtypes, get_bus.restype = [C.c_char_p, C.c_int, C.c_int], C.c_int
    raw = C.create_string_buffer(64)
    need(get_bus(raw, len(raw), device) == 0, "original physical PCI lookup failed")
    pci = raw.value.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", pci):
        pci = "0000:" + pci
    need(re.fullmatch(PCI_PATTERN, pci), "unexpected physical PCI text")
    return pci


def checked_logits(payload):
    need(type(payload) is bytes and len(payload) == VOCAB * 2
         and all((word & 0x7F80) != 0x7F80 for (word,) in struct.iter_unpack("<H", payload)),
         "rank0 did not return a complete finite BF16 vocab row")
    return sha256(payload).hexdigest()


def counts(value):
    need(type(value) is dict and set(value) == set(COUNTERS)
         and all(type(value[name]) is int and 0 <= value[name] <= LAYERS for name in COUNTERS)
         and value["attempted"] == LAYERS and sum(value[name] for name in COUNTERS[1:]) == LAYERS,
         "first prefill layer attempts or fallback denominator differs")
    return {name: value[name] for name in COUNTERS}


def run_loaded(ctx, io, token_ids, load_report, *, rank, native_sha, expected_pci, gate, memory, deadline, emit,
               clock=monotonic_ns):
    """Run both complete prompts without constructing, loading, or closing the context."""
    ready_geometry(ctx, rank)
    handle = ctx.ctx
    need(handle and ctx.past == 0 and ctx.rank == rank and ctx.ready is True and native_sha == NATIVE_SHA
         and len(token_ids) == 1312 and canonical_hash(token_ids) == PROMPT_SHA
         and load_report["context_creations"] == load_report["checkpoint_loads"] == 1
         and load_report["loaded_parameters"] == load_report["checkpoint_key_count"] == LOAD_KEYS
         and load_report["all78layers_loaded"] is True,
         "expected one fresh complete loaded context and the genuine prompt")
    pins = {past: io.pointers(past, count) for past,count in CHUNKS}
    need(all(len(p) == 4 and all(type(addr) is int and addr > 0 for addr in p) for p in pins.values())
         and len({(p[2],p[3]) for p in pins.values()}) == 1
         and all(p[0] == pins[0][0]+8*past and p[1] == pins[0][1]+8*past for past,p in pins.items()),
         "same contiguous resident IDs/positions/logits/producer stream required")
    pci = io.physical_pci()
    need(re.fullmatch(PCI_PATTERN, expected_pci) and pci == expected_pci, "owner real PCI differs from fresh admission")
    result = progress_base(rank, expected_pci)
    result.update(state="STARTED", context_creations=1, checkpoint_loads=1, parameters=LOAD_KEYS, layers=LAYERS,
                  Native_SHA256=native_sha, PCI=pci, full_input_SHA256=PROMPT_SHA)
    emit(result)
    gold = {}
    try:
        io.verify_inputs(token_ids)
        result["device_input_audits"] = 1
        gate("prefill_pair_loaded")
        for arm in ARMS:
            deadline("before reset " + arm)
            ctx.reset()
            result["reset_calls"] += 1
            need(ctx.ctx == handle and ctx.past == 0, "reset did not preserve original loaded context")
            before = memory_sample(memory, "after reset " + arm)
            gate("prefill_pair_ready_" + arm.lower())
            report = dict(arm=arm, memory_before=before, fragments=[])
            result["arms"].append(report)
            arm_started = clock()
            for past, n in CHUNKS:
                need(ctx.ctx == handle and ctx.past == past and io.pointers(past,n) == pins[past],
                     "owner context, resident tensor pointer, or causal cursor changed")
                opt_in = arm == "ON" and past == 0
                deadline("before forward")
                if rank == 0:
                    io.poison_root_logits()
                attempt = dict(past=past, tokens=n, opt_in=opt_in, IDs_SHA256=canonical_hash(token_ids[past:past+n]),
                               state="ENTERED")
                report["fragments"].append(attempt)
                result["attempted_forwards"] += 1
                emit(result)
                begin = clock()
                try:
                    budget = ctx.forward(n, past, *pins[past], phase="prefill", prefill512=opt_in)
                except Exception:
                    if opt_in and hasattr(ctx, "last_prefill512"):
                        attempt["observed_route_counts"] = ctx.last_prefill512
                    raise
                end = clock()
                result["successful_forwards"] += 1
                attempt["state"] = "RETURNED"
                if opt_in:
                    attempt["observed_route_counts"] = budget
                need(ctx.ctx == handle and ctx.past == past+n and type(end) is int and end > begin,
                     "actual Native completion or causal cursor differs")
                attempt.update(state="RETURNED", forward_ns=end-begin, memory_after=memory_sample(memory,"after forward"))
                if opt_in:
                    attempt["route_counts"] = counts(budget)
                else:
                    need(budget is None, "ordinary prefill unexpectedly reported trial route counters")
                if rank == 0:
                    actual = io.read_root_logits()
                    attempt["root_BF16_logits_SHA256"] = checked_logits(actual)
                    if arm == "OFF":
                        gold[past] = actual
                    else:
                        attempt["bitwise_equal_to_OFF"] = actual == gold[past]
                emit(result)
            report["first_forward_ns"] = report["fragments"][0]["forward_ns"]
            report["sum_three_forward_ns"] = sum(fragment["forward_ns"] for fragment in report["fragments"])
            report["host_scope_ns"] = clock() - arm_started
            report["memory_after"] = memory_sample(memory,"after whole prompt "+arm)
            need(ctx.past == 1312, "original full prompt was not consumed")
            io.verify_inputs(token_ids)
            result["device_input_audits"] += 1
            gate("prefill_pair_complete_" + arm.lower())
        deadline("before final reset")
        ctx.reset()
        result["reset_calls"] += 1
        need(ctx.ctx == handle and ctx.past == 0, "final original reset differs")
        result["final_memory"] = memory_sample(memory,"after final reset")
        gate("prefill_pair_reset_complete")
        evidence = result["arms"][1]["fragments"][0]["route_counts"]
        result.update(state="COMPLETED", all_local_layers_packed=evidence["packed"] == LAYERS,
                      root_logits_equal=all(item["bitwise_equal_to_OFF"] for item in result["arms"][1]["fragments"]) if rank==0 else None)
        emit(result)
        return result
    except Exception as error:
        # The keeper must stop the owned step after lost collective progress.
        # Calling reset or destroy here could re-enter a partially failed collective.
        result.update(state="FAILED", error_type=type(error).__name__, error=str(error)[:1024],
                      direct_reset_on_error=False, direct_close_on_error=False)
        emit(result)
        raise


def progress_base(rank, expected_pci):
    return dict(rank=rank, expected_PCI=expected_pci, state="BOUND", plans=PLAN, successful_forwards=0,
                attempted_forwards=0, reset_calls=0, arms=[], normal_close=False, Model_GO=False,
                model_speed_win=False)


def run_owner(*, node_index, device, expected_pci, world_uid, tp_uid, torch, safe_open, native,
              input_binding, loader_path, gate, memory, memory_summary, deadline, emit, mark, clock=monotonic_ns):
    """Normal entry for an admitted callable within the bounded same-thread owner."""
    rank = node_index * 8 + device
    progress = progress_base(rank, expected_pci)
    load = dict(context_create_attempts=0, context_creations=0, checkpoint_load_attempts=0, checkpoint_loads=0)

    def publish(report):
        nonlocal progress
        progress = report
        progress["load_progress"] = load
        emit(progress)

    publish(progress)
    try:
        deadline("bind original full input and actual .1071")
        token_ids = load_input(input_binding)
        need(re.fullmatch(PCI_PATTERN, expected_pci) and physical_pci(device) == expected_pci,
             "pre-create physical PCI differs from fresh admission")
        ctx = load_owner(native, node_index=node_index, device=device, world_uid=world_uid, tp_uid=tp_uid,
                         torch=torch, safe_open=safe_open, loader_path=loader_path, memory=memory,
                         deadline=deadline, mark=mark, gate=gate, load_report=load)
        deadline("allocate exact full resident input")
        io = DeviceIO(torch, device, token_ids)
        report = run_loaded(ctx, io, token_ids, load, rank=rank, native_sha=native.SHA256, expected_pci=expected_pci,
                            gate=gate, memory=memory, deadline=deadline, emit=publish, clock=clock)
        deadline("before synchronized normal close")
        ctx.close()
        need(ctx.ctx is None and ctx.ready is False, "typed close did not clear the owned context")
        report.update(normal_close=True, memory_after_close=memory_sample(memory, "after original normal close"))
        report["driver_memory"] = driver_summary(memory_summary)
        gate("prefill_pair_closed")
        report["state"] = "CLOSED"
        publish(report)
        return report
    except Exception as error:
        progress.update(state="FAILED", error_type=type(error).__name__, error=str(error)[:1024])
        publish(progress)
        raise
