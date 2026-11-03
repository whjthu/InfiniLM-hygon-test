"""Changed-loop CPU checks: actual queue, request, policy/wire and execute_plan.

Only Native/Torch/collective and tokenizer/history leaves are recorded. No
device, HTTP, provider construction, or real multi-rank execution occurs.
"""
import ast
import ctypes
from enum import Enum
import json
import math
import os
import queue
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

from native_owner import serve_requests, run_resident_owner, NATIVE_ENVIRONMENT
from service_queue_bridge_r268 import ServiceQueueBridge
from service_policy import bind_prequeue, encode_profile_wire
from longcontext_wire_r204 import WIRE, BEGIN, STOP


class Reason(str, Enum):
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    ERROR = "error"
    EOS_TOKEN = "eos_token"
    LENGTH = "length"


class Function:
    def __init__(self, callback=lambda *a: 0): self.callback = callback
    def __call__(self, *args): return self.callback(*args)


def run():
    results = []
    before_threads = {t.ident for t in threading.enumerate()}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        serial = 0

        def fixture(*, window=4, prompt_length=1025, cancel=None, failure=None,
                    stop_immediately=False, peer=False):
            nonlocal serial
            serial += 1
            report = {"library_sha256": "9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b"}
            report_path = root / (str(serial) + "-owner.json")
            queue_path = root / (str(serial) + "-queue.json")
            profile = "glm-nov16-w" + str(window)
            prompt = [1] * prompt_length
            def prepare(req, *args):
                req._admission_metadata_ready = True
                return {"CPU_recording_encoder_history": True}
            engine = ServiceQueueBridge(object(), 32768, Reason,
                encode_chat=lambda *a: ("recording prompt", list(prompt)),
                eos_token_ids=[9], default_max_tokens=4, prequeue_factory=prepare)
            bind_prequeue(engine, profile)
            req = engine.add_chat_request(messages=[{"role": "user", "content": "recording"}],
                sampling_params=SimpleNamespace(temperature=0.0, max_tokens=4, stop=None),
                request_id="cpu-" + str(serial), request_data={"max_tokens": 4})
            if cancel == "pending": req.mark_abort(Reason.CANCELED)
            engine.stop()  # real FIFO sentinel follows the already accepted request
            tensors = {}
            class Tensor:
                def __init__(self, values):
                    self.values = list(values)
                    self.buffer = ctypes.create_string_buffer(8)
                    tensors[ctypes.addressof(self.buffer)] = self
                def data_ptr(self): return ctypes.addressof(self.buffer)
                def cpu(self): return self
                def tolist(self): return list(self.values)
                def item(self): return self.values[0]
                def numel(self): return len(self.values)
                def __getitem__(self, index): return self.values[index]
                def __setitem__(self, index, value): self.values[index] = value
                def argmax(self, axis): return Tensor([max(range(len(self.values)), key=self.values.__getitem__)])
            events = []
            forwards = []
            control_shapes = []
            actual_prompt_length = prompt_length
            selected = 2
            def get(ptr): return tensors[ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr]
            lib = SimpleNamespace()
            for name in ["modern_glm_model_reset", "modern_glm_vector_policy_r8540",
                "modern_glm_model_owned_prefix_policy_r1669", "modern_glm_model_feature_bypass_r1406",
                "modern_glm_model_indexer_policy_r998", "modern_glm_model_decode_leases_policy_r1356",
                "modern_glm_model_batch_cache_policy_r3068", "glm_reduced_output_policy_r9032",
                "glm_attention_cohort_policy_r9054", "glm_ragged_policy_r9136"]:
                setattr(lib, name, Function(lambda *a, name=name: events.append(name) or 0))
            lib.modern_glm_model_transient_report_r5688 = Function(lambda ctx: json.dumps(
                dict(rows=1, capacity=32768, enabled=False, GPU_logical_bytes=0)).encode())
            def forward(ctx, count, rows, desc, ids, positions, logits, all_positions, stream):
                nonlocal selected
                assert rows == 1 and desc[0] == 0 and desc[2] == count and all_positions == 0
                past = desc[1]
                assert count <= 512 and past + count <= 32768
                assert get(positions).values == list(range(past, past + count))
                if peer: assert get(ids).values == prompt[past:past + count] if past < prompt_length else [2]
                else: assert get(ids).values == [0] * count
                forwards.append(dict(T=count, past=past))
                if failure == "Native": return 7
                selected = 9 if past >= actual_prompt_length else 2
                output = get(logits)
                output.values = [0.0] * 10
                output.values[selected] = float("nan") if failure == "finite" else 1.0
                if cancel == "after_first" and len(forwards) == 1: req.mark_abort(Reason.CANCELED)
                return 0
            lib.modern_glm_model_forward = Function(forward)
            peer_wires = [encode_profile_wire(1, BEGIN, prompt, 4, profile), encode_profile_wire(2, STOP)]
            def control(value):
                nonlocal actual_prompt_length
                control_shapes.append(len(value.values))
                if len(value.values) == WIRE:
                    if peer: value.values = peer_wires.pop(0)
                    if value.values[2] == BEGIN: actual_prompt_length = int(value.values[3])
                elif len(value.values) == 4:
                    value.values = [16 * v for v in value.values]
                elif len(value.values) == 2:
                    if peer: value.values = [selected, 1]
                elif len(value.values) == 1:
                    value.values = [16 * value.values[0]]
                else: raise AssertionError("unexpected control geometry")
            torch = SimpleNamespace(float32="F32", int64="I64", bfloat16="BF16",
                tensor=lambda values, **kw: Tensor(values), ones=lambda n, **kw: Tensor([1] * n),
                zeros=lambda n, **kw: Tensor([0] * n),
                arange=lambda begin, end, **kw: Tensor(range(begin, end)),
                full=lambda shape, value, **kw: Tensor([value] * math.prod(shape)),
                isfinite=lambda tensor: SimpleNamespace(all=lambda: all(math.isfinite(x) for x in tensor.values)),
                cuda=SimpleNamespace(synchronize=lambda rank: None,
                    mem_get_info=lambda rank: (1 if failure == "floor" else 2 * 1024 ** 3, 64 * 1024 ** 3)))
            def check(rc):
                if rc: raise RuntimeError("recording Native failure " + str(rc))
            args = dict(lib=lib, ctx=123, stage=0 if peer else 1, rank=0, control=control,
                check=check, ptr=lambda tensor: ctypes.c_void_p(tensor.data_ptr()), stream=lambda: None,
                vocab=10, torch=torch, report=report, report_path=report_path, queue_path=queue_path,
                engine=None if peer else engine, FinishReason=Reason,
                prepared_identity={"upstream_commit": "99f5c90a40f9b2546bd0202cba3397ed1d55c5a0", "eos_token_ids": [9]},
                eos_token_ids=[9], run_identity={"run": "CPU", "actual_step": "CPU.recording",
                    "parent_source_manifest_SHA256": "CPU-no-publication"}, default_profile=profile,
                profile_manifest_SHA256="CPU-policy-recording", deadline=float("inf"),
                should_stop=lambda: stop_immediately)
            return args, req, forwards, events, control_shapes

        for window in [2, 4]:
            args, req, forwards, events, controls = fixture(window=window)
            serve_requests(**args)
            assert req.is_finished() and req.finish_reason == Reason.EOS_TOKEN
            assert req.generated_token_ids == [2, 9]
            assert forwards == [dict(T=512, past=0), dict(T=512, past=512),
                                dict(T=1, past=1024), dict(T=1, past=1025)]
            saved = json.loads(args["report_path"].read_text())
            assert saved["attempted_Native"] == saved["completed_Native"] == 1
            assert saved["native_loop_drained"] and saved["normal_reset_close"]
            assert saved["normal_shutdown"] is False  # no context destroy in the borrowed loop
            assert saved["service_requests"][0]["actual_generated_tokens"] == 2
            assert saved["service_requests"][0]["generated_natural_EOS"]
            assert json.loads(args["queue_path"].read_text())["requests"][0]["finished"]
            results.append(dict(case="actual_queue_W" + str(window) + "_full_prompt_decode_EOS_STOP_drain", passed=True))

        for window in [2, 4]:
            args, req, forwards, events, controls = fixture(window=window, prompt_length=2049, cancel="after_first")
            serve_requests(**args)
            assert len(forwards) == window and req.generated_token_ids == []
            assert req.is_finished() and req.finish_reason == Reason.CANCELED
            assert args["report"]["service_requests"][0]["original_reset_all16_completed"]
            results.append(dict(case="cancel_after_W" + str(window) + "_whole_group_no_sample", passed=True))

        for style in ["pending", "STOP_pending"]:
            args, req, forwards, events, controls = fixture(cancel="pending" if style == "pending" else None,
                stop_immediately=style == "STOP_pending")
            serve_requests(**args)
            assert not forwards and req.is_finished() and req.finish_reason == Reason.CANCELED
            assert args["report"]["completed_Native"] == 0
            results.append(dict(case=style + "_canceled_only_after_global_drain", passed=True))

        args, req, forwards, events, controls = fixture()
        take = args["engine"].take
        calls = [0]
        def transient_empty(*a, **kw):
            calls[0] += 1
            if calls[0] == 1: raise queue.Empty()
            return take(*a, **kw)
        args["engine"].take = transient_empty
        serve_requests(**args)
        assert controls.count(WIRE) == 3 and args["report"]["completed_Native"] == 1
        results.append(dict(case="WAIT_preserves_sequence_and_does_not_admit_request", passed=True))

        args, req, forwards, events, controls = fixture(prompt_length=2049)
        args["deadline"] = 10
        args["clock"] = lambda: 20 if forwards else 0
        serve_requests(**args)
        assert len(forwards) == 4 and req.finish_reason == Reason.TIMEOUT
        assert args["report"]["native_loop_drained"]
        results.append(dict(case="original_deadline_votes_after_group_then_normal_reset_STOP", passed=True))

        for failure in ["Native", "finite", "floor"]:
            args, req, forwards, events, controls = fixture(failure=failure)
            try: serve_requests(**args)
            except (RuntimeError, AssertionError): pass
            else: raise AssertionError("failure accepted")
            assert len(forwards) == 1 and req.is_finished() and req.finish_reason == Reason.ERROR
            assert events.count("modern_glm_model_reset") == 2  # setup + before request, no failing-rank drain/reset
            saved = json.loads(args["report_path"].read_text())
            assert saved["completed_Native"] == 0 and not saved["normal_reset_close"]
            assert saved["active_request_calls"][0]["CAPI_returned"]
            if failure == "Native": assert saved["active_request_calls"][0]["CAPI_RC"] == 7
            if failure == "finite": assert saved["active_request_calls"][0]["finite"] is False
            assert "service_error" in saved
            results.append(dict(case=failure + "_failstop_preserves_first_call_no_next_request_or_WORLD_drain", passed=True))

        args, req, forwards, events, controls = fixture(failure="Native")
        original_write = Path.write_text
        def failed_error_record(path, *a, **kw):
            if path == args["report_path"].with_suffix(".tmp") and forwards:
                raise OSError("recording report filesystem failure")
            return original_write(path, *a, **kw)
        Path.write_text = failed_error_record
        try:
            try: serve_requests(**args)
            except RuntimeError as primary:
                assert str(primary) == "recording Native failure 7"
                assert any("persistence failed" in note for note in primary.__notes__)
            else: raise AssertionError("Native primary lost")
        finally:
            Path.write_text = original_write
        assert len(forwards) == 1 and "report_write_error" in args["report"]
        results.append(dict(case="report_write_failure_does_not_mask_Native_primary", passed=True))

        args, req, forwards, events, controls = fixture(peer=True)
        serve_requests(**args)
        assert len(forwards) == 4 and args["report"]["completed_Native"] == 1
        results.append(dict(case="nonemitter_uses_received_typed_wire_no_queue_no_HTTP", passed=True))

        # Exercise actual composition without redoing r422's already-qualified
        # 16 local cases: only the constructor leaf is recorded here.
        args, req, forwards, events, controls = fixture()
        class RecordedResident:
            report = args["report"]
            lifecycle_path = root / "recorded-context-lifecycle.json"
            binding = SimpleNamespace(lib=args["lib"], check=args["check"], ptr=args["ptr"])
            ctx = 123
            def __enter__(self): events.append("recording_constructor_enter"); return self
            def __exit__(self, kind, error, tb):
                events.append("recording_constructor_exit")
                self.lifecycle_path.write_text(json.dumps(dict(normal=error is None,
                    CPU_recording_context_leaf=True)))
                return False
        service = {k:v for k,v in args.items() if k not in
            {"lib", "ctx", "check", "ptr", "report", "torch", "stream", "control"}}
        saved_env = {k:os.environ.get(k) for k in NATIVE_ENVIRONMENT}
        try:
            run_resident_owner(RecordedResident(), torch=args["torch"], stream=args["stream"],
                environ=os.environ, control=args["control"], **service)
        finally:
            for k,v in saved_env.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        assert events[-1] == "recording_constructor_exit"
        assert json.loads(args["report_path"].read_text())["normal_shutdown"]
        results.append(dict(case="actual_owner_composition_context_exit_before_shutdown_flag", passed=True,
                            constructor_leaf="recording; actual r422 cases not repeated"))

    source = (Path(__file__).parent / "native_owner.py").read_text()
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ["exec", "eval"] for n in ast.walk(tree))
    assert not any(word in source for word in ["uvicorn", "threading", "server.start", "server.run", ".join(", "/refactor"])
    assert before_threads == {t.ident for t in threading.enumerate()}
    return dict(passed=True, cases=results, actual_new_loop=True,
        actual_existing_queue_request_policy_wire_execute_plan=True, GPU=False,
        Native_executed=False, HTTP_imported_or_started=False, new_jobs=0, Model_READY=False,
        recording_leaves=["Native/Torch/device storage", "collective peers", "official tokenizer/history",
                          "constructor leaf only in composition case; r422 not requalified"],
        all_16_owner_execution=False)


if __name__ == "__main__":
    q=run()
    (Path(__file__).parent/"local-CPU-result.json").write_text(json.dumps(q,indent=2)+"\n")
    print(json.dumps(q))
