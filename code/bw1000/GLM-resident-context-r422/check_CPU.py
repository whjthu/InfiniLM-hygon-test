"""Actual resident contextmanager + copied original helpers + r421 loader.

Native, provider/dladdr and tensor payload leaves are recording substitutes.
Lifecycle/snapshot/index files, context-manager control flow and checks are real.
"""
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

from checkpoint_loader import declare
from model_admission_r404 import NATIVE_SHA
from resident_context import ResidentContext
from topology_r404 import read_layout


class Function:
    def __init__(self, callback): self.callback = callback
    def __call__(self, *args): return self.callback(*args)


def run():
    D = Path(__file__).parent
    fixture = json.loads((D / "CPU-metadata.json").read_text())
    config = (D / "actual-config.json").read_bytes()
    assert hashlib.sha256(config).hexdigest() == fixture["config_SHA256"]
    env = dict(GLM_PP_SIZE_R3603="2", GLM_CACHE_ROWS_R3627="1", GLM_CACHE_CAP_R3627="32768",
        GLM_V_ON_DEMAND_R403="1", GLM_BOUNDED_ROPE_R4279="1", GLM_SHARED_CONTROL_R4279="1",
        GLM_SHARED_PREFIX_VALIDATE_R5814="0", GLM_ACTUAL_CAPTURE_R5164="0", USE_WEIGHT_CACHE_R3604="0",
        LD_PRELOAD="/refactor/candidates/blas-policy-r271/libblas_policy_r271.so",
        LD_LIBRARY_PATH="/install/hygon-glm-5f91ad7-r3672/lib:", GLM_STAGE_R5040="0", SLURM_PROCID="0",
        GLM_RUN_R5040="CPU-recording-only", SLURM_JOB_ID="CPU", SLURM_STEP_ID="recording",
        GLM_SOURCE_MANIFEST_SHA="CPU-recording-not-a-publication")
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    cases = []
    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_bytes(config)
            records = fixture["tensor_records"]
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {r["name"]: r["filename"] for r in records}}))
            sdk = root / "install/hygon-glm-5f91ad7-r3672/lib/libinfiniops.so"
            sdk.parent.mkdir(parents=True)
            sdk.write_bytes(b"recording only; no dlopen")
            layout = read_layout(env, model / "config.json", "a02r4n12")
            comm = {"world": base64.b64encode(b"WORLD").decode(),
                    "tp": [base64.b64encode(x).decode() for x in [b"TP0", b"TP1"]],
                    "control": base64.b64encode(b"CONTROL").decode()}
            pointer = 2 ** 40 + 123
            serial = 0

            def make(fail=None, deadline=None):
                nonlocal serial
                serial += 1
                calls = []
                def record(name, *args):
                    calls.append((name, threading.get_ident()))
                    return 1 if fail == name else 0
                lib = SimpleNamespace()
                def create(rank, stage, rows, capacity, raw_config, *buffers):
                    record("create")
                    assert (rank, stage, rows, capacity) == (0, 0, 1, 32768)
                    assert json.loads(raw_config) == json.loads(config)
                    assert [b.value for b in buffers] == [b"WORLD", b"TP0", b"CONTROL"]
                    return 0 if fail == "create" else pointer
                lib.modern_glm_model_create = Function(create)
                lib.modern_glm_model_error = Function(lambda: b"recorded Native primary")
                lib.modern_glm_model_keys = Function(lambda ctx: json.dumps([r["name"] for r in records]).encode())
                lib.modern_glm_partition_report_r3854 = Function(lambda ctx: json.dumps(
                    dict(begin=0, end=37 if fail == "partition" else 38, PP=2)).encode())
                def trim(ctx, out, count):
                    assert ctx == pointer and count == 6
                    for i, value in enumerate([0, 0, 0, 0, 0, 2]): out[i] = value
                    return record("trim")
                lib.modern_glm_unloaded_trim_r4052 = Function(trim)
                census = dict(layer_begin=0, layer_end=38, rows=1, capacity=32768, loaded=False,
                    cache_category_bytes=dict(latent=38, rotary=38, projected_key=38),
                    cache_unique_start_bytes=114,
                    cache_layers=[dict(tensors=[dict(category=k) for k in
                        ["latent", "rotary", "projected_key"]]) for _ in range(38)])
                if fail == "census": census["cache_category_bytes"]["projected_value"] = 1
                lib.modern_glm_constructor_census_r4158 = Function(lambda ctx: json.dumps(census).encode())
                def batch(ctx, raw):
                    assert ctx == pointer
                    desc = json.loads(raw)
                    assert all(x["pointer"] > 0 and x["dtype"] == 1 for x in desc)
                    return record("load")
                lib.modern_glm_model_load_batch = Function(batch)
                for operation in ["finish", "reset", "destroy", "shutdown"]:
                    setattr(lib, "modern_glm_model_" + operation,
                            Function(lambda *a, name=operation: record(name)))
                for operation in ["id_size", "id", "forward", "control"]:
                    setattr(lib, "modern_glm_model_" + operation, Function(lambda *a: 0))
                binding = declare(lib)
                buffers = []
                class Tensor:
                    def __init__(self, row):
                        self.shape, self.dtype = row["shape"], "BF16"
                        self.buf = ctypes.create_string_buffer(8)
                        buffers.append(self.buf)
                    def contiguous(self): return self
                    def data_ptr(self): return ctypes.addressof(self.buf)
                    def numel(self): return math.prod(self.shape)
                    def element_size(self): return 2
                tensors = {r["name"]: Tensor(r) for r in records}
                class Shard:
                    def __enter__(self): return self
                    def get_tensor(self, name): return tensors[name]
                    def __exit__(self, *args): return False
                def safe_open(path, **kwargs):
                    assert kwargs == dict(framework="pt", device="cpu")
                    return Shard()
                torch = SimpleNamespace(float32="F32", bfloat16="BF16", uint8="U8", float8_e4m3fn="FP8",
                    cuda=SimpleNamespace(current_device=lambda: 0,
                        synchronize=lambda rank: None, empty_cache=lambda: None, mem_get_info=lambda: (5, 9)))
                def check_deadline(name, *args):
                    if name == deadline: raise TimeoutError("original deadline: " + name)
                report = {}
                native = ResidentContext(binding, model_path=model, layout=layout, rank=0, stage=0,
                    comm_ids=comm, environ=env, hostname="a02r4n12", torch=torch, safe_open=safe_open,
                    deadline_check=check_deadline, phase_marker=lambda *a: None,
                    lifecycle_path=root / (str(serial) + "-lifecycle.json"),
                    constructor_record_path=root / (str(serial) + "-constructor.json"),
                    report=report, maps_text="0 " + sdk.as_posix(),
                    provider_reader=lambda lib: dict(library_SHA256="wrong" if fail == "provider" else NATIVE_SHA,
                        recording_provider_leaf=True, CAPI_providers="recording, not observed dladdr"))
                return native, calls, report

            def count(calls, op): return sum(name == op for name, thread in calls)
            def clean(calls):
                assert all(count(calls, op) == 1 for op in ["reset", "destroy", "shutdown"])
                assert len({thread for _, thread in calls}) == 1

            native, calls, report = make()
            with native as borrowed:
                assert borrowed is native and borrowed.ctx == pointer
                assert count(calls, "finish") == 1 and count(calls, "destroy") == 0
                assert report["loaded_parameters"] == len(records)
                assert report["global_Model_READY"] is False
            clean(calls)
            lifecycle = json.loads(native.lifecycle_path.read_text())
            assert lifecycle["normal"] and lifecycle["completed"]
            try: native.ctx
            except RuntimeError: pass
            else: raise AssertionError("stale handle returned after with")
            try: native.__enter__()
            except RuntimeError: pass
            else: raise AssertionError("duplicate constructor accepted")
            cases.append(dict(case="actual_contextmanager_load_yield_then_original_normal_cleanup", passed=True))

            for failure in ["partition", "census", "load", "finish"]:
                native, calls, report = make(fail=failure)
                try:
                    with native: raise AssertionError("must not yield")
                except (AssertionError, RuntimeError): pass
                else: raise AssertionError("entry failure accepted")
                clean(calls)
                assert not report.get("local_constructor_load_finish_completed", False)
                assert not json.loads(native.lifecycle_path.read_text())["normal"]
                cases.append(dict(case=failure + "_failure_never_yields_and_original_cleanup_once", passed=True))

            for boundary in ["before rank initialization", "safe_open", "modern_glm_model_load_batch",
                             "modern_glm_model_finish"]:
                native, calls, report = make(deadline=boundary)
                try:
                    with native: raise AssertionError("must not yield")
                except TimeoutError: pass
                else: raise AssertionError("deadline accepted")
                if boundary == "before rank initialization": assert not calls
                else: clean(calls)
                cases.append(dict(case="deadline_" + boundary, passed=True))

            for failure in ["reset", "destroy"]:
                native, calls, report = make(fail=failure)
                primary = ValueError("request primary")
                try:
                    with native: raise primary
                except ValueError as actual: assert actual is primary
                else: raise AssertionError("primary masked")
                clean(calls)
                assert not json.loads(native.lifecycle_path.read_text())["normal"]
                cases.append(dict(case=failure + "_failure_preserves_body_primary_and_attempts_both_cleanup", passed=True))

            native, calls, report = make(fail="reset")
            try:
                with native: pass
            except RuntimeError: pass
            else: raise AssertionError("normal reset failure hidden")
            clean(calls)
            cases.append(dict(case="normal_reset_failure_visible_after_destroy_shutdown", passed=True))

            native, calls, report = make()
            with native:
                errors = []
                def foreign():
                    try: native.__exit__(None, None, None)
                    except RuntimeError as e: errors.append(str(e))
                thread = threading.Thread(target=foreign)
                thread.start(); thread.join()
                assert errors and count(calls, "reset") == count(calls, "destroy") == 0
                assert native.ctx == pointer
            clean(calls)
            cases.append(dict(case="foreign_exit_refused_owner_can_still_close", passed=True))

            native, calls, report = make()
            os.environ["GLM_SHARED_CONTROL_R4279"] = "0"
            try:
                try:
                    with native: raise RuntimeError("must not yield")
                except AssertionError: pass
                else: raise AssertionError("process getenv mismatch accepted")
                assert not calls
            finally:
                os.environ["GLM_SHARED_CONTROL_R4279"] = env["GLM_SHARED_CONTROL_R4279"]
            cases.append(dict(case="actual_process_layout_getenv_mismatch_before_factory", passed=True))

            native, calls, report = make(fail="provider")
            try:
                with native: raise RuntimeError("must not yield")
            except AssertionError: pass
            else: raise AssertionError("foreign provider accepted")
            assert not calls
            cases.append(dict(case="original_Native952_provider_identity_before_factory", passed=True))

            native, calls, report = make(fail="create")
            try:
                with native: raise AssertionError("must not yield")
            except RuntimeError: pass
            assert count(calls, "create") == 1 and count(calls, "destroy") == count(calls, "shutdown") == 0
            cases.append(dict(case="null_factory_original_no_context_semantics", passed=True))
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return dict(passed=True, cases=cases, actual_contextmanager_consumed=True,
        original_OwnedContext_topology_census_consumed=True, static_r421_load_consumed=True,
        Native_dlopen=False, GPU=False, Native_calls_executed=False, checkpoint_payload_read=False,
        Model_READY=False, HTTP_started=False, jobs=0,
        recording_leaves=["Native constructor/keys/partition/trim/census/load/finish/reset/destroy/shutdown",
                          "provider dladdr/maps", "Torch device/storage/value", "safetensors tensor payload"],
        fixture_scope=fixture["scope"])


if __name__ == "__main__":
    q = run()
    (Path(__file__).parent / "local-CPU-result.json").write_text(json.dumps(q, indent=2) + "\n")
    print(json.dumps(q))
