"""Consume the actual static loader with checkpoint-header metadata.

Only Torch/safetensors/Native leaves are recording substitutes. No tensor
payload is loaded, no CAPI/device is run, and no model qualification is claimed.
"""
import ctypes
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from checkpoint_loader import Binding, declare, load_checkpoint


class Function:
    def __init__(self, callback=lambda *a: 0): self.callback = callback
    def __call__(self, *args): return self.callback(*args)


def run():
    metadata = json.loads((Path(__file__).parent / "checkpoint-metadata.json").read_text())
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdk = root / "install/hygon-glm-5f91ad7-r3672/lib/libinfiniops.so"
        sdk.parent.mkdir(parents=True)
        sdk.write_bytes(b"recording SDK file, never loaded")
        records = metadata["tensor_records"]
        dtype = {"F32": "float32", "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn"}
        storage = []
        tensors = {}
        events = []

        class Tensor:
            def __init__(self, record, override=None):
                self.shape = record["shape"]
                self.dtype = override or dtype[record["dtype"]]
                self.buffer = ctypes.create_string_buffer(8)
                storage.append(self.buffer)
            def contiguous(self):
                events.append("contiguous")
                return self
            def view(self, target):
                assert self.dtype == "float8_e4m3fn" and target == "uint8"
                events.append("FP8_raw_uint8_view")
                self.dtype = target
                return self
            def data_ptr(self): return ctypes.addressof(self.buffer)
            def numel(self): return math.prod(self.shape)
            def element_size(self):
                return {"float32": 4, "bfloat16": 2, "uint8": 1}[self.dtype]

        torch = SimpleNamespace(float32="float32", bfloat16="bfloat16", uint8="uint8",
                                float8_e4m3fn="float8_e4m3fn",
                                cuda=SimpleNamespace(mem_get_info=lambda: (5, 9)))

        class Shard:
            def __enter__(self): events.append("open_enter"); return self
            def get_tensor(self, name): events.append(("get", name)); return tensors[name]
            def __exit__(self, *exc): events.append("open_exit")

        def safe_open(path, **kw):
            assert path.name in {r["filename"] for r in records}
            assert kw == dict(framework="pt", device="cpu")
            events.append("safe_open")
            return Shard()

        def fresh():
            events.clear()
            tensors.clear()
            tensors.update({r["name"]: Tensor(r) for r in records})
            mapping = {r["name"]: r["filename"] for r in records}
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
            lib = SimpleNamespace()
            for name in ["error", "id_size", "id", "create", "keys", "load_batch",
                         "finish", "forward", "control", "reset", "destroy"]:
                setattr(lib, "modern_glm_model_" + name, Function())
            lib.modern_glm_model_error.callback = lambda: b"recorded Native error"
            lib.modern_glm_model_keys.callback = lambda ctx: json.dumps(list(mapping)).encode()
            def load(ctx, payload):
                events.append(("load", json.loads(payload)))
                return 0
            lib.modern_glm_model_load_batch.callback = load
            lib.modern_glm_model_finish.callback = lambda ctx: events.append("finish") or 0
            return declare(lib)

        def consume(binding, report, deadline=None):
            def check(name):
                events.append(("deadline", name))
                if deadline == name: raise TimeoutError(name)
            return load_checkpoint(binding, 123, model_path=root, torch=torch,
                safe_open=safe_open, stage=0, rank=0, report=report,
                deadline_check=check,
                phase_marker=lambda phase, rank: events.append(("phase", phase, rank)),
                maps_text="0 0 0 " + sdk.as_posix(), progress=lambda *a, **k: None)

        binding = fresh()
        assert binding.lib.modern_glm_model_create.restype is ctypes.c_void_p
        assert len(binding.lib.modern_glm_model_create.argtypes) == 8
        assert binding.lib.modern_glm_model_forward.argtypes[-2:] == [ctypes.c_int, ctypes.c_void_p]
        report = {}
        assert consume(binding, report) is report
        actual = next(e[1] for e in events if isinstance(e, tuple) and e[0] == "load")
        assert [r["name"] for r in actual] == sorted(tensors)
        assert {r["dtype"] for r in actual} == {0, 1, 2}
        assert all(r["pointer"] > 0 and r["shape"] == tensors[r["name"]].shape for r in actual)
        assert "FP8_raw_uint8_view" in events
        expected_bytes = sum((r["data_offsets"][1] - r["data_offsets"][0]) for r in records)
        assert report["loaded_parameters"] == 3 and report["source_parameter_bytes"] == expected_bytes
        assert events.index(("deadline", "safe_open")) < events.index("safe_open")
        assert events.index(("deadline", "modern_glm_model_finish")) < events.index("finish")
        assert events.index("finish") < events.index(("phase", "finish.return", 0))
        results.append(dict(case="actual_three_dtype_header_metadata_complete_static_load", passed=True,
                            source_bytes_metadata_only=expected_bytes))

        binding = fresh()
        binding.lib.modern_glm_model_keys.callback = lambda ctx: b'["missing.actual.Native.key"]'
        try: consume(binding, {})
        except AssertionError: pass
        else: raise AssertionError("missing key accepted")
        assert not events
        results.append(dict(case="missing_key_before_any_shard_or_commit", passed=True))

        for stop in ["safe_open", "modern_glm_model_load_batch", "modern_glm_model_finish"]:
            binding = fresh()
            try: consume(binding, {}, stop)
            except TimeoutError as e: assert str(e) == stop
            else: raise AssertionError("deadline ignored")
            assert "finish" not in events
            if stop == "modern_glm_model_load_batch": assert events[-1] == "open_exit"
            results.append(dict(case="original_deadline_before_" + stop, passed=True))

        binding = fresh()
        binding.lib.modern_glm_model_load_batch.callback = lambda *a: 1
        try: consume(binding, {})
        except RuntimeError as e: assert str(e) == "recorded Native error"
        else: raise AssertionError("load failure accepted")
        assert events[-1] == "open_exit" and "finish" not in events
        results.append(dict(case="Native_batch_error_propagates_closes_shard_no_finish", passed=True))

        binding = fresh()
        binding.lib.modern_glm_model_finish.callback = lambda *a: 1
        report = {}
        try: consume(binding, report)
        except RuntimeError: pass
        else: raise AssertionError("finish failure accepted")
        assert "loaded_parameters" not in report
        assert ("phase", "finish.return", 0) not in events
        results.append(dict(case="finish_error_no_false_ready_report", passed=True))

        binding = fresh()
        tensors[records[0]["name"]].dtype = "float16"
        try: consume(binding, {})
        except AssertionError: pass
        else: raise AssertionError("unsupported dtype accepted")
        assert "finish" not in events and not any(isinstance(e, tuple) and e[0] == "load" for e in events)
        results.append(dict(case="unsupported_dtype_refused", passed=True))

    return dict(passed=True, cases=results, GPU=False, Native_dlopen=False,
        Model_READY=False, checkpoint_payload_read=False,
        actual_static_load_function_consumed=True,
        recording_leaves=["Torch tensor storage/value", "safetensors payload get_tensor",
                          "Native keys/load_batch/finish and other CAPI functions", "SDK maps/file"],
        actual_metadata=[r["name"] for r in records],
        actual_all_Native_keys_or_device_weightload_qualified=False)


if __name__ == "__main__":
    result = run()
    (Path(__file__).parent / "local-CPU-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
