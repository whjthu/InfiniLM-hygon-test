"""Per-rank GLM C1/NoV context lifetime, without frontend or global readiness.

The original topology/create, constructor census/provider check, OwnedContext
and r421 complete raw loader are reused as ordinary static imports.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading

from checkpoint_loader import load_checkpoint
from model_admission_r404 import constructor_snapshot, actual_providers, NATIVE_SHA
from owned_context_r404 import OwnedContext
from topology_r404 import create_context, validate_layout


class ResidentContext:
    """Borrow ``handle.ctx`` only inside the entering owner thread's ``with``.

    All communicator peers must be supplied/coordinated by the caller. This
    primitive does not implement all-rank admission, ModelReady, HTTP or joins.
    """
    def __init__(self, binding, *, model_path, layout, rank, stage, comm_ids,
                 environ, hostname, torch, safe_open, deadline_check,
                 phase_marker, lifecycle_path, constructor_record_path,
                 report, provider_reader=actual_providers, maps_text=None):
        self.binding = binding
        self.model_path = Path(model_path)
        self.layout, self.rank, self.stage = dict(layout), rank, stage
        self.comm_ids = comm_ids
        self.environ, self.hostname = dict(environ), hostname
        self.torch, self.safe_open = torch, safe_open
        self.deadline_check, self.phase_marker = deadline_check, phase_marker
        self.lifecycle_path = Path(lifecycle_path)
        self.constructor_record_path = Path(constructor_record_path)
        self.report, self.provider_reader, self.maps_text = report, provider_reader, maps_text
        self._ctx = None
        self._owned = None
        self._thread = None
        self._entered = self._closed = False

    def _check_owner(self):
        if self._thread is not None and threading.get_ident() != self._thread:
            raise RuntimeError("GLM resident context belongs to another owner thread")

    @property
    def ctx(self):
        self._check_owner()
        if not self._entered or self._closed:
            raise RuntimeError("GLM resident context is not borrowed inside its active scope")
        return self._ctx

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("GLM resident context cannot be entered/replayed twice")
        self._thread = threading.get_ident()
        raw = (self.model_path / "config.json").read_bytes()
        config = json.loads(raw)
        actual = validate_layout(self.environ, config, self.hostname)
        # Native construction reads the actual process environment too. A
        # caller's validated dictionary cannot stand in for different getenv
        # values observed inside the CAPI.
        assert validate_layout(os.environ, config, self.hostname) == actual
        assert all(self.layout[k] == v for k, v in actual.items())
        assert self.layout["config_SHA256"] == hashlib.sha256(raw).hexdigest()
        assert type(self.rank) is int and 0 <= self.rank < 8 and self.stage == actual["stage"]
        # Original helper records these process identities. Verify before create
        # rather than mutating the process environment behind its owner.
        for key in ["GLM_RUN_R5040", "SLURM_JOB_ID", "SLURM_STEP_ID",
                    "GLM_SOURCE_MANIFEST_SHA", "GLM_V_ON_DEMAND_R403"]:
            assert os.environ[key] == self.environ[key]
        for p in [self.lifecycle_path, self.constructor_record_path]:
            assert p.parent.is_dir() and not p.exists(), str(p)
        assert self.torch.cuda.current_device() == self.rank
        self.deadline_check("before rank initialization", 120)
        lib, check = self.binding.lib, self.binding.check
        # Reuse the exact original provider reader before invoking create too;
        # constructor_snapshot repeats it after create as in the original owner.
        provider = self.provider_reader(lib)
        assert provider["library_SHA256"] == NATIVE_SHA
        self.report["preconstructor_Native_binding"] = provider
        self.phase_marker("create.enter", self.rank)
        self._ctx = create_context(lib, self.rank, self.stage, 1, 32768,
                                   config, self.comm_ids, environ=self.environ,
                                   hostname=self.hostname)
        if not self._ctx:
            raise RuntimeError(lib.modern_glm_model_error().decode())
        self._owned = OwnedContext(lib, self._ctx, check, self.stage, self.rank,
                                   path=self.lifecycle_path)
        try:
            self._owned.__enter__()  # this original helper cleans its own enter failure
        except BaseException:
            self._closed = True
            raise
        try:
            self.phase_marker("create.return", self.rank)
            lib.modern_glm_partition_report_r3854.argtypes = [ctypes.c_void_p]
            lib.modern_glm_partition_report_r3854.restype = ctypes.c_char_p
            part = json.loads(lib.modern_glm_partition_report_r3854(self._ctx))
            assert [part["begin"], part["end"]] == actual["bounds"][self.stage:self.stage + 2]
            assert part["PP"] == actual["PP"]
            self.report["actual_partition"] = part
            keys = set(json.loads(lib.modern_glm_model_keys(self._ctx)))
            layers = sorted({int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")})
            assert layers == list(range(part["begin"], part["end"]))
            # Original legal constructor trim, not a different cache allocation.
            lib.modern_glm_unloaded_trim_r4052.argtypes = [ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
            trim = (ctypes.c_uint64 * 6)()
            check(lib.modern_glm_unloaded_trim_r4052(self._ctx, trim, 6))
            assert list(trim)[3:] == [self.rank, self.stage, actual["PP"]]
            self.torch.cuda.synchronize(self.rank)
            self.torch.cuda.empty_cache()
            assert set(json.loads(lib.modern_glm_model_keys(self._ctx))) == keys
            self.report["idle_constructor_trim_r5394"] = list(trim)
            constructor_snapshot(lib, self._ctx, self.stage, self.rank, self.report,
                report_path=self.constructor_record_path, provider_reader=self.provider_reader)
            self.phase_marker("load.enter", self.rank)
            load_checkpoint(self.binding, self._ctx, model_path=self.model_path,
                torch=self.torch, safe_open=self.safe_open, stage=self.stage, rank=self.rank,
                report=self.report, deadline_check=self.deadline_check,
                phase_marker=self.phase_marker, maps_text=self.maps_text)
            self.phase_marker("load.return", self.rank)
            assert set(json.loads(lib.modern_glm_model_keys(self._ctx))) == keys
            assert self.report["loaded_parameters"] == len(keys)
            self.report.update(actual_layer_indices=layers,
                all_local_Native_keys_loaded=True,
                non_layer_Native_keys=sorted(k for k in keys if not k.startswith("model.layers.")),
                local_constructor_load_finish_completed=True, global_Model_READY=False)
            self._entered = True
            return self
        except BaseException as error:
            self._close(type(error), error, error.__traceback__)
            raise

    def _close(self, kind, error, tb):
        self._check_owner()
        if self._closed:
            raise RuntimeError("GLM resident context already closed")
        reset_error = None
        try:
            self.binding.check(self.binding.lib.modern_glm_model_reset(self._ctx))
            self._owned.q["resident_reset_RC"] = 0
        except BaseException as failure:
            reset_error = failure
            self._owned.q["resident_reset_error"] = repr(failure)
        # The original helper always attempts destroy then shutdown, preserving
        # the primary and refusing duplicate calls. Reset failure cannot skip it.
        effective = error if error is not None else reset_error
        try:
            self._owned.__exit__(type(effective) if effective is not None else kind,
                                 effective, effective.__traceback__ if effective is not None else tb)
        finally:
            self._closed = True
            self._ctx = None
        if error is None and reset_error is not None:
            raise reset_error
        return False

    def __exit__(self, kind, error, tb):
        # A foreign thread cannot consume cleanup; the real owner can still exit.
        self._check_owner()
        return self._close(kind, error, tb)
