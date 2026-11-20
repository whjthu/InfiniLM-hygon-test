"""Replay the candidate leaf policies during the original two Swift optimizer updates."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from replay_state import digest, rng, restore_rng
from text_base_preference import ACTIVE_SITE, TextBasePreference, value_lt


KEY = "qwen_native_rank1_vproj_lt_first_two"
VISION_ORDER = ("cublaslt",)
PREF = VISION_ORDER[0]
PERFORMANCE_PLUGIN_SHA256 = "d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_archive(rank):
    folder = Path("/replay-capture") / f"rank-{rank:02d}"
    name = "forward-01-input.pt"
    expected = json.loads(Path(os.environ["QWEN_REPEAT_PINS"]).read_bytes())["ranks"][str(rank)]["files"][name]
    path = folder / name
    raw = path.read_bytes()
    need(len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["SHA256"], "FIRST_RNG_ARCHIVE")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


class VisionFC2Preference:
    """Set one preference inside the synchronous FC2 dispatch and restore it on exit."""

    def __init__(self, vision):
        self.arm = None
        self.calls = []
        self.originals = []
        self.set_or_get = torch.backends.cuda.preferred_blas_library
        self.expected = torch.backends.cuda._BlasBackends
        self.inherited = self.set_or_get()
        need(len(vision.blocks) == 27, 'FULL_VISION_FC2_COUNT')
        for layer in vision.blocks:
            fc2 = layer.mlp.linear_fc2
            need(type(fc2) is torch.nn.Linear and fc2.bias is not None
                 and list(fc2.weight.shape) == [1152, 4304]
                 and fc2.weight.dtype == torch.bfloat16, 'ACTUAL_VISION_FC2_TYPE')
        for index, layer in enumerate(vision.blocks):
            fc2 = layer.mlp.linear_fc2
            original = fc2.forward
            self.originals.append((fc2, original))
            fc2.forward = self.wrap(index, original)

    def wrap(self, index, original):
        def call(operand):
            need(self.arm in VISION_ORDER, 'FC2_NATIVE_CALL_ORDER')
            need(operand.ndim == 2 and operand.shape[0] > 0 and operand.shape[1] == 4304
                 and operand.dtype == torch.bfloat16 and operand.device.type == 'cuda',
                 'FC2_NATIVE_BF16_OPERAND')
            previous = self.set_or_get()
            need(previous == self.inherited, 'FC2_EXTERNAL_PREFERENCE_CHANGE')
            try:
                got = self.set_or_get(self.arm)
                need(got == self.expected[self.arm] == self.set_or_get(), 'FC2_PREFERENCE_READBACK')
                out = original(operand)
            finally:
                self.set_or_get(previous)
            need(self.set_or_get() == previous, 'FC2_PREFERENCE_RESTORE')
            need(out.dtype == operand.dtype and tuple(out.shape) == (operand.shape[0],1152),
                 'FC2_NATIVE_BF16_OUTPUT')
            self.calls.append(dict(layer=index, rows=operand.shape[0], stride=list(operand.stride()),
                                   dtype=str(operand.dtype), actual_preference=str(got)))
            return out
        return call

    def close(self):
        for module, original in self.originals:
            module.forward = original


class NativeTextLtFirstTwo(TrainerCallback):
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kw):
        self.rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        self.deadline = float(os.environ["FIRST_UPDATE_DEADLINE"])
        need(0 <= self.rank < 8 and local == self.rank == torch.cuda.current_device()
             and torch.cuda.device_count() == torch.distributed.get_world_size() == 8
             and torch.version.hip and state.global_step == 0 and args.max_steps == 25, "NATIVE_ORIGINAL_TOPOLOGY")
        self.model = kw.get("model", self.trainer.model)
        core = self.model.get_base_model()
        callbacks = self.trainer.callback_handler.callbacks
        recorders = [cb for cb in callbacks if type(cb).__name__ == "NativeFirstUpdates"]
        need(len(recorders) == 1 and callbacks.index(recorders[0]) < callbacks.index(self), "NATIVE_CAPTURE_ORDER")
        performance_type = callbacks_map["nvidia_baseline"]
        matches = [cb for cb in callbacks if type(cb) is performance_type]
        need(len(matches) == 1 and performance_type.__name__ == "BaselineCallback"
             and callbacks.index(recorders[0]) < callbacks.index(matches[0]) < callbacks.index(self),
             "NATIVE_PERFORMANCE_CALLBACK_ORDER")
        self.performance_callback = matches[0]
        baseline_file = Path(inspect.getsourcefile(performance_type))
        need(hashlib.sha256(baseline_file.read_bytes()).hexdigest() == PERFORMANCE_PLUGIN_SHA256
             and "on_train_end" not in vars(self.performance_callback), "NATIVE_PERFORMANCE_CALLBACK_SOURCE")
        self.capture = recorders[0].capture
        need(not self.capture.detached and not self.capture.failed and len(self.capture.handles) == 2
             and self.capture.inputs == self.capture.outputs == self.capture.updates == 0
             and self.capture.train_end.__self__ is self.capture, "NATIVE_CAPTURE_ATTACHED")
        provider = Path(torch.backends.cuda.__file__).read_bytes()
        need(hashlib.sha256(provider).hexdigest() == "94f8fde8aa08235c9d06c0508dc68d94a004bf336a24130241df35bd28b68d6a",
             "ORIGINAL_BLAS_PROVIDER")
        self.text = TextBasePreference(core, rank=self.rank, call_limit=4096)
        need(self.text.inherited == self.text.expected["cublas"], "MATCHED_INHERITED_BACKWARD_PREFERENCE")
        self.text.arm = PREF
        self.vision = VisionFC2Preference(core.get_submodule("model.visual"))
        self.vision.arm = PREF
        self.archive = checked_archive(self.rank)
        self.forwards = []
        self.updates = []
        self.math_calls = 0
        self.capture_finalized = False
        self.performance_end_seen = False
        self.capture.train_end = self.finish_capture
        self.performance_callback.on_train_end = self.end_performance_capture
        self.route_registry = qwen.ALL_ATTENTION_FUNCTIONS
        self.original_route = self.route_registry["sdpa"]
        need(self.original_route is sdpa_attention_forward, "ORIGINAL_SDPA_PROVIDER")
        torch.use_deterministic_algorithms(True, warn_only=False)

        @functools.wraps(self.original_route)
        def route(module, query, key, value, attention_mask, *values, **options):
            if not isinstance(module, qwen.Qwen3VLTextAttention):
                return self.original_route(module, query, key, value, attention_mask, *values, **options)
            need(query.dtype == torch.bfloat16 and query.device.type == "cuda", "ORIGINAL_TEXT_QUERY")
            self.math_calls += 1
            with sdpa_kernel(SDPBackend.MATH):
                return self.original_route(module, query, key, value, attention_mask, *values, **options)

        self.route_registry.register("sdpa", route)
        need(self.route_registry["sdpa"] is route, "ACTUAL_SDPA_DISPATCH")
        self.original_forward = self.model.forward

        @functools.wraps(self.original_forward)
        def forward(*values, **options):
            ordinal = len(self.forwards) + 1
            need(time.time() < self.deadline and 1 <= ordinal <= 8
                 and state.global_step == (ordinal-1)//4 and torch.is_grad_enabled(), "ORIGINAL_FORWARD_ORDER")
            start, vision, math = len(self.text.calls), len(self.vision.calls), self.math_calls
            result = self.original_forward(*values, **options)
            sites = self.text.calls[start:]
            need(len(sites) == 252 and {call["site"] for call in sites} == self.text.names
                 and self.math_calls-math == 36 and (len(self.vision.calls)-vision) % 27 == 0,
                 "COMPLETE_BASE_AND_MATH_FORWARD")
            selected = [call for call in sites if call["site"] == ACTIVE_SITE]
            need(len(selected) == 1
                 and selected[0]["preference"] == str(self.text.expected[PREF])
                 and all(self.matched_preference(call) for call in sites),
                 "ACTUAL_FROZEN_PROJECTION_SELECTION")
            value_count = sum(c["selected_value"] for c in sites)
            need(0 <= value_count <= 1 and (self.rank != 1 or ordinal != 1 or value_count == 1),
                 "FIRST_ORIGINAL_RANK1_VALUE_SHAPE")
            self.forwards.append(dict(global_step=state.global_step, ordinal=ordinal, base_calls=len(sites),
                                      selected_calls=len(selected), value_lt_calls=value_count,
                                      vision_calls=len(self.vision.calls)-vision, math_calls=self.math_calls-math))
            return result

        def prepare_first_rng(_module, values, options):
            if self.archive is None:
                return
            archive = self.archive
            need(state.global_step == 0 and archive["global_step"] == 0 and archive["micro_ordinal"] == 1
                 and digest((values, options)) == digest((archive["args"], archive["kwargs"])), "ORIGINAL_FIRST_INPUT")
            restore_rng(archive["RNG_before_forward"])
            self.first_rng = digest(rng())
            need(self.first_rng == digest(archive["RNG_before_forward"]), "ORIGINAL_FIRST_RANK_RNG")
            self.archive = None

        self.initial_rng_handle = self.model.register_forward_pre_hook(prepare_first_rng, with_kwargs=True, prepend=True)
        self.model.forward = forward
        return control

    def matched_preference(self, call):
        selected = value_lt(self.rank, call["site"], call["input_shape"], call["input_dtype"])
        desired = self.text.expected[PREF] if call["site"] == ACTIVE_SITE or selected else self.text.inherited
        return call["selected_value"] is selected and call["preference"] == str(desired)

    def finish_capture(self):
        self.capture.check_deadline()
        need(self.trainer.state.global_step == 2 and not self.capture.failed and self.capture.detached
             and self.capture.inputs == self.capture.outputs == 8
             and self.capture.updates == self.capture.clip_calls == 2 and len(self.capture.files) == 25,
             "ORIGINAL_CAPTURE_SHORT_SCOPE")
        self.capture.write_json("first-two-scope-complete.json", dict(rank=self.rank, global_step=2))
        self.capture_finalized = True

    def end_performance_capture(self, args, state, control, **kw):
        need(state.global_step == 2 and args.max_steps == 25 and control.should_training_stop
             and self.capture_finalized and not self.performance_end_seen, "SHORT_DIAGNOSTIC_PERFORMANCE_END")
        if self.performance_callback.hook is not None:
            self.performance_callback.hook.remove()
        self.performance_end_seen = True
        return control

    def on_step_end(self, args, state, control, **kw):
        step = state.global_step
        need(time.time() < self.deadline and step in (1,2) and len(self.updates) == step-1
             and len(self.forwards) == step*4 and self.capture.updates == step, "NATIVE_REAL_UPDATE_BOUNDARY")
        base = sum(row["base_calls"] for row in self.forwards[-4:])
        vision = sum(row["vision_calls"] for row in self.forwards[-4:])
        extra = len(self.text.calls)-base
        need(base == 1008 and extra >= 0 and len(self.vision.calls) >= vision
             and self.text.preference() == self.text.inherited and self.vision.set_or_get() == self.vision.inherited,
             "LEAF_PREFERENCE_BACKWARD_BOUNDARY")
        selected = sum(call["site"] == ACTIVE_SITE for call in self.text.calls)
        need(selected >= 4 and all(self.matched_preference(call) for call in self.text.calls),
             "FROZEN_PROJECTION_CHECKPOINT_RECOMPUTE_POLICY")
        value_forwards = sum(row["value_lt_calls"] for row in self.forwards[-4:])
        value_recomputes = sum(call["selected_value"] for call in self.text.calls)-value_forwards
        need(value_recomputes >= 0, "VALUE_RECOMPUTE_COUNT")
        self.updates.append(dict(global_step=step, base_forward_calls=base, base_checkpoint_recompute_calls=extra,
                                 selected_forward_calls=4, selected_checkpoint_recompute_calls=selected-4,
                                 value_lt_forwards=value_forwards, value_lt_checkpoint_recomputes=value_recomputes,
                                 vision_forward_calls=vision, vision_checkpoint_recompute_calls=len(self.vision.calls)-vision))
        self.text.calls.clear()
        self.vision.calls.clear()
        if step == 2:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kw):
        need(state.global_step == 2 and len(self.updates) == 2 and self.performance_end_seen
             and self.capture_finalized, "NATIVE_TRAIN_END")
        del self.performance_callback.on_train_end
        self.initial_rng_handle.remove()
        self.text.close()
        self.vision.close()
        self.model.forward = self.original_forward
        self.route_registry.register("sdpa", self.original_route)
        path = Path(os.environ["QWEN_REPEAT_OUTPUT"]) / f"lt-policy-rank-{self.rank:02d}.json"
        with path.open("x", encoding="utf8") as stream:
            json.dump(dict(rank=self.rank, preference=PREF, selected_site=ACTIVE_SITE,
                           performance_report_emitted=False, shortened_performance_end=True,
                           updates=self.updates, forwards=self.forwards,
                           peft_provider_SHA256=self.text.lora_provider, first_rng_SHA256=self.first_rng, formal_training=False), stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        return control


need(KEY not in callbacks_map, "NATIVE_TEXT_LT_DUPLICATE_CALLBACK")
callbacks_map[KEY] = NativeTextLtFirstTwo
