"""Bounded, metadata-only callable and exception witness for generation."""

import ast
from collections import Counter, deque
from contextlib import ExitStack, contextmanager
import dis
from functools import wraps
from hashlib import sha256
import inspect
import os
from pathlib import Path
import re


KNOWN_MESSAGE_SHA = "36c357cc64e91c70ef5ed3a22e2348665c772c896c4ce9fd8e48c2a0568ddba7"
KEYS = frozenset((
    "input_ids", "inputs", "inputs_embeds", "attention_mask", "pixel_values", "image_sizes",
    "image_grid_thw", "pixel_attention_mask", "past_key_values", "cache_position", "position_ids",
    "decoder_input_ids", "decoder_attention_mask", "cache", "use_cache", "generation_config",
    "logits_processor", "stopping_criteria", "synced_gpus", "streamer", "assistant_model",
    "negative_prompt_ids", "negative_prompt_attention_mask", "adapter_names", "return_dict",
    "num_logits_to_keep", "logits_to_keep", "outputs",
))
LABELS = frozenset(("template.generate", "model.generate", "model.prepare_inputs_for_generation",
                    "model.forward", "template.get_generate_ids"))
MAX_EVENTS = 20
MAX_SOURCE = 4 << 20
ABSENT = object()
SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.<>]{0,119}")
LIBRARIES = ("swift", "torch", "transformers", "transformers_modules", "peft", "vlm")


class PreForwardSentinel(RuntimeError):
    """Stop the CPU witness at the first real forward boundary."""


def project_name(name):
    return name if type(name) is str and SAFE_NAME.fullmatch(name) else "redacted"


def library(name, path=None):
    if type(name) is str and any(name == lib or name.startswith(lib + ".") for lib in LIBRARIES):
        return project_name(name)
    if path and Path(path).name in {"native.py", "cpu_probe.py", "generation_boundary.py"}:
        return "candidate"
    return None


def type_name(value):
    klass = type(value)
    module = klass.__module__
    if module == "builtins" or library(module):
        return project_name(f"{module}.{klass.__qualname__}")
    return "other_type"


def fields(values):
    return {key: {"present": key in values, "type": type_name(values[key]) if key in values else None,
                  "is_none": values[key] is None if key in values else None}
            for key in sorted(KEYS)}


def source_bytes(path):
    try:
        canonical = Path(path).resolve(strict=True)
        with canonical.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not 0 < before.st_size <= MAX_SOURCE:
                return None
            raw = stream.read(MAX_SOURCE + 1)
            after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                 after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return None
        return raw
    except (OSError, TypeError, ValueError):
        return None


def callable_source(callable_):
    resolved = inspect.unwrap(callable_)
    try:
        path = inspect.getsourcefile(resolved)
    except (TypeError, OSError):
        path = None
    module = library(getattr(resolved, "__module__", None), path)
    code = getattr(resolved, "__code__", None)
    raw = source_bytes(path) if module is not None and path is not None else None
    return dict(module=module, qualname=project_name(getattr(resolved, "__qualname__", None)),
                line=code.co_firstlineno if code is not None else None,
                file_SHA256=sha256(raw).hexdigest() if raw is not None else None)


def none_operand(tb):
    if not any(op.offset == tb.tb_lasti and op.opname == "BINARY_SUBSCR"
               for op in dis.get_instructions(tb.tb_frame.f_code)):
        return None
    raw = source_bytes(tb.tb_frame.f_code.co_filename)
    if raw is None:
        return None
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return None
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)
             and node.lineno <= tb.tb_lineno <= node.end_lineno]
    if len(nodes) == 1 and isinstance(nodes[0].value, ast.Name):
        name = nodes[0].value.id
        if tb.tb_frame.f_locals.get(name, ABSENT) is None:
            return dict(local=project_name(name), is_none=True, basis="unique_direct_local_binary_subscript")
    return None


def error_evidence(error):
    chain = []
    seen = set()
    while error is not None and len(chain) < 3 and id(error) not in seen:
        seen.add(id(error))
        message_sha = sha256(str(error).encode()).hexdigest()
        frames, final = deque(maxlen=8), None
        tb = error.__traceback__
        for _ in range(128):
            if tb is None:
                break
            code = tb.tb_frame.f_code
            module = library(tb.tb_frame.f_globals.get("__name__"), code.co_filename)
            if module is not None:
                frames.append(dict(module=module, function=project_name(code.co_name), line=tb.tb_lineno))
                final = tb
            tb = tb.tb_next
        known = type(error) is TypeError and message_sha == KNOWN_MESSAGE_SHA
        row = dict(type=type_name(error), message_SHA256=message_sha, original_5207_message_match=known,
                   frames=list(frames), traceback_truncated=tb is not None)
        if final is not None:
            local = final.tb_frame.f_locals
            row["deepest_nullable"] = fields(local)
            row["deepest_keyword_containers"] = {key: fields(local[key]) for key in
                    ("kwargs", "model_kwargs", "model_inputs") if type(local.get(key)) is dict}
            row["verified_none_subscript"] = none_operand(final) if known and final.tb_next is None else None
        chain.append(row)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return chain


class BoundaryTrace:
    def __init__(self, *, stop_before_forward=False):
        self.stop_before_forward = stop_before_forward
        self.events = []
        self.counts = Counter()
        self.providers = {}
        self.error = None
        self.pre_forward_reached = False
        self.real_forward_calls = 0
        self.truncated_events = 0

    def project(self):
        return dict(events=self.events, counts=dict(sorted(self.counts.items())), providers=self.providers,
                    first_error=self.error, pre_forward_reached=self.pre_forward_reached,
                    real_forward_calls=self.real_forward_calls, truncated_events=self.truncated_events)

    def emit(self, label, phase, **extra):
        self.counts[f"{label}:{phase}"] += 1
        row = dict(label=label, phase=phase, **extra)
        if len(self.events) < MAX_EVENTS:
            self.events.append(row)
        else:
            self.truncated_events += 1

    def wrapper(self, label, bound):
        if label not in LABELS or not callable(bound):
            raise ValueError("FIXED_BOUNDARY_CALLABLE_REQUIRED")
        self.providers[label] = callable_source(bound)

        @wraps(bound)
        def observed(*args, **kwargs):
            first_forward = label == "model.forward" and not self.pre_forward_reached
            if label == "model.forward" and not first_forward and not self.stop_before_forward:
                self.real_forward_calls += 1
                return bound(*args, **kwargs)
            self.emit(label, "entry", positional_count=len(args), positional_types=[type_name(x) for x in args[:8]],
                      nullable=fields(kwargs), unlisted_keyword_count=len(set(kwargs) - KEYS))
            if first_forward:
                self.pre_forward_reached = True
            try:
                if label == "model.forward":
                    if self.stop_before_forward:
                        self.emit(label, "sentinel")
                        raise PreForwardSentinel("CPU_PRE_FORWARD_SENTINEL")
                    self.real_forward_calls += 1
                value = bound(*args, **kwargs)
            except Exception as error:
                self.emit(label, "exception", type=type_name(error))
                if self.error is None and not isinstance(error, PreForwardSentinel):
                    self.error = error_evidence(error)
                raise
            self.emit(label, "return", type=type_name(value))
            return value

        return observed

    @contextmanager
    def attach(self, obj, method, label):
        old = vars(obj).get(method, ABSENT)
        bound = getattr(obj, method)
        wrapped = self.wrapper(label, bound)
        setattr(obj, method, wrapped)
        try:
            yield
        finally:
            if old is ABSENT:
                delattr(obj, method)
            else:
                setattr(obj, method, old)

    @contextmanager
    def attach_model(self, model):
        with ExitStack() as stack:
            for name in ("generate", "prepare_inputs_for_generation", "forward"):
                stack.enter_context(self.attach(model, name, "model." + name))
            yield

    @contextmanager
    def attach_template(self, template):
        with ExitStack() as stack:
            for name in ("generate", "get_generate_ids"):
                stack.enter_context(self.attach(template, name, "template." + name))
            yield
