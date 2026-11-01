"""Latest prepared GLM/Deep queue adapter; no weights or Native imports.

The complete upstream route prepares constraints once. Actual processors render
and tokenize; the retained owner loop still controls Native/reset/shutdown.
"""
import copy
import hashlib
import json
import pathlib
import threading
from collections.abc import Mapping

COMMIT = "99f5c90a40f9b2546bd0202cba3397ed1d55c5a0"
BUDGET_KEYS = ("max_tokens", "max_completion_tokens", "max_new_tokens")


def raw_budget(data):
    if not isinstance(data, Mapping):
        raise ValueError("Request must be an object")
    nested = data.get("sampling_params")
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ValueError("sampling_params must be an object")
    supplied = []
    for label, obj in (("top", data), ("sampling_params", nested)):
        for key in BUDGET_KEYS:
            if key not in obj:
                continue
            value = obj[key]
            if type(value) is not int or not 1 <= value <= 8192:
                raise ValueError("Output budget must be a JSON integer1..8192; no clamp")
            supplied.append((label + "." + key, value))
    if supplied and len({value for _, value in supplied}) != 1:
        raise ValueError("Conflicting output budget fields")
    budget = supplied[0][1] if supplied else 4096
    return budget, dict(supplied=supplied, selected=budget, coercion=False,
                       clamp=False, default_applied=not supplied)


def check_sampling(data, params):
    budget, proof = raw_budget(data)
    if type(params.max_tokens) is not int or params.max_tokens != budget:
        raise ValueError("Sampling budget must equal original integer budget")
    return budget, proof


def template_prefix_resolver(processor):
    # Derive a concrete suffix from the actual template, separately from the
    # admitted history. These CPU renders are part of admission wall time.
    probe = [{"role": "user", "content": "Prepared generation prefix audit."}]
    def resolve(data, kwargs):
        before = processor.apply_chat_template(copy.deepcopy(probe), tokenize=False,
                     add_generation_prompt=False, **copy.deepcopy(kwargs))
        after = processor.apply_chat_template(copy.deepcopy(probe), tokenize=False,
                     add_generation_prompt=True, **copy.deepcopy(kwargs))
        if not isinstance(before, str) or not isinstance(after, str) or not after.startswith(before):
            raise ValueError("Actual template generation suffix is not separable")
        suffix = after[len(before):]
        if not suffix:
            raise ValueError("Actual template generation suffix is empty")
        return suffix
    return resolve


def prepare_app(family, model_path, *, processor_path=None, port=None):
    if family not in {"GLM", "Deep"}:
        raise ValueError("Unsupported Native family")
    from infinilm.processors import AutoInfinilmProcessor
    from infinilm.llm import FinishReason
    import infinilm.server.inference_server as public_server
    from prepared_chat import make_prepared_encode_chat
    from prepared_history import make_prepared_history_prequeue
    from native_engine_bridge_r268 import NativeEngineBridge, existing_server_app
    processor_path = pathlib.Path(processor_path or model_path)
    processor = AutoInfinilmProcessor.from_pretrained(str(processor_path))
    tokenizer = processor.get_tokenizer()
    if family == "GLM":
        template = tokenizer.get_chat_template()
        if hashlib.sha256(template.encode()).hexdigest() != "3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c":
            raise ValueError("Actual GLM template differs from normalized-content contract")
    raw_encode = make_prepared_encode_chat(processor)
    def encode(messages, data, generation, kwargs):
        if "enable_thinking" in kwargs and type(kwargs["enable_thinking"]) is not bool:
            raise ValueError("enable_thinking must be boolean")
        if "thinking_mode" in kwargs and kwargs["thinking_mode"] not in {"thinking", "chat"}:
            raise ValueError("Unsupported explicit thinking mode")
        if "reasoning_effort" in kwargs and kwargs["reasoning_effort"] not in {"low", "high", "max"}:
            raise ValueError("Unsupported explicit reasoning effort")
        return raw_encode(messages, data, generation, kwargs)
    prefix_resolver = template_prefix_resolver(processor)
    opening = "<|assistant|><think>" if family == "GLM" else "<｜Assistant｜><think>"
    def normalized_content(message):
        value = message.get("content")
        # Actual pinned GLM template3740ab... calls strip on assistant content.
        return value.strip() if family == "GLM" and message.get("role") == "assistant" and isinstance(value, str) else value
    history = make_prepared_history_prequeue(encode, prefix_resolver,
                    reasoning_boundaries=(opening, "</think>"),
                    normalize_content=normalized_content)

    def prequeue(req, messages, data, generation, kwargs):
        budget, proof = check_sampling(data, type("Budget", (), {"max_tokens": req.max_new_tokens})())
        bound = history(req, messages, data, generation, kwargs)
        h = bound["history_render_audit"]
        h["official_reencode_equals_actual_prompt_and_IDs"] = h["same_prepared_encoder_reproduces_prompt_and_IDs"]
        # Old finite clients consume this alias; retain exact new source identity.
        actual_prefix = prefix_resolver(data, kwargs) if generation else None
        h["pending_thinking_prefix_open"] = bool(generation and actual_prefix.endswith("<think>") and req.prompt.endswith(actual_prefix))
        h["pending_chat_prefix_closed"] = bool(generation and actual_prefix.endswith("</think>") and req.prompt.endswith(actual_prefix))
        h["generation_thinking"] = h["pending_thinking_prefix_open"] if generation else None
        h["requested_thinking"] = kwargs.get("thinking_mode", "thinking" if kwargs.get("enable_thinking", True) else "chat")
        bound.update(raw_budget_contract=proof, diagnostic_user=data.get("user"),
                     chat_template_kwargs=copy.deepcopy(data.get("chat_template_kwargs") or {}),
                     real_cache_rows=1, real_cache_capacity=32768, no_output_clamp=True,
                     request_constraints_owned_by_upstream_prepared_route=True,
                     Native_prompt_token_ids_SHA256=h["Native_prompt_token_ids_SHA256"])
        req._admission_metadata_ready = True
        return bound

    class DeepBridge(NativeEngineBridge):
        def __init__(self, *args, **kwargs):
            self._admission_lock = threading.RLock()
            super().__init__(*args, **kwargs)
        def validate_data(self, data, params):
            check_sampling(data, params)
            return super().validate_data(data, params)
        def add_chat_request(self, **kwargs):
            with self._admission_lock:
                check_sampling(kwargs.get("request_data") or {}, kwargs["sampling_params"])
                return super().add_chat_request(**kwargs)

    if family == "GLM":
        from service_queue_bridge_r268 import ServiceQueueBridge
        bridge_class, parser, timeout = ServiceQueueBridge, "glm5", 3600.0
    else:
        bridge_class, parser, timeout = DeepBridge, "deepseek_v4", 1800.0
    generation_path = pathlib.Path(model_path) / "generation_config.json"
    generation_config = json.loads(generation_path.read_bytes())
    eos = generation_config["eos_token_id"]
    eos = [eos] if type(eos) is int else eos
    if not isinstance(eos, list) or not eos or any(type(v) is not int or v < 0 for v in eos):
        raise ValueError("Actual generation config must define valid EOS IDs")
    engine = bridge_class(tokenizer, 32768, FinishReason, encode_chat=encode,
              eos_token_ids=eos, default_max_tokens=4096, max_pending=8,
              prequeue_factory=prequeue)
    # The server uses the weight path only as an identity. The processor can use
    # the metadata-only official encoder view; Native keeps the real weight path.
    server, app = existing_server_app(engine, str(model_path), parser,
              port=port or (43843 if family == "GLM" else 43882),
              stream_timeout_s=timeout, request_timeout_s=7200.0)
    original_sampling = server._build_sampling_params
    def strict_sampling(data):
        budget, _ = raw_budget(data)
        checked = dict(data)
        checked["max_tokens"] = budget
        params = original_sampling(checked)
        check_sampling(data, params)
        return params
    server._build_sampling_params = strict_sampling
    server_path = pathlib.Path(public_server.__file__)
    return dict(engine=engine, server=server, app=app, processor=processor,
                tokenizer=tokenizer, FinishReason=FinishReason, identity={
                "upstream_commit": COMMIT, "family": family,
                "processor_path": str(processor_path), "Native_weight_path": str(model_path),
                "processor_class": type(processor).__module__ + "." + type(processor).__qualname__,
                "server_SHA256": hashlib.sha256(server_path.read_bytes()).hexdigest(),
                "context": 32768, "ordinary_output_budget": 4096, "max_output_budget": 8192,
                "validation_timeouts_s": engine.validation_timeouts_s, "eos_token_ids": eos,
                "generation_config_SHA256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
                "Native_loaded_by_factory": False, "weights_opened_by_factory": False})
