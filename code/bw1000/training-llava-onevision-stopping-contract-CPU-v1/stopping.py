"""Observe the actual Swift-to-Transformers stopping contract before any model forward."""

from contextlib import contextmanager
from copy import copy
from functools import wraps
from hashlib import sha256
import inspect
import json
from pathlib import Path

CAP = 128
HF_STOPPING = "e8b5f2cef067dde6b1202a2f77e09495cde63828dbce038ce171a5fb4620ea88"
HF_GENERATION = "dde2df36821c0d724b5af47cb0ff71c3c4c1990c86d81b821911127ae4dc1254"


class StopBeforeForward(BaseException):
    """Private sentinel which cannot be mistaken for a generation response."""


def need(ok, code):
    if not ok:
        raise ValueError(code)


def hash_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    need(len(raw) <= 64 << 10, "BOUNDED_STOPPING_METADATA")
    return sha256(raw).hexdigest()


def cpu_ids(value):
    if value is None:
        return ()
    if hasattr(value, "detach"):
        need(value.device.type == "cpu" and value.ndim <= 1 and 0 < value.numel() <= 64
             and not value.is_floating_point(), "BOUNDED_CPU_EOS_TENSOR")
        value = value.detach().reshape(-1).tolist()
    if type(value) is int:
        value = [value]
    need(type(value) in (list, tuple) and len(value) <= 64
         and all(type(item) is int and item >= 0 for item in value), "BOUNDED_CPU_EOS_IDS")
    return tuple(value)


def id_info(value):
    ids = cpu_ids(value)
    return dict(count=len(ids), unique=len(set(ids)), SHA256=hash_json(ids),
                members_SHA256=hash_json(sorted(set(ids))), present=value is not None)


def source(callable_, pin=None):
    fn = inspect.unwrap(callable_)
    path = Path(inspect.getsourcefile(fn)).resolve(strict=True)
    with path.open("rb") as stream:
        data = stream.read((4 << 20) + 1)
    digest = sha256(data).hexdigest()
    need(0 < len(data) <= 4 << 20 and (pin is None or digest == pin), "EXACT_STOPPING_PROVIDER_SOURCE")
    return path, dict(bytes=len(data), SHA256=digest, basename=path.name, qualname=fn.__qualname__,
                      line=inspect.getsourcelines(fn)[1])


def stop_info(value):
    if value is None:
        return dict(present=False)
    need(type(value) in (str, list, tuple), "BOUNDED_NATIVE_STOP_WORDS")
    return dict(present=True, items=len(value) if type(value) in (list, tuple) else 1, SHA256=hash_json(value))


def config_info(value):
    fields = ("eos_token_id", "bos_token_id", "pad_token_id", "forced_eos_token_id")
    result = {name: id_info(getattr(value, name, None)) for name in fields}
    result["stopping"] = stop_info(getattr(value, "stop_strings", None))
    for name in ("max_length", "max_new_tokens", "min_length", "min_new_tokens", "num_beams", "do_sample"):
        current = getattr(value, name, None)
        need(type(current) in (int, bool, type(None)), "BOUNDED_NATIVE_LENGTH_POLICY")
        result[name] = current
    return result


@contextmanager
def replace(obj, name, replacement):
    owned = name in vars(obj)
    value = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, value)
        else:
            delattr(obj, name)


def truth(result):
    need(result.device.type == "cpu" and result.numel() == 1 and str(result.dtype) == "torch.bool",
         "EXACT_SINGLE_BOOLEAN_STOPPING_RESULT")
    return bool(result.item())


def exercise(criteria, eos, length, prompt_length, tokenizer, torch, prompt_ids=None):
    eos_ids = cpu_ids(eos.eos_token_id) if eos is not None else ()
    special = set(tokenizer.all_special_ids) | set(eos_ids)
    ordinary = next((token for token in range(min(4096, len(tokenizer))) if token not in special), None)
    need(ordinary is not None and length == prompt_length + CAP and 0 < prompt_length <= 4096,
         "FINITE_ACTUAL_PROMPT_WIDTH_AND_CONTROL_TOKEN")
    tests = {}
    for name, width, token in (("eos", prompt_length + 1, eos_ids[0] if eos_ids else None),
                               ("ordinary", prompt_length + 1, ordinary), ("cap", length, ordinary)):
        if token is None:
            tests[name] = {"covered": False, "reason": "NO_ACTUAL_EOS_CRITERION"}
            continue
        ids = torch.full((1, width), ordinary, dtype=torch.long, device="cpu")
        if prompt_ids is not None:
            ids[:, :prompt_length] = prompt_ids
        ids[0, -1] = token
        actual = [truth(copy(criterion)(ids, None)) for criterion in criteria]
        combined = truth(type(criteria)(copy(criterion) for criterion in criteria)(ids, None))
        tests[name] = dict(covered=True, sequence_length=width, individual=actual, combined=combined,
                           list_matches_individual_OR=combined == any(actual))
    return tests


def observe(model, template, engine, cfg, run, torch, facts, swift_paths):
    from transformers.generation.utils import GenerationMixin
    from transformers.generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList

    generation = model._get_stopping_criteria
    need(generation.__func__ is GenerationMixin._get_stopping_criteria, "EXACT_NATIVE_HF_CRITERIA_OWNER")
    generation_path, generation_proof = source(generation, HF_GENERATION)
    need(all(generation.__func__.__globals__.get(value.__name__) is value
             for value in (EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList)), "EXACT_PROVIDER_CRITERIA_CLASSES")
    for value in (EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList):
        need(source(value, HF_STOPPING)[0] == generation_path.with_name("stopping_criteria.py"), "EXACT_HF_SIBLING_CRITERIA_FILE")
    tokenizer = engine.tokenizer
    meta = template.template_meta
    need(cfg.max_tokens == CAP and cfg.temperature == 0 and cfg.num_beams == 1 and cfg.stop == [],
         "EXACT_5232_REQUEST_POLICY")
    facts["request_policy"] = dict(max_new_tokens=CAP, greedy=True, stop_count=0)
    facts["policy"] = dict(model=config_info(model.config), model_generation=config_info(model.generation_config),
                           vocabulary_EOS=id_info(tokenizer.eos_token_id), vocabulary_BOS=id_info(tokenizer.bos_token_id),
                           vocabulary_PAD=id_info(tokenizer.pad_token_id),
                           template=dict(suffix=stop_info(getattr(meta, "suffix", None)),
                                         meta_stop=stop_info(getattr(meta, "stop_words", None)),
                                         instance_stop=stop_info(getattr(template, "stop_words", None))))
    generate = template.generate
    forward = model.forward
    swift_path, swift_proof = source(generate)
    need(swift_path in swift_paths and swift_proof["SHA256"] == swift_paths[swift_path],
         "EXACT_SWIFT_TEMPLATE_GENERATE_PROVIDER")
    facts["providers"] = dict(hf_get_stopping_criteria=generation_proof, swift_template_generate=swift_proof,
                              hf_eos=source(EosTokenCriteria, HF_STOPPING)[1],
                              hf_max_length=source(MaxLengthCriteria, HF_STOPPING)[1],
                              first_forward=source(forward)[1])
    actual = {}

    @wraps(generate)
    def observed_generate(*args, **kwargs):
        need("swift" not in facts["policy"], "EXACT_ONE_SWIFT_GENERATE_ENTRY")
        effective = kwargs["generation_config"]
        facts["policy"]["swift"] = config_info(effective)
        need(effective.max_new_tokens == CAP and effective.do_sample is False and effective.num_beams == 1,
             "ACTUAL_SWIFT_GREEDY_128")
        return generate(*args, **kwargs)

    @wraps(generation)
    def observed_criteria(*args, **kwargs):
        bound = inspect.signature(generation).bind(*args, **kwargs)
        effective = bound.arguments["generation_config"]
        need(not actual and "swift" in facts["policy"], "EXACT_ONE_HF_STOPPING_LIST_BUILD")
        criteria = generation(*args, **kwargs)
        need(type(criteria) is StoppingCriteriaList and 1 <= len(criteria) <= 16, "BOUNDED_ACTUAL_HF_STOPPING_LIST")
        limits = [item for item in criteria if type(item) is MaxLengthCriteria]
        eos = [item for item in criteria if type(item) is EosTokenCriteria]
        need(len(limits) == 1 and len(eos) <= 1, "EXACT_HF_EOS_AND_MAX_LENGTH_CARDINALITY")
        actual.update(criteria=criteria, eos=eos[0] if eos else None, length=limits[0].max_length)
        internal = getattr(effective, "_eos_token_tensor", None)
        facts["policy"]["HF"] = config_info(effective)
        facts["policy"]["HF"]["special_EOS_tensor"] = id_info(internal)
        facts["criteria"] = dict(
            providers=[source(type(item))[1] for item in criteria],
            eos_ids=id_info(eos[0].eos_token_id if eos else None),
            eos_index=criteria.index(eos[0]) if eos else None,
            max_index=criteria.index(limits[0]), max_length=limits[0].max_length,
            max_position_embeddings=getattr(limits[0], "max_position_embeddings", None))

        def same_members(left, right):
            return (left["members_SHA256"] == right["members_SHA256"]
                    if left["present"] and right["present"] else None)

        facts["EOS_agreement"] = dict(
            tokenizer_vs_swift=same_members(facts["policy"]["vocabulary_EOS"], facts["policy"]["swift"]["eos_token_id"]),
            model_generation_vs_swift=same_members(facts["policy"]["model_generation"]["eos_token_id"],
                                                     facts["policy"]["swift"]["eos_token_id"]),
            swift_vs_HF=same_members(facts["policy"]["swift"]["eos_token_id"], facts["policy"]["HF"]["eos_token_id"]),
            HF_tensor_vs_criterion=same_members(facts["policy"]["HF"]["special_EOS_tensor"], facts["criteria"]["eos_ids"]))
        return criteria

    @wraps(forward)
    def stop(*args, **kwargs):
        facts["first_forward_guard_hits"] += 1
        need(facts["first_forward_guard_hits"] == 1 and actual,
             "EXACT_GUARDED_FIRST_FORWARD_AFTER_CRITERIA")
        values = inspect.signature(forward).bind(*args, **kwargs).arguments
        effective = dict(values.get("kwargs", {}), **{key: value for key, value in values.items() if key != "kwargs"})
        input_ids, embeds = effective.get("input_ids"), effective.get("inputs_embeds")
        need((input_ids is None) != (embeds is None), "EXACT_ONE_CPU_PREFILL_OPERAND")
        value = input_ids if input_ids is not None else embeds
        need(value.device.type == "cpu" and value.shape[0] == 1 and value.ndim == (2 if input_ids is not None else 3),
             "EXACT_PREFILL_CPU_BS1_GEOMETRY")
        facts["prompt_width"] = value.shape[1]
        facts["criterion_implied_prompt_width"] = actual["length"] - CAP
        facts["max_length_matches_prefill_plus_128"] = actual["length"] == facts["prompt_width"] + CAP
        reusable_prefix = input_ids if facts["max_length_matches_prefill_plus_128"] else None
        facts["controlled_prefix"] = ("actual_native_exposed_prefill" if reusable_prefix is not None
                                      else "synthetic_prompt_of_criterion_implied_width")
        facts["controlled_suffix"] = exercise(actual["criteria"], actual["eos"], actual["length"],
                                               facts["criterion_implied_prompt_width"], tokenizer, torch, reusable_prefix)
        raise StopBeforeForward

    with replace(template, "generate", observed_generate), replace(model, "_get_stopping_criteria", observed_criteria), \
            replace(model, "forward", stop):
        try:
            run()
        except StopBeforeForward:
            facts["pre_forward_stop"] = True
        else:
            raise ValueError("MODEL_INVOCATION_RETURNED_WITHOUT_FIRST_FORWARD_SENTINEL")
