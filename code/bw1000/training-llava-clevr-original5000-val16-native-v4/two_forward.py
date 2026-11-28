"""Observe actual OneVision prefill and first greedy decode without emitting text."""

from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import json


class TwoForwardComplete(RuntimeError):
    """The real decode returned and its witness has been captured."""


def need(condition, code):
    if not condition:
        raise ValueError(code)


def seq_length(cache, dynamic_cache, torch):
    need(type(cache) is dynamic_cache, "EXACT_NATIVE_DYNAMIC_CACHE")
    value = cache.get_seq_length()
    if torch.is_tensor(value):
        need(value.ndim == 0 and not value.is_floating_point(), "EXACT_NATIVE_CACHE_LENGTH_SCALAR")
        value = value.item()
    need(type(value) is int and value >= 0, "EXACT_NATIVE_CACHE_LENGTH")
    return value


def query_width(options):
    ids, embeds = options.get("input_ids"), options.get("inputs_embeds")
    need((ids is None) != (embeds is None), "EXACT_ONE_EFFECTIVE_QUERY_TENSOR")
    effective = ids if ids is not None else embeds
    need(effective.ndim == (2 if ids is not None else 3) and effective.shape[0] == 1
         and effective.shape[1] > 0, "EXACT_BS1_QUERY_WIDTH")
    return effective.shape[1], ids


class TwoForwards:
    def __init__(self, torch, dynamic_cache):
        self.torch, self.dynamic_cache = torch, dynamic_cache
        self.records = []
        self.eos = None
        self.chosen = None
        self.entered = 0
        self.complete = False

    def around_generate(self, template_generate):
        @wraps(template_generate)
        def observe(*args, **kwargs):
            cfg = kwargs["generation_config"]
            eos = cfg.eos_token_id
            self.eos = set(eos if isinstance(eos, list) else [eos])
            need(not cfg.do_sample and cfg.num_beams == 1 and cfg.max_new_tokens == 128
                 and self.eos and all(type(value) is int and value >= 0 for value in self.eos),
                 "EXACT_REAL_GREEDY_AND_EOS_POLICY")
            return template_generate(*args, **kwargs)
        return observe

    def around_forward(self, bound):
        signature = inspect.signature(bound)

        @wraps(bound)
        def observe(*args, **kwargs):
            pos = len(self.records)
            need(pos < 2 and self.eos is not None, "EXACT_TWO_NATIVE_FORWARDS")
            values = signature.bind(*args, **kwargs).arguments
            options = dict(values.get("kwargs", {}), **{k: v for k, v in values.items() if k != "kwargs"})
            q, ids = query_width(options)
            cache = options.get("past_key_values")
            before = seq_length(cache, self.dynamic_cache, self.torch)
            visual = options.get("pixel_values") is not None
            need(options.get("pixel_values_videos") is None and ((pos == 0 and before == 0 and q > 1 and visual)
                 or (pos == 1 and before == self.records[0]["cache_after"] and q == 1 and not visual)),
                 "EXACT_NATIVE_VISUAL_PREFILL_AND_ONE_TOKEN_DECODE")
            if pos == 1:
                need(ids is not None and int(ids[0, 0].item()) == self.chosen
                     and self.chosen not in self.eos, "ACTUAL_FIRST_DECODE_GREEDY_TOKEN")
            self.entered += 1
            output = bound(*args, **kwargs)
            after = seq_length(output.past_key_values, self.dynamic_cache, self.torch)
            need(after == before + q, "ACTUAL_DYNAMIC_CACHE_ADVANCE")
            logits = output.logits
            need(logits.ndim == 3 and logits.device.type == "cpu" and logits.shape[0] == 1 and logits.shape[1] in (1, q)
                 and logits.shape[-1] > 1 and bool(self.torch.isfinite(logits).all()), "FINITE_ACTUAL_NATIVE_LOGITS")
            token = int(logits[0, -1].argmax().item())
            self.records.append(dict(phase=("prefill" if pos == 0 else "decode"), query_tokens=q,
                cache_before=before, cache_after=after, visual_present=visual, finite_logits=True,
                forwarded_cache_position_is_none=options.get("cache_position") is None,
                logits_shape=list(logits.shape), logits_dtype=str(logits.dtype),
                greedy_token_SHA256=token_pin(token), greedy_token_is_eos=token in self.eos,
                actual_decode_matches_prefill_greedy=True if pos else None))
            if pos == 0:
                self.chosen = token
                return output
            self.complete = True
            raise TwoForwardComplete("PLANNED_TWO_FORWARD_WITNESS_RETURNED")

        return observe

    def early_eos(self, choice):
        return (len(self.records) == 1 and self.chosen in self.eos and choice.finish_reason in ("stop", "eos")
                and type(choice.token_ids) is list and choice.token_ids == [self.chosen])

    @contextmanager
    def install(self, template, model):
        absent = object()
        old_generate = vars(template).get("generate", absent)
        old_forward = vars(model).get("forward", absent)
        template.generate = self.around_generate(template.generate)
        model.forward = self.around_forward(model.forward)
        try:
            yield
        finally:
            if old_generate is absent:
                del template.generate
            else:
                template.generate = old_generate
            if old_forward is absent:
                del model.forward
            else:
                model.forward = old_forward


def token_pin(token):
    return sha256(json.dumps(token, separators=(",", ":")).encode()).hexdigest()
