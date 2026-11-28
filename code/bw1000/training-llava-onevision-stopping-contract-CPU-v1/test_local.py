"""Offline source derivation and EOS/ordinary/cap guard fixtures."""

import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import run_cpu
import stopping as s


CPU = SimpleNamespace(type="cpu")


class Bool:
    dtype, device = "torch.bool", CPU

    def __init__(self, value):
        self.value = bool(value)

    def numel(self):
        return 1

    def item(self):
        return self.value


class Tokens:
    ndim, device = 2, CPU

    def __init__(self, width, value=0):
        self.shape = (1, width)
        self.data = [value] * width

    def __setitem__(self, key, value):
        if isinstance(key[1], slice):
            self.data[key[1]] = value.data
        else:
            self.data[key[1]] = value


class EosTokenCriteria:
    eos_token_id = [42, 43]

    def __call__(self, ids, scores):
        return Bool(ids.data[-1] in self.eos_token_id)


class MaxLengthCriteria:
    max_length, max_position_embeddings = 130, 4096

    def __call__(self, ids, scores):
        return Bool(len(ids.data) >= self.max_length)


class StoppingCriteriaList(list):
    def __call__(self, ids, scores):
        return Bool(any(item(ids, scores).item() for item in self))


class GenerationMixin:
    def _get_stopping_criteria(self, generation_config, stopping_criteria):
        return StoppingCriteriaList([MaxLengthCriteria(), EosTokenCriteria(), *stopping_criteria])


class Model(GenerationMixin):
    config = SimpleNamespace(eos_token_id=42)
    generation_config = SimpleNamespace(eos_token_id=[42, 43])
    forward_calls = 0

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.forward_calls += 1


class Template:
    template_meta = SimpleNamespace(suffix=["fixture only"], stop_words=["fixture stop"])

    def __init__(self, model):
        self.model = model

    def generate(self, *, generation_config):
        self.model._get_stopping_criteria(generation_config, [])
        self.model.forward(input_ids=Tokens(2))


class Tokenizer:
    eos_token_id, bos_token_id, pad_token_id, all_special_ids = 42, 1, 2, [1, 2, 42, 43]

    def __len__(self):
        return 64


def fixture():
    utils = ModuleType("transformers.generation.utils")
    utils.GenerationMixin = GenerationMixin
    criteria = ModuleType("transformers.generation.stopping_criteria")
    for value in (EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList):
        setattr(criteria, value.__name__, value)
    model = Model()
    template = Template(model)
    facts = dict(first_forward_guard_hits=0, pre_forward_stop=False)
    config = SimpleNamespace(do_sample=False, num_beams=1, max_new_tokens=128, eos_token_id=[42, 43],
                             _eos_token_tensor=[42, 43])
    cfg = SimpleNamespace(max_tokens=128, temperature=0, num_beams=1, stop=[])
    engine = SimpleNamespace(tokenizer=Tokenizer())
    torch = SimpleNamespace(long="int64", full=lambda shape, value, **_: Tokens(shape[1], value))

    def source(callable_, pin=None):
        if pin == s.HF_GENERATION:
            name = "utils.py"
        elif getattr(callable_, "__name__", "") == "generate":
            name = "base.py"
        else:
            name = "stopping_criteria.py"
        return Path("/fixture") / name, {"SHA256": "fixture"}

    with (patch.dict(sys.modules, {utils.__name__: utils, criteria.__name__: criteria}),
          patch.object(s, "source", side_effect=source)):
        s.observe(model, template, engine, cfg, lambda: template.generate(generation_config=config),
                  torch, facts, {Path("/fixture/base.py"): "fixture"})
    assert model.forward_calls == 0 and facts["pre_forward_stop"] and facts["first_forward_guard_hits"] == 1
    assert "forward" not in vars(model) and "generate" not in vars(template)
    assert facts["criteria"]["eos_ids"]["count"] == 2
    assert facts["EOS_agreement"]["tokenizer_vs_swift"] is False and facts["EOS_agreement"]["HF_tensor_vs_criterion"]
    assert facts["max_length_matches_prefill_plus_128"] and facts["controlled_prefix"] == "actual_native_exposed_prefill"
    tests = facts["controlled_suffix"]
    assert tests["eos"]["individual"] == [False, True] and tests["eos"]["combined"]
    assert tests["ordinary"]["individual"] == [False, False] and not tests["ordinary"]["combined"]
    assert tests["cap"]["individual"] == [True, False] and tests["cap"]["combined"]
    missing = s.exercise(StoppingCriteriaList([MaxLengthCriteria()]), None, 130, 2, engine.tokenizer, torch)
    assert missing["eos"] == {"covered": False, "reason": "NO_ACTUAL_EOS_CRITERION"}


def request_config(source):
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "RequestConfig"]
    call, = calls
    preferences = {item.arg: ast.unparse(item.value) for item in call.keywords}
    assert preferences.pop("max_tokens") in ("native.CAP", "CAP")
    return preferences


def main():
    repo = Path(__file__).resolve().parents[3]
    code = repo / "code/bw1000"
    v4 = code / "training-llava-clevr-original5000-val16-native-v4"
    boundary = code / "training-llava-clevr-original5000-val16-generation-boundary-v1"
    for directory, pins in ((v4, run_cpu.V4), (boundary, run_cpu.BOUNDARY)):
        for key, sha in pins.items():
            assert s.sha256((directory / key).read_bytes()).hexdigest() == sha
    original = repo / "results/bw1000/2026-11-28/specialized-llava-val16-v4-5232-two-arms-all-cap-and-clean.json"
    run_cpu.verify_original(original.read_bytes())
    provider = repo / "local-only/llava-native-eos-provider-source-pins-only-24a9a75da04942d0a79885f9a7dc872a.json"
    raw = provider.read_bytes()
    assert s.sha256(raw).hexdigest() == run_cpu.PROVIDER_PROOF
    actual = json.loads(raw)
    assert actual["authority_verified"] is actual["remote_received"] is True
    pins = actual["response"]["pins"]
    assert pins["hf_generation"]["SHA256"] == s.HF_GENERATION and pins["hf_stopping"]["SHA256"] == s.HF_STOPPING
    for key, exact in (("generation_config", run_cpu.MODEL_FILES["generation_config.json"]),
                       ("model_config", run_cpu.MODEL_FILES["config.json"]),
                       ("tokenizer_config", run_cpu.MODEL_FILES["tokenizer_config.json"]),
                       ("swift_template_base", run_cpu.SWIFT_TEMPLATE["swift/template/base.py"]),
                       ("swift_llava_template", run_cpu.SWIFT_TEMPLATE["swift/template/templates/llava.py"]),
                       ("swift_qwen_template", run_cpu.SWIFT_TEMPLATE["swift/template/templates/qwen.py"])):
        assert (pins[key]["bytes"], pins[key]["SHA256"]) == exact
    derived = run_cpu.derive((boundary / "cpu_probe.py").read_bytes())
    assert request_config((v4 / "native.py").read_bytes()) == request_config(derived)
    compile(derived, "original-exposed-invocation-stop-observer", "exec")
    fixture()
    print("actual .5232 counts and exact Swift argv/source; guarded stopping/EOS/ordinary/cap fixtures PASS; no model and no initialization")


if __name__ == "__main__":
    main()
