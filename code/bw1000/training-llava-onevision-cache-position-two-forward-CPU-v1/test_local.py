"""Test exact readback method derivation, gate branches, and two-return observer."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import repair
from run_cpu import SOURCE_AUDIT, derive
from two_forward import TwoForwardComplete, TwoForwards, token_pin


class Cache:
    is_compileable = False

    def __init__(self, length=0):
        self.length = length

    def get_seq_length(self):
        return self.length


class Ids:
    ndim = 2

    def __init__(self, width, token=7):
        self.shape = (1, width)
        self.token = token

    def __getitem__(self, key):
        assert key == (0, 0)
        return SimpleNamespace(item=lambda: self.token)


class Logits:
    ndim = 3
    shape = (1, 1, 9)
    device = SimpleNamespace(type="cpu")
    dtype = "torch.float32"

    def __init__(self, token=7, finite=True):
        self.token, self.finite = token, finite

    def __getitem__(self, key):
        assert key == (0, -1)
        return SimpleNamespace(argmax=lambda: SimpleNamespace(item=lambda: self.token))


TORCH = SimpleNamespace(is_tensor=lambda _: False,
                        isfinite=lambda logits: SimpleNamespace(all=lambda: logits.finite))


def rejects(fn, code):
    try:
        fn()
    except ValueError as error:
        assert str(error) == code, str(error)
    else:
        raise AssertionError(code)


def method_test(raw, root):
    # Use the exact OneVision method excerpt with a small parent projection fixture.
    source = ('from types import SimpleNamespace\n'
              'class Parent:\n'
              '    def prepare_inputs_for_generation(self, input_ids, **kwargs):\n'
              '        self.parent_calls += 1\n'
              '        q = self.decode_ids if kwargs.get("next_sequence_length") == 1 else input_ids\n'
              '        return dict(kwargs, input_ids=q)\n'
              'class OneVision(Parent):\n'
              '    config = SimpleNamespace(_attn_implementation="eager", is_encoder_decoder=False)\n') + raw.decode() + "\n"
    saved = root / "onevision_fixture.py"
    saved.write_bytes(source.encode())
    spec = importlib.util.spec_from_file_location("onevision_fixture", saved)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.OneVision()
    model.parent_calls = 0
    model.decode_ids = Ids(1)
    pre_ids, cache, image = Ids(8), Cache(), object()
    try:
        model.prepare_inputs_for_generation(pre_ids, past_key_values=cache, pixel_values=image, is_first_iteration=True)
    except TypeError as error:
        assert str(error) == "'NoneType' object is not subscriptable"
    else:
        raise AssertionError("frozen original did not reproduce")
    with patch.object(repair, "provider") as pinned, repair.install(model, mod.OneVision, TORCH, Cache, mod.Parent) as state:
        first = model.prepare_inputs_for_generation(pre_ids, past_key_values=cache, pixel_values=image,
                                                    is_first_iteration=True)
        assert first["cache_position"] is None and first["position_ids"] is None and first["pixel_values"] is image
        cache.length = 8
        second = model.prepare_inputs_for_generation(pre_ids, past_key_values=cache, pixel_values=image,
                                                     next_sequence_length=1)
        assert second["input_ids"] is model.decode_ids and second["cache_position"] is None and second["pixel_values"] is None
        positions = [8]
        retained = model.prepare_inputs_for_generation(pre_ids, past_key_values=cache, cache_position=positions,
                                                       pixel_values=image)
        assert retained["cache_position"] is positions and retained["pixel_values"] is None
        rejects(lambda: model.prepare_inputs_for_generation(pre_ids, past_key_values=cache, pixel_values=image,
                     next_sequence_length=1, is_first_iteration=True), "EXACT_BS1_EAGER_GREEDY_DYNAMIC_CACHE_GEOMETRY")
        assert [row["drop_visual"] for row in state.events] == [False, True, True]
        assert [row.args[1] for row in pinned.call_args_list] == [repair.ONEVISION, repair.HF_GENERATION, repair.HF_CACHE]
    assert "prepare_inputs_for_generation" not in vars(model) and model.parent_calls == 5


def witness_test():
    def actual_fixture(input_ids=None, past_key_values=None, pixel_values=None, cache_position=None):
        past_key_values.length += input_ids.shape[1]
        return SimpleNamespace(past_key_values=past_key_values, logits=Logits())

    cache = Cache()
    witness = TwoForwards(TORCH, Cache)
    wrapped = witness.around_forward(actual_fixture)
    call = witness.around_generate(lambda _, **kwargs: None)
    cfg = SimpleNamespace(eos_token_id=8, do_sample=False, num_beams=1, max_new_tokens=128)
    call(None, generation_config=cfg)
    wrapped(input_ids=Ids(8), past_key_values=cache, pixel_values=object())
    try:
        wrapped(input_ids=Ids(1), past_key_values=cache)
    except TwoForwardComplete:
        pass
    else:
        raise AssertionError("missing post-decode stop")
    assert witness.complete and witness.entered == 2 and cache.length == 9
    assert [row["cache_after"] for row in witness.records] == [8, 9]
    assert witness.records[0]["greedy_token_SHA256"] == token_pin(7)
    assert witness.records[1]["actual_decode_matches_prefill_greedy"] is True
    early = TwoForwards(TORCH, Cache)
    early.eos = {7}
    early.around_forward(actual_fixture)(input_ids=Ids(8), past_key_values=Cache(), pixel_values=object())
    assert not early.complete and early.early_eos(SimpleNamespace(finish_reason="stop", token_ids=[7]))
    early.eos = {8}
    cache = Cache(8)
    rejects(lambda: early.around_forward(actual_fixture)(input_ids=Ids(1, token=6), past_key_values=cache),
            "ACTUAL_FIRST_DECODE_GREEDY_TOKEN")


def main():
    repo = Path(__file__).resolve().parents[3]
    evidence = (repo / "local-only/llava-cache-position-pinned-source-audit-d782a1a90fd5465a93710b5f3eda31c7.json").read_bytes()
    assert sha256(evidence).hexdigest() == SOURCE_AUDIT
    audit = json.loads(evidence)["response"]
    assert audit["hf_install"][0]["sha256"] == repair.HF_GENERATION
    assert audit["hf_install"][0]["cache_sha256"] == repair.HF_CACHE
    raw = audit["model_prepare"]["code"].encode()
    assert "super(_exact_original_onevision_class, self)" in repair.derive_method(raw)
    rejects(lambda: repair.derive_method(raw + b" "), "EXACT_PINNED_ONEVISION_METHOD_TEXT")
    with TemporaryDirectory() as directory:
        method_test(raw, Path(directory).resolve())
    witness_test()
    source = repo / "code/bw1000/training-llava-clevr-original5000-val16-generation-boundary-v1/cpu_probe.py"
    derived = derive(source.read_bytes())
    compile(derived, "derived-actual-two-forward-runner.py", "exec")
    print(json.dumps(dict(actual_source_readback_exact=True, visual_only_derive=True, fixture_two_returns=True,
                          early_EOS_uncovered=True, derived_CPU_INVOCATION_SHA256=sha256(derived).hexdigest(),
                          actual_model_forwards=0, GPU_calls=0, target_calls=0)))


if __name__ == "__main__":
    main()
