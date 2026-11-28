"""Exercise actual wrapping, exact original TypeError projection and sentinel with no model payload."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from cpu_probe import EXPOSED_BINDING, EXPOSED_BYTES, EXPOSED_IMAGE, EXPOSED_SHA, frozen_native, verify_diagnostic_image
from derive_native import derive
from generation_boundary import BoundaryTrace, KNOWN_MESSAGE_SHA, PreForwardSentinel, error_evidence


SOURCE = '''class Model:
    def __init__(self, fail):
        self.fail = fail
        self.forward_calls = 0

    def generate(self, input_ids=None, **kwargs):
        prepared = self.prepare_inputs_for_generation(input_ids=input_ids, **kwargs)
        return self.forward(**prepared)

    def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None, **kwargs):
        if self.fail:
            return past_key_values[0]
        return {"input_ids": input_ids, "position_ids": None}

    def forward(self, input_ids=None, position_ids=None):
        self.forward_calls += 1
        return input_ids

class Template:
    def generate(self, model, **kwargs):
        return self.get_generate_ids(model.generate(**kwargs))

    def get_generate_ids(self, ids):
        return ids

def aliased_none(past_key_values=None, position_ids=None):
    return past_key_values[0] if position_ids[0] is None else 0
'''


def test_trace(folder):
    provider = folder / "source_fixture.py"
    provider.write_text(SOURCE, encoding="utf-8")
    namespace = {"__name__": "transformers.fixture_metadata"}
    exec(compile(SOURCE, str(provider), "exec"), namespace)
    Model, Template = namespace["Model"], namespace["Template"]
    model, template = Model(fail=True), Template()
    original_signature = inspect.signature(model.generate)
    trace = BoundaryTrace()
    with trace.attach_template(template), trace.attach_model(model):
        assert inspect.signature(model.generate) == original_signature
        assert model.generate.__wrapped__.__self__ is model
        try:
            template.generate(model, input_ids=["private_value_not_logged"], past_key_values=None,
                              unknown_private_kw={"sensitive": "no"})
        except TypeError as error:
            assert sha256(str(error).encode()).hexdigest() == KNOWN_MESSAGE_SHA
        else:
            raise AssertionError("original-style error absent")
    result = trace.project()
    assert all(result["providers"][label]["file_SHA256"] == sha256(provider.read_bytes()).hexdigest()
               for label in result["providers"])
    assert result["real_forward_calls"] == 0 and not result["pre_forward_reached"]
    assert result["first_error"][0]["original_5207_message_match"] is True
    assert result["first_error"][0]["verified_none_subscript"] == dict(
        local="past_key_values", is_none=True, basis="unique_direct_local_binary_subscript")
    assert result["first_error"][0]["deepest_nullable"]["past_key_values"] == {
        "present": True, "type": "builtins.NoneType", "is_none": True}
    assert all(key not in json.dumps(result) for key in ("private_value_not_logged", "unknown_private_kw", "sensitive"))
    assert "generate" not in vars(model) and "generate" not in vars(template)

    model = Model(fail=False)
    trace = BoundaryTrace(stop_before_forward=True)
    with trace.attach_template(template), trace.attach_model(model):
        try:
            template.generate(model, input_ids=["opaque"])
        except PreForwardSentinel:
            pass
        else:
            raise AssertionError("forward sentinel did not stop")
    result = trace.project()
    assert result["pre_forward_reached"] is True and result["real_forward_calls"] == model.forward_calls == 0
    assert result["counts"]["model.forward:sentinel"] == 1 and result["first_error"] is None

    passing = BoundaryTrace()
    with passing.attach_template(template), passing.attach_model(model):
        assert template.generate(model, input_ids=["opaque"], past_key_values=None) == ["opaque"]
    assert passing.counts["template.get_generate_ids:return"] == 1 and model.forward_calls == 1
    try:
        try:
            namespace["aliased_none"]()
        except TypeError as error:
            raise RuntimeError("test_cause_content_not_logged") from error
    except RuntimeError as error:
        chain = error_evidence(error)
    assert len(chain) == 2 and chain[1]["original_5207_message_match"] is True
    assert chain[1]["verified_none_subscript"] is None and "test_cause_content_not_logged" not in json.dumps(chain)


def main():
    repo = Path(__file__).resolve().parents[3]
    root = repo / "code/bw1000/training-llava-clevr-original5000-val16-native-v3"
    raw = (repo / "local-only/llava-clevr-native64-cpu-v1/INPUTS.json").read_bytes()
    first = json.loads(raw)["rows"][0]
    assert sha256(raw).hexdigest() == EXPOSED_BINDING and first["ordinal"] == 1
    assert (first["path"], first["bytes"], first["sha256"]) == ("images/" + EXPOSED_IMAGE, EXPOSED_BYTES, EXPOSED_SHA)
    loaded_native = frozen_native(root)
    assert loaded_native.CAP == 128
    opts = SimpleNamespace(diagnostic_image=repo / "local-only/domain-llava-clevr-caption-v1-staged/images" /
                           EXPOSED_IMAGE, diagnostic_image_sha=EXPOSED_SHA)
    messages = verify_diagnostic_image(opts, loaded_native)
    assert [item["role"] for item in messages] == ["user"]
    archive = (repo / "results/bw1000/2026-11-28/"
               "specialized-llava-clevr-val16-5207-checkpoint-load-pass-first-generation-none.json").read_bytes()
    assert sha256(archive).hexdigest() == "e0d5a4004e9da53c22bfaea36e3410c03728ed869489ac0e563ccf1f3840539d"
    historical = json.loads(archive)["run"]
    assert historical["accounting"]["gpu"][0] == "1103606.5207" and historical["cleanup_verified"]
    assert historical["worker_failure"]["detail_SHA256"] == KNOWN_MESSAGE_SHA
    generated = derive((root / "native.py").read_bytes())
    compile(generated, "derived-native-boundary.py", "exec")
    with TemporaryDirectory() as directory:
        test_trace(Path(directory).resolve())
    print(json.dumps(dict(v3_source_exact=True, actual_exposed_training_PNG_decoded=True, original_error_digest_verified=True,
                          derived_native_SHA256=sha256(generated).hexdigest(),
                          real_model_calls=0, device_calls=0, wrapped_error_fixture=True,
                          sentinel_forbids_real_forward=True, ambiguous_operand_not_claimed=True)))


if __name__ == "__main__":
    main()
