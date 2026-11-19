"""Generate the isolated device caller's residual method from the frozen model source."""
import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "1fc4c11757d5f73e7e217268a73d01d3b9a5ccbdc67709e95ce93afd055207e3"
SIGNATURE = """infinilm::runtime::Tensor KimiK3TextModel::apply_output_attn_res(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &block_residual_storage,
    size_t block_residual_count) const {"""
ISOLATED_SIGNATURE = """infinilm::runtime::Tensor ResidualProbe::apply(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &block_residual_storage,
    size_t block_residual_count, bool tiled) const {"""
MODEL_OWNER = """tp_size_ == 32 && pp_size_ == 1 && pp_stage_ == 0
            && dynamic_cast<const infinilm::layers::VocabParallelEmbeddingR461 *>(embed_tokens_.get())
            && runtime.device().type() == Device::Type::kHygon"""
MODEL_SETTING = 'std::getenv("INFINILM_KIMI_OUTPUT_RESIDUAL_TILE64")'


def extract(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("frozen CPU330 model source differs")
    source = raw.decode("utf-8")
    method = source[source.index(SIGNATURE):source.index("infinilm::runtime::Tensor KimiK3TextModel::forward(")]
    for original, replacement in (
        (SIGNATURE, ISOLATED_SIGNATURE),
        (MODEL_SETTING, '(tiled ? "1" : nullptr)'),
        (MODEL_OWNER, "runtime.device().type() == Device::Type::kHygon"),
    ):
        if method.count(original) != 1:
            raise ValueError("ambiguous isolated component boundary")
        method = method.replace(original, replacement)
    return (method.rstrip() + "\n").encode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_model_source", type=Path)
    parser.add_argument("generated_include", type=Path)
    args = parser.parse_args()
    with args.generated_include.open("xb") as output:
        output.write(extract(args.frozen_model_source.read_bytes()))
