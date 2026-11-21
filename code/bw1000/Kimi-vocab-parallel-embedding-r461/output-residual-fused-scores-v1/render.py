"""Extend the frozen Kimi residual component with a third, fused score arm."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'output-residual-component-CPU-v1'
BASE = {
    'component.cpp': '2a20e0017d3cc7ed090638ae45b6bc42a09c6d3a815b573a8a0eec83a91f4ed6',
    'residual_method.inc': '7bb4d627f918465415a4e47456dca83e197735a38e77de099ea6d08d24adf74b',
    'output_residual_tiles.hpp': 'd554d7e131e1eb5f6e2a39c1afd162acb1ba04124c67051991b91cd87116a9b2',
}


def change(source, old, new):
    if source.count(old) != 1:
        raise ValueError('frozen Kimi residual source boundary differs')
    return source.replace(old, new, 1)


def component(source):
    changes = (
        ('namespace rt = infinilm::runtime;',
         'extern "C" __attribute__((visibility("hidden"))) int kimi_output_residual_scores_v1_launch(\n'
         '    void*, const void*, const void*, const void*, float, void*);\n\n'
         'namespace rt = infinilm::runtime;'),
        ('    std::vector<BF16> host_values, observed_values;',
         '    rt::Tensor fused_norm, fused_projection;\n'
         '    mutable rt::Tensor fused_score_hold;\n'
         '    std::vector<BF16> host_values, observed_values, observed_fused_norm, observed_fused_projection;'),
        ('&, size_t, bool) const;', '&, size_t, int, float) const;'),
        ('(mode != 0 && mode != 1)', '(mode < 0 || mode > 2)'),
        ('        probe.host_values.assign(value_words + 2 * guard_words, canary);',
         '        if (mode == 2) {\n'
         '            probe.fused_norm = rt::Tensor::empty(runtime, {shape::hidden}, type);\n'
         '            probe.fused_projection = rt::Tensor::empty(runtime, {1, shape::hidden}, type);\n'
         '            probe.fused_norm->copy_from(runtime, probe.norm_host);\n'
         '            probe.fused_projection->copy_from(runtime, probe.projection_host);\n'
         '        }\n'
         '        probe.host_values.assign(value_words + 2 * guard_words, canary);'),
        ('shape::residuals - 1, mode == 1);', 'shape::residuals - 1, mode, static_cast<float>(epsilon));'),
        ('        metrics[0] = std::chrono::duration<double>(done - dispatch).count();',
         '        if (mode == 2) {\n'
         '            probe.observed_fused_norm.resize(shape::hidden);\n'
         '            probe.observed_fused_projection.resize(shape::hidden);\n'
         '            const auto norm_back = host_data(probe.observed_fused_norm.data(), {shape::hidden});\n'
         '            const auto projection_back = host_data(probe.observed_fused_projection.data(), {1, shape::hidden});\n'
         '            try {\n'
         '                norm_back->copy_from(runtime, probe.fused_norm);\n'
         '                projection_back->copy_from(runtime, probe.fused_projection);\n'
         '                runtime.syncStream();\n'
         '            } catch (...) {\n'
         '                try { runtime.syncStream(); } catch (...) { std::terminate(); }\n'
         '                throw;\n'
         '            }\n'
         '            if (std::memcmp(norm, probe.observed_fused_norm.data(), nn * sizeof(BF16)) ||\n'
         '                std::memcmp(projection, probe.observed_fused_projection.data(), np * sizeof(BF16)))\n'
         '                throw std::runtime_error("Kimi fused residual weight storage changed");\n'
         '        }\n'
         '        metrics[0] = std::chrono::duration<double>(done - dispatch).count();'),
    )
    for old, new in changes:
        source = change(source, old, new)
    return source


def method(source):
    source = change(source, 'size_t block_residual_count, bool tiled) const',
                    'size_t block_residual_count, int mode, float epsilon) const')
    source = change(source, '(tiled ? "1" : nullptr)', '(mode == 1 ? "1" : nullptr)')
    return change(
        source, '    if (tile_setting && shape.size() == 3',
        '    if (mode == 2) {\n'
        '        if (shape != Shape{1, tiling::full_tokens, tiling::hidden} ||\n'
        '            runtime.device().type() != Device::Type::kHygon || runtime.isGraphRecording() ||\n'
        '            runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive ||\n'
        '            values->shape() != Shape{tiling::full_tokens, tiling::residuals, tiling::hidden} ||\n'
        '            values->dtype() != DType::kBFloat16 || !values->is_contiguous() ||\n'
        '            !fused_norm || !fused_projection)\n'
        '            throw std::invalid_argument("Kimi fused residual geometry or owner differs");\n'
        '        fused_score_hold = Tensor::empty(runtime, {tiling::full_tokens, tiling::residuals, 1}, DType::kBFloat16);\n'
        '        scores = fused_score_hold;\n'
        '        if (kimi_output_residual_scores_v1_launch(scores->data(), values->data(),\n'
        '                fused_norm->data(), fused_projection->data(), epsilon, runtime.stream()))\n'
        '            throw std::runtime_error("Kimi fused residual score launch failed");\n'
        '    } else if (tile_setting && shape.size() == 3',
    )


def materialize(output):
    sources = {name: (PARENT / name).read_bytes() for name in BASE}
    if {name: sha256(raw).hexdigest() for name, raw in sources.items()} != BASE:
        raise ValueError('Kimi residual source SHA-256 differs')
    sources['component.cpp'] = component(sources['component.cpp'].decode('utf-8')).encode('utf-8')
    sources['residual_method.inc'] = method(sources['residual_method.inc'].decode('utf-8')).encode('utf-8')
    sources.update({name: (HERE / name).read_bytes() for name in ('score_math.hpp', 'fused_scores.cu')})
    output.mkdir(mode=0o700)
    for name, raw in sources.items():
        (output / name).write_bytes(raw)
    return {name: sha256(raw).hexdigest() for name, raw in sources.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output), indent=2))
