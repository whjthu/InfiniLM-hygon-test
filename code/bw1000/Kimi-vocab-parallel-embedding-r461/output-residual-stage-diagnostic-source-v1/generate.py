"""Freeze original source plus diagnostics, with a byte-exact reverse check."""
import argparse
import hashlib
import json
from pathlib import Path
import re


ORIGINAL_MANIFEST = "6e6a99e1b5c2bafa1744feaf96f122a6a148b3e4e5a241bf1e4886eeab6e0731"


def statements(data, markers, emitter):
    for stage, text in markers:
        text = text.encode("utf-8")
        if data.count(text) != 1:
            raise ValueError(f"ambiguous original source at {stage}")
        ind = text[:len(text) - len(text.lstrip(b" "))]
        a = ind + emitter(stage + ".enter").encode() + b"\n"
        b = ind + emitter(stage + ".return").encode() + b"\n"
        data = data.replace(text, a + text + b)
    return data


def generate(source):
    raw = (source / "SOURCE_FILES.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MANIFEST:
        raise ValueError("different GPU333 original source manifest")
    manifest = json.loads(raw)["files"]
    originals = {}
    for name in ("component.cpp", "residual_method.inc", "run_arm.py", "output_residual_tiles.hpp",
                 "CHECKPOINT_WEIGHTS.json"):
        data = (source / name).read_bytes()
        pin = manifest[name]
        if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["SHA256"]:
            raise ValueError(f"original source changed: {name}")
        originals[name] = data
    native = lambda stage: f'kimi_stage::mark("{stage}");'
    wrapper = originals["component.cpp"]
    wrapper = wrapper.replace(b'#include "output_residual_tiles.hpp"\n',
                              b'#include "output_residual_tiles.hpp"\n#include "stage.hpp"\n')
    wrapper = statements(wrapper, [
        ("runtime_drain", '                runtime->syncStream();\n'),
        ("cuda_device_count", '        cuda_ok(cudaGetDeviceCount(&device_count));\n'),
        ("cuda_set_device", '        cuda_ok(cudaSetDevice(0));\n'),
        ("cuda_memory", '    cuda_ok(cudaMemGetInfo(&available, &total));\n'),
        ("runtime_create", '        probe.runtime = rt::Runtime::create({rt::Device::Type::kHygon, 0});\n'),
        ("owner_thread_setup", '        runtime.enableOwnerDrainFailStop();\n        runtime.assertOwnerThread();\n'),
        ("norm_create", '        probe.output_attn_res_norm_ =\n'
                       '            std::make_unique<infinilm::layers::RMSNorm>(runtime, shape::hidden, epsilon, type);\n'),
        ("linear_create", '        probe.output_attn_res_proj_ =\n'
                         '            std::make_unique<infinilm::layers::linear::ReplicatedLinear>(runtime, shape::hidden, 1, false, type);\n'),
        ("weight_load", '        probe.output_attn_res_norm_->load_parameter(runtime, "weight", probe.norm_host);\n'
                       '        probe.output_attn_res_proj_->load_parameter(runtime, "weight", probe.projection_host);\n'),
        ("weight_postload", '        probe.output_attn_res_norm_->process_weights_after_loading(runtime, pre_transpose != 0);\n'
                           '        probe.output_attn_res_proj_->process_weights_after_loading(runtime, pre_transpose != 0);\n'),
        ("residual_device_allocate", '        probe.backing = rt::Tensor::empty(runtime, {probe.host_values.size()}, type);\n'),
        ("residual_h2d_enqueue", '        probe.backing->copy_from(runtime, probe.input_host);\n'),
        ("preforward_sync", '        runtime.syncStream();\n        sample_driver_memory(memory + 2);\n'),
        ("forward", '        probe.output = probe.apply(runtime, hidden, probe.values, shape::residuals - 1, mode == 1);\n'),
        ("forward_sync", '        runtime.syncStream();\n        const auto done = Clock::now();\n'),
        ("d2h_enqueue_and_sync", '        probe.output_host->copy_from(runtime, probe.output);\n'
                                '        probe.input_readback->copy_from(runtime, probe.backing);\n'
                                '        runtime.syncStream();\n'),
    ], native)
    method = statements(originals["residual_method.inc"], [
        ("tile_rmsnorm", "                normalized = output_attn_res_norm_->forward(runtime, tile_values);\n"),
        ("tile_projection", "                tile_scores = output_attn_res_proj_->forward(runtime, normalized);\n"),
        ("tile_score_copy", "                target->copy_from(runtime, tile_scores);\n"),
        ("tile_sync", "                runtime.syncStream();\n            } catch (...) {\n"),
        ("full_rmsnorm", "        normalized_full = output_attn_res_norm_->forward(runtime, values);\n"),
        ("full_projection", "        scores = output_attn_res_proj_->forward(runtime, normalized_full);\n"),
        ("softmax", "    infinilm::execution::submit<infini::ops::Softmax>(\n"
                    "        runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), scores, std::int64_t{1}, std::optional<infini::ops::DataType>{}, scores);\n"),
        ("final_gemm", "    infinilm::execution::submit<infini::ops::Gemm>(\n"
                       "        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,\n"
                       "        infini::ops::Gemm::DefaultConfig(runtime.device().type()), score_matrix, values,\n"
                       "        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},\n"
                       "        std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, matmul_output);\n"),
    ], native)
    # Keep tile_sync's return marker inside the try scope, immediately after sync.
    marker = b'                kimi_stage::mark("tile_sync.return");\n'
    anchor = b"            } catch (...) {\n" + marker
    if method.count(anchor) != 1:
        raise ValueError("tile sync marker has an unexpected exception boundary")
    method = method.replace(anchor, marker + b"            } catch (...) {\n")
    py = originals["run_arm.py"].replace(b"import sys\n", b"import sys\n\nfrom stages import mark\n")
    py = statements(py, [
        ("load_library", "        library = C.CDLL(str(library_path), mode=C.RTLD_LOCAL)\n"),
        ("native_cabi", "        output, seconds, driver = native_arm(library, args.mode, int(pre_transpose), eps, operands)\n"),
        ("output_write", '        with (args.output_directory / "output.bf16").open("xb") as f:\n'
                        '            f.write(raw)\n'),
    ], lambda stage: f'mark("{stage}")')
    clean = re.compile(rb'^[ ]*kimi_stage::mark\("[a-z0-9_.]+"\);\n', re.MULTILINE)
    if clean.sub(b"", wrapper).replace(b'#include "stage.hpp"\n', b"") != originals["component.cpp"]:
        raise ValueError("component changed beyond logging")
    if clean.sub(b"", method) != originals["residual_method.inc"]:
        raise ValueError("residual arithmetic changed beyond logging")
    if (re.sub(rb'^[ ]*mark\("[a-z0-9_.]+"\)\n', b"", py, flags=re.MULTILINE)
            .replace(b"\nfrom stages import mark\n", b"") != originals["run_arm.py"]):
        raise ValueError("Python arm inputs or dispatch changed beyond logging")
    return dict(originals, **{"component.cpp": wrapper, "residual_method.inc": method, "run_arm.py": py})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for name, value in generate(args.original_source).items():
        with (args.output / name).open("xb") as file:
            file.write(value)
