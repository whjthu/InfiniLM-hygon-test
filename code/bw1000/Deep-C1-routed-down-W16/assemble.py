"""Prepare an isolated C1 W16 down-projection component from exact a314 inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

LOCAL = Path(__file__).resolve().parents[4]
CACHE = LOCAL / "Deep463-source-cache"
PINNED = {
    "kernel": ("a675db956e91e28e-deepseek_routed_i8_kernels_r20.cu",
               "a675db956e91e28ece725fab54fee844cc0622a2e632b4b32976137a77a9eb41"),
    "caller": ("e0ff23b093e51e3f-deepseek_routed_mlp_batch_r20.hpp",
               "e0ff23b093e51e3ff22f9229f563e7cd4e7973580ad68b0a10b4ad34c6ee97af"),
}
NATIVE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
ADAPTER_SHA256 = "1dcffb8cbd78d01f2b243dc0b4d54e0bec483a59be51c3dd397bd8ce86ca0369"
BASELINE_OBJECT = "08ed7c8bfb7fe56cee66a6d6bdc5a5d569b05574ae68b2e7741d755a563ea756"


def read_source(key):
    name, digest = PINNED[key]
    raw = (CACHE / name).read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"a314 {key} differs")
    return raw.decode()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("unexpected a314 derivation anchor")
    return text.replace(old, new)


def derive_kernel():
    text = read_source("kernel")
    start = text.index("__global__ void validate(")
    stop = text.index("template<int W,bool Gate>", start)
    text = text[:start] + text[stop:]
    text = text[:text.index('extern "C" int deep_routed_validate_r20(')]
    return text + """extern "C" int deep_routed_down_w16_c1(
 void* output,const void* x,const void* xs,const void* ids,
 const void* weights,const void* scales,size_t routes,void* stream){
 if(routes!=6)return int(cudaErrorInvalidValue);
 auto s=static_cast<cudaStream_t>(stream);
 routed_dot<16,false><<<dim3(256,6),256,0,s>>>(
  static_cast<__nv_bfloat16*>(output),static_cast<const int8_t*>(x),
  static_cast<const float*>(xs),static_cast<const int64_t*>(ids),
  static_cast<const int8_t*>(weights),static_cast<const float*>(scales),routes);
 return int(cudaGetLastError());
}
"""


def derive_component():
    text = read_source("caller")
    text = replace_once(text, 'namespace infinilm::models::deepseek_v4::routed_r20 {', """
extern "C" int deep_routed_down_w16_c1(void*,const void*,const void*,const void*,const void*,const void*,size_t,void*);
namespace infinilm::models::deepseek_v4::routed_down_w16_c1 {""".lstrip())
    text = replace_once(text, "r.assertOwnerThread();Work work{r};size_t tokens=x->size(0),routes=tokens*6;",
                        """r.assertOwnerThread();size_t tokens=x->size(0);
 if(tokens!=1)throw std::invalid_argument("W16 routed down requires one input token");
 Work work{r};size_t routes=6;""")
    return replace_once(text, "deep_routed_dot_r20(0,32,local->data()", "deep_routed_down_w16_c1(local->data()")


def derive_adapter():
    path = Path(__file__).resolve().parent.parent / "deep-native-dot4/deep_native_dot_adapter_r8742.cpp"
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != ADAPTER_SHA256:
        raise ValueError("unexpected historical component adapter bytes")
    text = raw.decode().replace("\r\n", "\n")
    text = replace_once(text, "// Finite component only. Mode0 default baseline; mode1 native signed dot4.",
                        "// Fixed C1 component. Mode 0 is W32; mode 1 is W16 down only.")
    text = replace_once(text, '#include "deep_routed_sdot4_forward_r8742.hpp"',
                        '#include "deep_routed_down_w16_component.hpp"')
    text = text.replace("deep_native_dot_", "deep_routed_w16_component_").replace("_r8742(", "(")
    text = text.replace("[32,", "[1,").replace("{32,", "{1,")
    text = replace_once(text, "uint64_t g=model::routed_sdot4_r8742::native_calls[0],d=model::routed_sdot4_r8742::native_calls[1];", "")
    text = replace_once(text, "else value=model::routed_sdot4_r8742::forward", "else value=model::routed_down_w16_c1::forward")
    return replace_once(text,
                        '{"native_gate_calls",model::routed_sdot4_r8742::native_calls[0]-g},{"native_down_calls",model::routed_sdot4_r8742::native_calls[1]-d}',
                        '{"down_width",mode==0?32:16}')


def verify_integer_contract():
    # Emulate each original packed-dot group and the power-of-two lane tree.
    rng = np.random.default_rng(46316)
    x = rng.integers(-128, 128, size=(6, 256), dtype=np.int32)
    w = rng.integers(-128, 128, size=(6, 4096, 256), dtype=np.int32)
    x[0], w[0] = -128, -128
    x[1], w[1] = -128, 127
    groups = (x[:, None] * w).reshape(6, 4096, 64, 4).sum(axis=3, dtype=np.int64)
    reference = groups.sum(axis=2)
    results = {}
    for width in (32, 16):
        blocks = 4096 // (256 // width)
        rows = (np.arange(blocks)[:, None] * (256 // width)
                + np.arange(256)[None, :] // width)
        if not np.array_equal(rows.ravel(), np.repeat(np.arange(4096), width)):
            raise ValueError("row/lane launch mapping differs")
        partials = groups.reshape(6, 4096, 64 // width, width).sum(axis=2)
        while partials.shape[-1] > 1:
            low, high = np.split(partials, 2, axis=2)
            partials = low + high
        if not np.array_equal(partials[..., 0], reference):
            raise ValueError("integer reduction differs")
        results[str(width)] = dict(
            blocks_per_rank_per_layer=blocks * 6,
            threads_per_output=width,
            packed_dp4a_iterations_per_lane=64 // width,
            shuffle_rounds=width.bit_length() - 1,
        )
    # Every intermediate sums at most 256 products with magnitude at most 16384.
    assert 256 * 128**2 == 4194304 < 2**31
    assert reference[0, 0] == 4194304 and reference[1, 0] == -4161536
    return results


def assemble(destination):
    sources = {
        "deep_routed_down_w16_c1.cu": derive_kernel(),
        "deep_routed_down_w16_component.hpp": derive_component(),
        "deepseek_routed_mlp_batch_r20.hpp": read_source("caller"),
        "deep_routed_down_w16_adapter.cpp": derive_adapter(),
    }
    proof = verify_integer_contract()
    manifest = {
        "Native_SHA256": NATIVE,
        "baseline_routed_object_SHA256": BASELINE_OBJECT,
        "adapter_parent_SHA256": ADAPTER_SHA256,
        "inputs_normalized_LF_SHA256": {key: digest for key, (_, digest) in PINNED.items()},
        "files_SHA256": {name: sha256(text.encode()).hexdigest() for name, text in sources.items()},
        "launch": proof,
        "CPU_integer_reference_outputs": 6 * 4096,
        "GPU_GO": False,
        "scope": "Private component candidate, only C1 down launch changes. Baseline routed object is reused "
                 "for the original gate and validation; the production caller is untouched.",
        "limits": "No device compile, timing, or BF16 backend parity yet. A halved block count can trade "
                  "scheduler overhead for longer per-lane work and different memory behavior. "
                  "Do not promote without exact-shape device parity and warm paired timing.",
    }
    destination.mkdir()
    for name, text in sources.items():
        with (destination / name).open("xb") as out:
            out.write(text.encode())
    with (destination / "SOURCE_MANIFEST.json").open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    assemble(parser.parse_args().destination)
