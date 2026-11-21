"""Render one DeepSeek inverse RoPE object with an aligned 64-bit prefix copy."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "inverse-output-rope-prefix-pack32-target-v1"
BASE_MANIFEST_SHA = "dfb1012575cb5c9615a9dbbc471734f527b7a4adb6a98235e99220060b90538d"
LABEL = "Deep-inverse-output-rope-prefix-pack64-gfx936-CPU-v1"
SYMBOL = "deep_inverse_rope_h1_pack64_v1"
SOURCE = "inverse_output_rope.cu"
LAUNCH = "inverse_output<HeadsPerBlock, PrefixBits><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>("


def digest(raw):
    return sha256(raw).hexdigest()


def checked(path, pin):
    raw = path.read_bytes()
    if digest(raw) != pin:
        raise ValueError("frozen source differs: " + path.name)
    return raw


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen source anchor differs")
    return text.replace(old, new)


def kernel(source):
    marker = "template<unsigned HeadsPerBlock, unsigned PrefixBits>\n__global__"
    source = source.replace("bool PackedPrefix", "unsigned PrefixBits").replace("PackedPrefix", "PrefixBits")
    source = replace_once(source, marker, """__device__ __forceinline__ void copy_prefix_group64(uint16_t* out, const uint16_t* in) {
    uint64_t group;
    __builtin_memcpy(&group, __builtin_assume_aligned(in, alignof(uint64_t)), sizeof(group));
    __builtin_memcpy(__builtin_assume_aligned(out, alignof(uint64_t)), &group, sizeof(group));
}

""" + marker)
    first = source.index("    for (unsigned pair = lane;")
    end = source.index("    if (lane < tail_pairs)", first)
    prior = source[first:end]
    prior = replace_once(prior, "if constexpr (PrefixBits)", "if constexpr (PrefixBits == 32)")
    source = source[:first] + """    static_assert(prefix_pairs % 2 == 0, "prefix must hold whole 64-bit groups");
    if constexpr (PrefixBits == 64) {
        for (unsigned group = lane; group < prefix_pairs / 2; group += wave) {
            copy_prefix_group64(out + 4 * group, in + 4 * group);
        }
    } else {
""" + "".join("    " + line for line in prior.splitlines(keepends=True)) + "    }\n" + source[end:]
    source = replace_once(source, """    if constexpr (PrefixBits) {
        if ((x | y) % alignof(uint32_t)) {
            return launch<HeadsPerBlock, false>(out, in, table, tokens, heads, stream);
        }
    }""", """    if constexpr (PrefixBits == 64) {
        if ((x | y) % alignof(uint64_t)) {
            return launch<HeadsPerBlock, 32>(out, in, table, tokens, heads, stream);
        }
    }
    if constexpr (PrefixBits == 32) {
        if ((x | y) % alignof(uint32_t)) {
            return launch<HeadsPerBlock, 16>(out, in, table, tokens, heads, stream);
        }
    }""")
    source = replace_once(source, "deep_inverse_rope_h1_pack32_v1", SYMBOL)
    return replace_once(source, "return launch<1, true>", "return launch<1, 64>")


def render(output):
    pins = json.loads(checked(BASE / "SOURCE_FILES.json", BASE_MANIFEST_SHA))["files_SHA256"]
    cooked = kernel(checked(BASE / SOURCE, pins[SOURCE]).decode()).encode()
    worker = checked(BASE / "compile_once.py", pins["compile_once.py"]).decode()
    plan = json.loads(checked(BASE / "PLAN.json", pins["PLAN.json"]))
    label, export = plan["label"], plan["variants"][0]["export"]
    worker = replace_once(worker, 'LABEL = "' + label + '"', 'LABEL = "' + LABEL + '"')
    worker = replace_once(worker, export, SYMBOL)
    worker = replace_once(worker, "EXACT_PREFIX_PACK32_CABI_EXPORT", "EXACT_PREFIX_PACK64_CABI_EXPORT")
    baseline = dict(plan["variants"][0], manifest_SHA256=BASE_MANIFEST_SHA)
    option = dict(heads_per_CTA=1, block_threads=64, grid_x_primary=8, prefix_copy_bits=64,
                  alignment_fallback_bits=[32, 16], export=SYMBOL)
    option["id"] = digest(json.dumps(option, sort_keys=True, separators=(",", ":")).encode())
    plan.update(label=LABEL, source_SHA256=digest(cooked), variants=[option], aligned_pack32_comparator=baseline)
    files = {SOURCE: cooked, "compile_once.py": worker.encode(),
             "PLAN.json": (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode()}
    manifest = (json.dumps({"files_SHA256": {name: digest(raw) for name, raw in files.items()}},
                           sort_keys=True, indent=2) + "\n").encode()
    output.mkdir()
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    (output / "SOURCE_FILES.json").write_bytes(manifest)
    return dict(manifest_SHA256=digest(manifest), variant=option, files_SHA256=json.loads(manifest)["files_SHA256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(render(parser.parse_args().output), sort_keys=True, indent=2))
