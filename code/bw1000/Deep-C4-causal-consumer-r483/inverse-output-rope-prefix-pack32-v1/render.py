"""Render a one-TU Deep inverse RoPE candidate that changes aligned prefix copy width."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE.parent / "inverse-output-rope-CPU-v1"
BASE_MANIFEST = "fb0edd19695838a6df632e88ae10351569362de28ed11067848bb909b3aa7465"
PROFILE = "analysis/bw1000/Deep-r467-post1498-main-phase-decision.json"
PROFILE_PIN = "7ffef1122f09ecaaf319da3f55c9e7b76d21ceaf4e086a99106a9016691c2da4"
LABEL = "Deep-inverse-output-rope-prefix-pack32-gfx936-CPU-v1"
SYMBOL = "deep_inverse_rope_h1_pack32_v1"
COPY_PAIR = """__device__ __forceinline__ void copy_prefix_pair32(uint16_t* out, const uint16_t* in) {
    uint32_t pair;
    __builtin_memcpy(&pair, __builtin_assume_aligned(in, alignof(uint32_t)), sizeof(pair));
    __builtin_memcpy(__builtin_assume_aligned(out, alignof(uint32_t)), &pair, sizeof(pair));
}
"""
LAUNCH = "inverse_output<HeadsPerBlock, PackedPrefix><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>("


def digest(raw):
    return sha256(raw).hexdigest()


def checked(path, pin):
    raw = path.read_bytes()
    if digest(raw) != pin:
        raise ValueError("frozen original differs: " + path.name)
    return raw


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("frozen transform anchor differs")
    return text.replace(before, after)


def replace_exports(source):
    h1 = '\nextern "C" int deep_inverse_rope_h1_v1('
    h2 = '\nextern "C" int deep_inverse_rope_h2_v1('
    if source.count(h1) != 1 or source.count(h2) != 1:
        raise ValueError("frozen H1/H2 source exports differ")
    return source[:source.index(h1)] + f"""
extern "C" int {SYMBOL}(void* out, const void* in, const void* table,
                                             size_t tokens, size_t heads, void* stream) {{
    return launch<1, true>(out, in, table, tokens, heads, stream);
}}
"""


def candidate(source):
    source = once(source, "template<unsigned HeadsPerBlock>\n__global__",
                  COPY_PAIR + "\ntemplate<unsigned HeadsPerBlock, bool PackedPrefix>\n__global__")
    source = once(source, """        out[2 * pair] = in[2 * pair];
        out[2 * pair + 1] = in[2 * pair + 1];""", """        if constexpr (PackedPrefix) {
            copy_prefix_pair32(out + 2 * pair, in + 2 * pair);
        } else {
            out[2 * pair] = in[2 * pair];
            out[2 * pair + 1] = in[2 * pair + 1];
        }""")
    source = once(source, "template<unsigned HeadsPerBlock>\nint launch",
                  "template<unsigned HeadsPerBlock, bool PackedPrefix>\nint launch")
    source = once(source, "    const unsigned grid =", """    if constexpr (PackedPrefix) {
        if ((x | y) % alignof(uint32_t)) {
            return launch<HeadsPerBlock, false>(out, in, table, tokens, heads, stream);
        }
    }
    const unsigned grid =""")
    source = once(source, "inverse_output<HeadsPerBlock><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(",
                  LAUNCH)
    return replace_exports(source)


def materialize(output):
    index = json.loads(checked(BASE / "SOURCE_FILES.json", BASE_MANIFEST))["files_SHA256"]
    profile = json.loads(checked(REPO / PROFILE, PROFILE_PIN))
    source_name = "inverse_output_rope.cu"
    source = candidate(checked(BASE / source_name, index[source_name]).decode())
    worker = checked(BASE / "compile_once.py", index["compile_once.py"]).decode()
    plan = json.loads(checked(BASE / "PLAN.json", index["PLAN.json"]))
    worker = once(worker, 'LABEL = "' + plan["label"] + '"', 'LABEL = "' + LABEL + '"')
    worker = once(worker, 'EXPORTS = {"deep_inverse_rope_h1_v1", "deep_inverse_rope_h2_v1"}',
                  f'EXPORTS = {{"{SYMBOL}"}}')
    worker = once(worker, "TWO_TARGET_CABI_EXPORTS", "EXACT_PREFIX_PACK32_CABI_EXPORT")
    variant = dict(heads_per_CTA=1, block_threads=64, grid_x_primary=8, export=SYMBOL, prefix_copy_bits=32,
                   unsupported_alignment_fallback_bits=16, parameter="unrotated BF16 prefix copy width")
    variant["id"] = digest(json.dumps(variant, sort_keys=True, separators=(",", ":")).encode())
    old = dict(source_SHA256=plan["source_SHA256"], variant=plan["variants"][0], package_manifest_SHA256=BASE_MANIFEST)
    plan.update(label=LABEL, source_SHA256=digest(source.encode()), variants=[variant], existing_H1_baseline=old,
                frozen_compiler_template_SHA256=index["compile_once.py"], exact_profile_SHA256=PROFILE_PIN,
                measured_inverse_rope_stream_enclosures_ms=[
                    r["attention_inner_siblings_stream_ms"]["inverse_output_rope"] for r in profile["windows"]],
                prefix_bytes_per_head_per_direction=896, whole_output_bytes_per_head=1024)
    files = {source_name: source.encode(), "compile_once.py": worker.encode(),
             "PLAN.json": (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode()}
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix="deep-prefix-pack32-") as temp:
        staged = Path(temp) / "source"
        staged.mkdir()
        for name, data in files.items():
            (staged / name).write_bytes(data)
        manifest = (json.dumps({"files_SHA256": {name: digest(data) for name, data in files.items()}},
                               sort_keys=True, indent=2) + "\n").encode()
        (staged / "SOURCE_FILES.json").write_bytes(manifest)
        staged.rename(output)
    return dict(target_manifest_SHA256=digest(manifest), files_SHA256=json.loads(manifest)["files_SHA256"],
                candidate_id=variant["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), sort_keys=True, indent=2))
