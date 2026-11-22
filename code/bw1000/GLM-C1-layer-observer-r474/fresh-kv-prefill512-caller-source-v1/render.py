"""Render the narrowly scoped first-chunk T512 caller against exact .918 source."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = HERE.parent / "fresh-kv-explicit-ar-source-v1"
PARENT_MANIFEST_SHA = "60b7b3b3d1a74bed37d948e9397a8b0035d1a438c411b194cf8c023b0f59f2ac"
TENSOR = Path("code/bw1000/GLM-value-latent-reassociation-r431/parent/tensor.hpp")
TENSOR_SHA = "dfc787f233bef2fb5888d0954440746334386023bad3f6207164024b377eb658"
BASE_PERF = Path("results/bw1000/2026-11-22/GLM474-fresh-kv-tile-paired-original1026-terminal.json")
BASE_PERF_SHA = "bef3703906cf4b16da42635c44750694cbbfcb15cc98300db7d9696b2c5e6d95"
SYMBOL = "modern_glm_global_fresh_kv_prefill512_r474"
EXTRA = ("fresh_kv_prefill512.hpp", "forward_explicit_prefill512.inc", "fresh_kv_prefill512_materialize.inc")
COUNTERS = ("attempted", "packed", "rotary_layout", "tensor_layout", "address_span")


def digest(data):
    return sha256(data).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("pinned source/evidence changed: " + path.name)
    return raw


def once(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError("original prefill edit boundary differs")
    return raw.replace(before, after, 1)


def original():
    plan = json.loads(checked(PARENT / "SOURCE_FILES.json", PARENT_MANIFEST_SHA))
    for item in plan["files"]:
        checked(REPO / item["path"], item["SHA256"])
    checked(REPO / TENSOR, TENSOR_SHA)
    checked(REPO / BASE_PERF, BASE_PERF_SHA)
    script = PARENT / "render.py"
    loaded = {"__file__": str(script), "__name__": "verified_explicit_ar"}
    exec(compile(script.read_bytes(), str(script), "exec"), loaded)
    files = loaded["sources"]()
    if {name: digest(raw) for name, raw in files.items()} != plan["generated_sources_SHA256"]:
        raise ValueError("actual .918 generated source differs")
    return files


def typed(raw):
    attach = b"    result['decode_forward']=decode\n"
    raw = once(raw, attach, attach +
               b"    trial=getattr(library,'modern_glm_global_fresh_kv_prefill512_r474')\n"
               b"    trial.argtypes,trial.restype=SIGNATURES['forward'][0]+[C.POINTER(Z),Z],I\n"
               b"    result['prefill512_forward']=trial\n")
    raw = once(raw, b'*,phase="prefill"):', b'*,phase="prefill",prefill512=False):')
    raw = once(raw, b"        forward=self.f['decode_forward' if phase=='decode' else 'forward']\n",
               b"""        if type(prefill512) is not bool or (prefill512 and (phase!='prefill' or tokens!=512 or past!=0 or not stream_pointer)):
            raise ValueError('fresh K/V trial requires an explicitly admitted initial C1 T512 prefill')
        if prefill512:
            counts=(Z*5)()
            rc=self.f['prefill512_forward'](self.ctx,tokens,1,packet,V(input_pointer),V(position_pointer),
                                            V(logits_pointer),0,V(stream_pointer),counts,len(counts))
            self.last_prefill512=dict(zip(('attempted','packed','rotary_layout','tensor_layout','address_span'),counts))
            self.check(rc)
            self.past+=tokens
            return self.last_prefill512.copy()
        forward=self.f['decode_forward' if phase=='decode' else 'forward']
""")
    return raw


def sources():
    files = original()
    files["full_bridge_observer.cpp"] += b'\n#include "forward_explicit_prefill512.inc"\n'
    attention = files["NoV_tp16.cpp"]
    header = b'#include "fresh_kv_explicit_ar.hpp"\n'
    source = once(attention, header, header + b'#include "fresh_kv_prefill512.hpp"\n')
    before = b"""auto fresh_all=::glm_r474_fresh_kv_route::with_fresh_call(
    &r, ::glm_r474_fresh_kv_ar::active_for(&r) && requests.size()==1 && tokens==1 && requests.front().tokens==1 && requests.front().past>0,
    [&]{ return prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot); });"""
    after = b"""auto fresh_all=::glm_r474_fresh_kv_prefill512::with_fresh(
    &r, requests.size()==1 && tokens==512 && requests.front().row==0 && requests.front().past==0
        && requests.front().tokens==512 && prepare_->local_heads()==4,
    [&] { return ::glm_r474_fresh_kv_route::with_fresh_call(
        &r, ::glm_r474_fresh_kv_ar::active_for(&r) && requests.size()==1 && tokens==1 && requests.front().tokens==1 && requests.front().past>0,
        [&]{ return prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot); }); });"""
    files["NoV_tp16.cpp"] = once(source, before, after)
    if once(once(files["NoV_tp16.cpp"], after, before), header + b'#include "fresh_kv_prefill512.hpp"\n', header) != attention:
        raise ValueError("original request validation, mask or cache changed")
    header = b'#include "fresh_kv_state.inc"\n'
    files["prepare_tp16.cpp"] = once(files["prepare_tp16.cpp"], header, header + b'#include "fresh_kv_prefill512.hpp"\n')
    materialize = files["fresh_kv_materialize.inc"]
    first, tail = materialize.split(b"rt::Tensor value;\n", 1)
    files["fresh_kv_materialize.inc"] = (
        first + b'rt::Tensor value;\n#include "fresh_kv_prefill512_materialize.inc"\nif (!prefill_packed) {\n' + tail + b'}\n')
    binding = files["typed_binding.py"]
    files["typed_binding.py"] = typed(binding)
    files["native_arm.py"] = once(files["native_arm.py"], digest(binding).encode(), digest(files["typed_binding.py"]).encode())
    for name in EXTRA:
        files[name] = (HERE / name).read_bytes()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sources()
    args.output.mkdir()
    for name, raw in files.items():
        with (args.output / name).open("xb") as target:
            target.write(raw)
    result = dict(source_SHA256={name: digest(raw) for name, raw in sorted(files.items())}, added_CABI=SYMBOL,
                  expected_host_units=["u00", "u15", "u17"], additional_gpu_TU=0,
                  on_route="initial C1 row0 past0 T512 + exact runtime size/stride/address validation",
                  actual_target_builds=0, actual_GPU_steps=0, GPU_GO=False, Model_GO=False)
    with (args.output / "BINDING.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
