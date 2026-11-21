"""Derive only the fresh K/V caller and materialization boundary from CPU308."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "depth-stratified-observer-CPU-v1"
PINNED_PLAN = "48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6"
PREPARE = "35fec2e8202395423e4937857bc41dc6a5ea00371aed7c3233209a9102f1afcb"
ATTENTION = "6548aacdf94d0ab481cc27f613fc4713906795a0b31cdfc73aa5e22430f63f29"
DECLARATION = "8027b00c3b88ed6fd6445ddf68076e6896c309a82d67b6e97cb872eaa587ea68"
OLD_FRESH = b"auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);"
FRESH = (
    b"auto fresh_all=::glm_r474_fresh_kv_route::with_fresh_call(\n"
    b"    &r, requests.size()==1 && tokens==1 && requests.front().tokens==1 && requests.front().past>0,\n"
    b"    [&]{ return prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot); });"
)
OLD_PACK = (
    b"key->narrow({{2,0,192}})->copy_from(r,expanded->narrow({{2,0,192}}));\n"
    b" for(size_t head=0;head<heads_;++head)key->narrow({{1,head,1},{2,192,64}})->copy_from(r,rotary);\n"
    b" auto value=expanded->narrow({{2,192,256}})->contiguous(r);work.held.push_back(value);"
)
PACK = b'\n#include "fresh_kv_materialize.inc"'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"CPU308 source identity differs: {path.name}")
    return raw


def replace(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError("original caller edit must match exactly once")
    return raw.replace(old, new, 1)


def originals():
    plan = json.loads(pin(PARENT / "PLAN.json", PINNED_PLAN))
    actual_units = {entry["name"]: entry["source_SHA256"] for entry in plan["units"]}
    if actual_units["u15"] != ATTENTION or actual_units["u17"] != PREPARE:
        raise ValueError("CPU308 original NoV/prepare units differ")
    headers = {digest for path, digest in plan["project_pins"].items()
               if path.endswith("/glm_moe_dsa_attention_prepare.hpp")}
    if headers != {DECLARATION}:
        raise ValueError("CPU308 AttentionPrepare header ABI differs")
    rows = [pin(PARENT / plan["source_blobs"][digest], digest) for digest in (ATTENTION, PREPARE, DECLARATION)]
    return rows


def sources():
    attention, prepare, header = originals()
    guard_header = b'#include "geometry.hpp"\n'
    modified_attention = replace(attention, guard_header, guard_header + b'#include "fresh_kv_route.hpp"\n')
    modified_attention = replace(modified_attention, OLD_FRESH, FRESH)
    modified_prepare = replace(prepare, guard_header, guard_header + b'#include "fresh_kv_state.inc"\n')
    modified_prepare = replace(modified_prepare, OLD_PACK, PACK)
    if modified_attention.count(b'prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot)') != 1:
        raise ValueError("fresh projection count changed")
    return {"NoV_tp16.cpp": modified_attention, "prepare_tp16.cpp": modified_prepare,
            "glm_moe_dsa_attention_prepare.hpp": header,
            **{name: (HERE / name).read_bytes() for name in ("fresh_kv_route.hpp", "fresh_kv_state.inc",
                                                             "fresh_kv_materialize.inc")}}


def stage(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files = sources()
    with TemporaryDirectory(prefix="glm-fresh-caller-", dir=output.parent) as temporary:
        staged = Path(temporary) / "target-source"
        staged.mkdir(mode=0o700)
        for name, raw in files.items():
            (staged / name).write_bytes(raw)
        staged.rename(output)
    return {name: sha(raw) for name, raw in files.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_output", type=Path)
    print(json.dumps(stage(parser.parse_args().new_output), indent=2))
