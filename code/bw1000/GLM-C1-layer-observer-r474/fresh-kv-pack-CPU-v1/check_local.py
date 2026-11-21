"""Verify CPU308/GLM310 source ancestry and run both TP16 host kernel shims."""
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CPU = HERE.parent / "depth-stratified-observer-CPU-v1"
CPU_PLAN = "48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6"
PREPARE = "35fec2e8202395423e4937857bc41dc6a5ea00371aed7c3233209a9102f1afcb"
ATTENTION = "6548aacdf94d0ab481cc27f613fc4713906795a0b31cdfc73aa5e22430f63f29"
NATIVE = b"ddfeae81526758fb06d68e4f688e9eeaa48e3b173018d0040a3de0c99ba8dfb5"
PROFILE = REPO / "results/bw1000/2026-11-20/GLM474-original310-live-depth-48-rank-frames-and-call-budget.json"
TERMINAL = REPO / "results/bw1000/2026-11-20/GLM474-original310-terminal-TP16-full-paired-screen.json"
BUILD = REPO / "results/bw1000/2026-11-20/GLM474-depth-observer-original-CPU308-target-build-and-independent-audit.json"
SHIM = REPO / "code/bw1000/Deep-C4-causal-consumer-r483/inverse-output-rope-CPU-v1/local_runtime.hpp"
LAUNCH = b"pack<PairsPerBlock><<<grid, PairsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>("


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, pin):
    raw = path.read_bytes()
    assert sha(raw) == pin, str(path)
    return raw


def origin():
    plan = json.loads(checked(CPU / "PLAN.json", CPU_PLAN))
    for index, pin in ((15, ATTENTION), (17, PREPARE)):
        assert plan["units"][index]["source_SHA256"] == pin
    prepare = checked(CPU / plan["source_blobs"][PREPARE], PREPARE)
    attention = checked(CPU / plan["source_blobs"][ATTENTION], ATTENTION)
    assert all(line in prepare for line in (
        b"heads_=64/size_t(rank.tp_size)",
        b'kb_=bind_child("kv_b_proj",std::make_shared<Fp8Projection>(r,28672,512,rank.tp_size,rank.tp_rank))',
        b"kb_->forward(r,latent)->view({tokens,heads_,448})",
        b"key->narrow({{2,0,192}})->copy_from(r,expanded->narrow({{2,0,192}}))",
        b"head<heads_;++head)key->narrow({{1,head,1},{2,192,64}})->copy_from(r,rotary)",
        b"expanded->narrow({{2,192,256}})->contiguous(r)",
    ))
    event = attention.index(b'attnd1.fresh.expand_cached.begin"')
    assert b"prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot)" in attention[event:event + 420]
    assert NATIVE in checked(BUILD, "5ff55c2829f889f5252795d40f20cd5a82ef75ba5b770a0aa11315c6495e13ca")
    checked(TERMINAL, "014dd7113f43c4e43293ce19189eb1756d42741c118b1fab872fb0d03216dd9e")
    obs = json.loads(checked(PROFILE, "a4201149e076c1195e1e0e66718073f72ae816ab96c6858b524ad2a3a6055ad7"))
    assert obs["actual_accounting"][0] == "1103606.310" and "true AR tokens" in obs["scope"]
    return {key: value["windows"]["attention.fresh.expand_cached"]["stream"]
            for key, value in obs["detailed_layers"].items()}


def run():
    stream_windows = origin()
    assert set(stream_windows) == {"18", "19", "38", "40", "66", "69"}
    source = (HERE / "fresh_kv_pack.cu").read_bytes()
    assert source.count(LAUNCH) == source.count(b"#include <cuda_runtime.h>\n") == 1
    local = source.replace(b"#include <cuda_runtime.h>\n", b'#include "local_runtime.hpp"\n').replace(
        LAUNCH, b"cpu_launch(pack<PairsPerBlock>, grid, PairsPerBlock, 0, stream,")
    assert b"<<<" not in local
    shim = checked(SHIM, "c5141170b13aa247fae064fc58610477585164d6ea434f39116b5c4357141a48")
    with TemporaryDirectory() as folder:
        temp = Path(folder)
        (temp / "local_runtime.hpp").write_bytes(shim)
        src, exe = temp / "fresh_kv.cpp", temp / "check.exe"
        src.write_bytes(local)
        flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic"]
        subprocess.run(["g++", *flags, "-I" + str(temp), str(src), str(HERE / "check_local.cpp"), "-o", str(exe)],
                       check=True, timeout=30)
        row = json.loads(subprocess.check_output([str(exe)], timeout=12))
    assert row["accepted"] == 24 and row["rejected_no_launch"] == 28 and row["launch_errors_propagated"] == 2
    assert row["exact_key_words"] == row["exact_value_words"] == (1 + 2 + 17 + 512) * 4 * 256 * 6
    assert 4 * row["exact_rotary_key_words"] == row["exact_key_words"]
    return dict(row, baseline_native_SHA256=NATIVE.decode(), source_SHA256=sha(source),
                profile_scope="current real GLM310 fresh+FP8projection+copy+synchronization interval",
                original_fresh_expand_stream_windows_ms=stream_windows,
                host_compiles=1, host_runs=1, actual_target_compiles=0, actual_GPU_steps=0)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
