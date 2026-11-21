"""Verify archived call-site and provider identity and exercise both host shims."""
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


D = Path(__file__).resolve().parent
REPO = D.parents[3]
LAUNCH = b"inverse_output<HeadsPerBlock><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(\n"
CALLER = "code/bw1000/Deep-PD-disaggregation-r382/native/original/deepseek_v4_attention_prepare.cpp"
SDK = "results/bw1000/2026-11-02/Deep344-exact-production-rope-source.json"
SDK_KERNEL = "src/native/cuda/ops/rotary_embedding/kernel.cuh"
PROFILE = "analysis/bw1000/Deep-r467-post1498-main-phase-decision.json"
OLDER = "analysis/bw1000/Deep-C1-1195-diagnosis-r172.json"
PINS = {
    CALLER: "3f388df12b9f8884d72cbe6d4a29f2b273934a3dd0cc49c81c93b66fe67ba56b",
    SDK: "2730c372b4fc930a3a80df398a6d318b23fe93312ce8e552083edcda51f31d8e",
    PROFILE: "7ffef1122f09ecaaf319da3f55c9e7b76d21ceaf4e086a99106a9016691c2da4",
    OLDER: "002ae8ec1ba00b835207b482621d687792f7b0d25c20c072cebbac458af90c55",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_origin():
    original = {}
    for name, pin in PINS.items():
        raw = (REPO / name).read_bytes()
        assert sha(raw) == pin, name
        original[name] = raw
    assert all(word in original[CALLER] for word in (
        b"local_heads_=64/size_t(rank.tp_size)", b"std::iota(rows.begin(),rows.end(),0)",
        b"result->copy_from(r,x);rotate(r,result,std::nullopt,pos,true)", b"int64_t{512},false,int64_t{448},inverse",
        b"cache[t*64+i]=float(std::cos(angle));cache[t*64+32+i]=float(std::sin(angle))",
    ))
    leaf = json.loads(original[SDK])["files"][SDK_KERNEL]
    assert sha(leaf["text"].encode()) == leaf["SHA256"] == (
        "c65e69e03820bf04e79e5dd76585b829ca4490103ca64f0200e1cc3dec524b74")
    assert all(marker in leaf["text"] for marker in (
        "kIsNeox ? rot_offset : 2 * rot_offset", "kIsNeox ? embed_dim + rot_offset : 2 * rot_offset + 1",
        "sin_value = -sin_value", "Cast<T>(x * cos_value - y * sin_value)",
        "Cast<T>(y * cos_value + x * sin_value)",
    ))
    return [entry["attention_inner_siblings_stream_ms"]["inverse_output_rope"]
            for entry in json.loads(original[PROFILE])["windows"]]


def run():
    windows = validate_origin()
    source = (D / "inverse_output_rope.cu").read_bytes()
    assert source.count(LAUNCH) == source.count(b"#include <cuda_runtime.h>\n") == 1
    body = source.replace(LAUNCH, b"cpu_launch(inverse_output<HeadsPerBlock>, grid, threads, 0, stream,\n")
    body = body.replace(b"#include <cuda_runtime.h>\n", b'#include "local_runtime.hpp"\n')
    assert b"<<<" not in body
    with TemporaryDirectory() as temporary:
        temp = Path(temporary)
        input_file, exe = temp / "translated.cpp", temp / "check.exe"
        input_file.write_bytes(body)
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        "-ffp-contract=off", "-I" + str(D), str(input_file),
                        str(D / "check_local.cpp"), "-o", str(exe)], timeout=30, check=True)
        report = json.loads(subprocess.check_output([str(exe)], timeout=25))
    assert report["shapes"] == 6 and report["cases"] == 36 and report["invalid_no_launch"] == 22
    assert report["rotated_exact"] * 7 == report["prefix_exact"]
    return dict(report, actual_host_compiles=1, actual_host_runs=1, tested_source_SHA256=sha(source),
                compiler=subprocess.check_output(["g++", "--version"], timeout=4, text=True).splitlines()[0],
                measured_historic_inverse_enclosure_stream_ms=windows,
                actual_gfx936_compiles=0, actual_gpu_steps=0)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
