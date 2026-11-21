"""Compile the actual patched CPU308 fresh call and expand_cached body with a recorder."""
import json
import os
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
RUN_ENV = "INFINILM_GLM474_FRESH_KV_PACK"


def run():
    generator = runpy.run_path(str(HERE / "render.py"))
    base_attention, base_prepare, base_declaration = generator["originals"]()
    with TemporaryDirectory() as folder:
        root = Path(folder)
        generated = root / "generated"
        pins = generator["stage"](generated)
        original_stage = generator["stage"](root / "second")
        assert pins == original_stage
        nova = (generated / "NoV_tp16.cpp").read_bytes()
        prepare = (generated / "prepare_tp16.cpp").read_bytes()
        assert nova.replace(generator["FRESH"], generator["OLD_FRESH"]).replace(
            b'#include "fresh_kv_route.hpp"\n', b"") == base_attention
        assert prepare.replace(generator["PACK"], generator["OLD_PACK"]).replace(
            b'#include "fresh_kv_state.inc"\n', b"") == base_prepare
        assert (generated / "glm_moe_dsa_attention_prepare.hpp").read_bytes() == base_declaration
        fallback = (generated / "fresh_kv_materialize.inc").read_bytes().split(b"} else {", 1)[-1]
        for line in generator["OLD_PACK"].splitlines():
            text = line.strip().removeprefix(b"auto ")
            assert text in fallback
        begin = prepare.index(b"std::pair<rt::Tensor,rt::Tensor> AttentionPrepare::expand_cached(")
        end = prepare.index(b"rt::Tensor AttentionPrepare::expand_cached_values(")
        actual = prepare[begin:end]
        assert actual.count(b"kb_->forward(r,latent)") == actual.count(b"r.syncStream()") == 1
        assert actual.index(b"kb_->forward(r,latent)") < actual.index(b'fresh_kv_materialize.inc')
        (generated / "actual_prepare.inc").write_bytes(actual)
        (generated / "actual_fresh_call.inc").write_bytes(generator["FRESH"] + b"\n")
        (generated / "state.cpp").write_bytes(b'#include "fresh_kv_state.inc"\n')
        exe = root / "record.exe"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
             "-I" + str(generated), str(generated / "state.cpp"), str(HERE / "check_local.cpp"), "-o", str(exe)],
            check=True, timeout=35)
        tested = {}
        for env_value, expected in ((None, 0), ("0", 0), ("064", 0), ("64", 64), ("128", 128)):
            env = os.environ.copy()
            env.pop(RUN_ENV, None)
            if env_value is not None:
                env[RUN_ENV] = env_value
            row = json.loads(subprocess.check_output([str(exe), str(expected)], env=env, timeout=8))
            assert row == ({"fallback": 11, "candidate": 1, "errors": 3} if expected else
                           {"fallback": 12, "candidate": 0, "errors": 1})
            tested["unset" if env_value is None else env_value] = row
    return dict(generated_sources_SHA256=pins, settings=tested, host_CPP_builds=1, host_TUs=2, host_processes=5,
                original_fallback_routes=sum(row["fallback"] for row in tested.values()),
                mock_cabi_routes=sum(row["candidate"] for row in tested.values()),
                validated_error_paths=sum(row["errors"] for row in tested.values()),
                actual_native_compiles=0, actual_checkpoint_bytes=0, actual_GPU_steps=0)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
