"""Compile the exact generated commit fragment against a recording C++ owner fixture."""

import json
import os
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / "render.py"))


def run():
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        rendered = temp / "source"
        pins = R["render"](rendered)
        assert pins == R["render"](temp / "again")
        source = (rendered / Path(R["CSA"]).name).read_bytes().decode()
        assert source.count(R["NEW"]) == 1
        original = (R["REPO"] / R["CSA"]).read_bytes()
        unpatched = R["once"](source, R["NEW"], R["OLD"])
        unpatched = R["once"](unpatched, '#include "identity_swa_commit.hpp"\n#include <cstdlib>\n', "")
        assert unpatched.encode() == original
        (temp / "derived_commit.inc").write_bytes(R["NEW"].encode() + b"\n")
        exe = temp / "check.exe"
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        "-I" + str(temp), "-I" + str(rendered), str(HERE / "check_local.cpp"), "-o", str(exe)],
                       timeout=30, check=True)
        checked = {}
        for name, value in (("unset", None), ("enabled", "1"), ("invalid", "10")):
            environment = dict(os.environ)
            if value is None:
                environment.pop(R["FLAG"], None)
            else:
                environment[R["FLAG"]] = value
            checked[name] = json.loads(subprocess.check_output([str(exe)], env=environment, timeout=15))
        assert checked == dict(unset=dict(direct=0, legacy=9), enabled=dict(direct=4, legacy=5),
                               invalid=dict(direct=0, legacy=9))
    return dict(actual_host_proxy_compiles=1, actual_host_proxy_runs=3, route_cases=checked, rendered=pins,
                exact_original_after_reversing_two_edits=True, actual_native_TU_compiles=0,
                observed_HBM_bytes=None, measured_sync_or_kernel_savings=None, actual_GPU_steps=0,
                compiler=subprocess.check_output(["g++", "--version"], text=True, timeout=5).splitlines()[0])


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
