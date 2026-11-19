"""Compile the edited Kimi method against CPU recording tensors and check the source scope."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
START = "infinilm::runtime::Tensor KimiK3TextModel::apply_output_attn_res("
END = "\ninfinilm::runtime::Tensor KimiK3TextModel::forward("


def main(compiler):
    old = (HERE.parent / "native/kimi_k3_text_model.cpp").read_bytes()
    assert hashlib.sha256(old).hexdigest() == "914888f78997b3af1b23f265cc158f68d8eaf3a6fe24f19bd068681c48a3430a"
    new = (HERE / "kimi_k3_text_model.cpp").read_text(encoding="utf-8")
    baseline = old.decode()
    include = '#include "output_residual_tiles.hpp"\n'
    assert new[:new.index(START)].replace(include, "", 1) == baseline[:baseline.index(START)]
    assert new[new.index(END):] == baseline[baseline.index(END):]
    softmax = "    infinilm::execution::submit<infini::ops::Softmax>("
    assert new[new.index(softmax):new.index(END)] == baseline[baseline.index(softmax):baseline.index(END)]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "actual_method.inc").write_text(new[new.index(START):new.index(END)], encoding="utf-8")
        program = output / "check_residual_tiles.exe"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O2",
                        "-I", str(output), str(HERE / "check_residual_tiles.cpp"), "-o", str(program)],
                       check=True, timeout=45)
        subprocess.run([str(program)], check=True, timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="g++")
    main(parser.parse_args().compiler)
