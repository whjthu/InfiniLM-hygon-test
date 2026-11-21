"""Exercise the new caller route and compare its table with the frozen original calculation."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from render_caller import PARENT_SHA, render


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = REPO / "code/bw1000/Deep-PD-disaggregation-r382/native/original"


def main():
    file = PARENT / "deepseek_v4_attention_prepare.cpp"
    parent = file.read_bytes()
    header = file.with_suffix(".hpp").read_bytes()
    result = render(parent, header)
    base = parent.decode("utf-8").replace("\r\n", "\n")
    orig = base[base.index(" constexpr double pi2"):base.index(" Work work{r};auto table=")]
    modified = result.decode("utf-8").replace("\r\n", "\n")
    a = base.index("AttentionPrepare::Output AttentionPrepare::forward(")
    b = base.index("rt::Tensor AttentionPrepare::inverse_output_rope(")
    assert base[a:b] in modified
    orig_constructor = base[base.index("AttentionPrepare::AttentionPrepare("):base.index("void AttentionPrepare::rotate(")]
    assert orig_constructor in modified
    assert modified.count("result->copy_from(r, x);") == modified.count("r.memcpyD2H(") == 1
    assert modified.index("auto cache = output_rope_table", modified.index("static const auto requested")) < modified.index(
        "Work enqueue{r};") < modified.index("r.memcpyH2D(table->data(), cache.data(), cache.size() * sizeof(float), false);")
    assert modified.count("int64_t{512},false,int64_t{448},inverse") == 1
    try:
        render(parent + b"\n", header)
    except ValueError:
        changed_source_rejected = True
    else:
        raise AssertionError("Changed original caller accepted")
    reference = (
        "#include <algorithm>\n#include <cstddef>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n"
        "std::vector<float> reference(const std::vector<std::int64_t>& positions,"
        "double theta_, double factor_, double beta_fast_, double beta_slow_, std::size_t original_) {\n"
        "using std::size_t;\nconst std::size_t tokens = positions.size();\n" + orig + "\nreturn cache;\n}\n"
    )
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Local C++ compiler not available")
    with TemporaryDirectory(prefix="deep-inverse-rope-caller-") as temp:
        directory = Path(temp)
        source = directory / "original_table.cpp"
        source.write_text(reference, encoding="utf-8")
        executable = directory / "check_policy.exe"
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        "-ffp-contract=off", "-I", str(HERE), str(source), str(HERE / "check_policy.cpp"),
                        "-o", str(executable)], check=True, timeout=30)
        receipt = json.loads(subprocess.check_output([str(executable)], timeout=10))
    ast.parse(Path(__file__).with_name("render_caller.py").read_bytes(), feature_version=(3, 10))
    print(json.dumps(dict(receipt, archived_caller_SHA256=PARENT_SHA,
                          rendered_caller_SHA256=sha256(result).hexdigest(),
                          mutated_parent_rejected=changed_source_rejected, actual_target_cpp_builds=0,
                          actual_HIP_steps=0, actual_model_trials=0)))


if __name__ == "__main__":
    main()
