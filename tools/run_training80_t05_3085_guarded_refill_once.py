"""Bind the reviewed single-node renewal to the completed T05 worker."""

from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve()
ORIGINAL = HERE.with_name("run_training80_t07_3081_guarded_refill_once.py")
ORIGINAL_SHA = "b0fcc5ec4992f20580bd8f2508c4ff3ffa4ff102e77964f614f5dfb7b7d488ae"
SCOPE = {
    "training80-t07-3081-guarded-bf16-v1": "training80-t05-3085-guarded-bf16-v1",
    "training80-T07-original3081-terminal-dual-source-20261125.json": "training80-T05-original3085-terminal-dual-source-20261125.json",
    "f13r2n07": "f13r1n19",
    '"1103606.3081", "1103606.3080"': '"1103606.3085", "1103606.3084"',
    "b3ce4e40bc664e32ab74ce203cb8eeff": "7648acf635794540b0b0c7c9837fa960",
    "df982d1cc29d4ab39515b752eea2c164": "e016a011a6d14f16a7cf3dc2fce96b5f",
    "82a8c53449660601600c4bd01b3ae4d59eedb142ddb1e04f643f067ff829198d": "a08324bfdd0accf168277ddaf8871d5812b340e863b1f3cf912f1779a2cc8fe5",
    "04924af56c5e654bf7ad652031d7fb7153660f63d4794fe79c93b7fd4ab59494": "029e3e8083ad2fd75b38e53d77cbd57bf6f7b08d88e6fc3a42704d700ae81aef",
    "06503f4598333761a6638d110780c2f248ede71f35e14d0bdcb2878aa3d8a045": "4652f70605b9d275df3ef778b6f470bd1a8af3db156bc2572b36253b4ee31fa2",
    "cf6f6c226bfb1c1e6bde663b085098acc2a282a6f4341a192b9aea722f72ffec": "5141a9680dca9a942c2f3593777d2e9477240b4af664d018652242d25ccc7c6d",
    'e["seconds"] == 10560': 'e["seconds"] == 10506',
}


def main():
    raw = ORIGINAL.read_bytes()
    if ORIGINAL.resolve(strict=True) != ORIGINAL or sha256(raw).hexdigest() != ORIGINAL_SHA:
        raise ValueError("frozen guarded renewal changed")
    if len(sys.argv) == 3 and sys.argv[1] == "--publication":
        revision = sys.argv[2]
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("exact publication required")
        repo = HERE.parents[1]
        ref = revision + ":" + ORIGINAL.relative_to(repo).as_posix()
        git = ["git", "--no-optional-locks", "-c", "safe.directory=" + repo.as_posix()]
        if subprocess.check_output([*git, "cat-file", "blob", ref], cwd=repo, timeout=10) != raw:
            raise ValueError("published guarded renewal differs")
    source = raw.decode()
    for old, new in SCOPE.items():
        if source.count(old) != 1:
            raise ValueError("fixed T05 scope substitution changed")
        source = source.replace(old, new)
    exec(compile(source.replace("T07", "T05"), str(ORIGINAL), "exec"), {"__name__": "__main__", "__file__": str(HERE)})


if __name__ == "__main__":
    main()
