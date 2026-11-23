"""Freeze the exact Slurm-record fix and a new zero-GRES request."""

import argparse
from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.with_name("zimage-lt-eight-card-host-preflight-v1") / "preflight.py"
BASE_SHA = "a67ba785b2004fc375b387d09527c7ee638e60cbd266dce949ba9f6a84b3066d"
OLD_REQUEST = "00ba290bf5844476a5b84ceaa9930f5c"
OLD_LABEL = "O01-zimage-hybrid-host-none-00ba290bf584"
REQUEST = "4c03407a22424a34a5a0c3f73900acf1"
LABEL = "O01-zimage-hybrid-host-none-4c03407a2242"


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen source anchor differs")
    return text.replace(old, new)


def render():
    base = BASE.read_bytes()
    if sha256(base).hexdigest() != BASE_SHA:
        raise ValueError("previous published source differs")
    source = replace(base.decode("utf-8"), OLD_REQUEST, REQUEST)
    source = replace(source, OLD_LABEL, LABEL)
    source = replace(source, "def mount(path):", "def exact_record(text, key, identity):\n"
                     "    matches = [row for line in text.splitlines() if (row := fields(line)).get(key) == identity]\n"
                     "    if len(matches) != 1:\n"
                     "        raise RuntimeError(f'expected exactly one Slurm record for {identity}')\n"
                     "    return matches[0]\n\n\n"
                     "def mount(path):")
    source = replace(source, 'job = fields(query("scontrol", "show", "job", PARENT, "--oneliner"))',
                     'job = exact_record(query("scontrol", "show", "job", PARENT, "--oneliner"), "JobId", PARENT)')
    source = replace(source, 'standby = fields(query("scontrol", "show", "step", STANDBY, "--oneliner"))',
                     'standby = exact_record(query("scontrol", "show", "step", STANDBY, "--oneliner"),'
                     ' "StepId", STANDBY)')
    source = replace(source, '    ROOT.mkdir(mode=0o700)\n',
                     f'    old = ROOT.with_name({OLD_LABEL!r})\n'
                     '    if old.exists() or old.is_symlink():\n'
                     '        raise RuntimeError("original result needs reconciliation")\n'
                     '    raw = query("sacct", "-nP", "-j", PARENT, "--format", "JobIDRaw,JobName%100")\n'
                     f'    if any(row.endswith("|" + label) for row in raw.splitlines() for label in ({OLD_LABEL!r}, LABEL)):\n'
                     '        raise RuntimeError("original or current accounting label already exists")\n'
                     '    ROOT.mkdir(mode=0o700)\n')
    compile(source, "preflight.py", "exec")
    return source.encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("xb") as stream:
        stream.write(render())


if __name__ == "__main__":
    main()
