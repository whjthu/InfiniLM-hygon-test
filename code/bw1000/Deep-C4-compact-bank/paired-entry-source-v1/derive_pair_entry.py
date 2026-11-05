"""Build the source-only DeepSeek C4 paired entry from the exact audited package."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

ORIGINAL = "72a6896246aee0cc37d62429ca4444148ee56108532052db49d4b0228d63bac0"
NATIVE = "7fcea960c5f218234ed8a4733fa7bd94b31055ff7396755c2afd6f5bba502f90"
CPU_RECEIPT = "710e6046deb71019f7e5e015206d08c630100b9999f9cc884307903e50e9bf38"
LIBRARY = "/refactor/candidates/Deep-C4-compact-bank-CPU-private-state-v2/libDeep_c4_compact_bank_v1.so"
SCHEDULE = [False, True, False, True, True, False]

BINDING = r'''
"""Wrap the original a314 and split receipts with the checked one-object candidate."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path

from split_binding import receipt_from_worker as split_receipt

NATIVE = "{native}"
LIBRARY = Path("{library}")
CPU_RECEIPT = "{receipt}"


def receipt_from_worker(worker):
    split = split_receipt(worker)
    raw = LIBRARY.with_name("CPU-build.json").read_bytes()
    assert sha256(raw).hexdigest() == CPU_RECEIPT
    proof = json.loads(raw)
    assert proof["completed"] and proof["actual_step"] == "910374.1465"
    assert not any(proof[k] for k in ("GPU", "Model", "dlopen"))
    assert proof["replaced_objects"] == 1 and proof["retained_objects"] == 49
    assert proof["dependency_paths"] == 997
    assert proof["new_exports"] == ["deep_c4_compact_bank_set_v1", "deep_c4_compact_bank_stats_v1"]
    assert proof["library"] == str(LIBRARY) and proof["library_SHA256"] == NATIVE
    return dict(split, library=str(LIBRARY), library_SHA256=NATIVE,
                split_library_SHA256=split["library_SHA256"], compact_CPU_receipt_SHA256=CPU_RECEIPT)


def bind_library(path, expected):
    from public_deep_model.checkpoint_loader import declare

    assert Path(path) == LIBRARY and expected == NATIVE
    assert sha256(LIBRARY.read_bytes()).hexdigest() == NATIVE
    return declare(ctypes.CDLL(str(LIBRARY)))
'''


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def substitute(source, before, after):
    if source.count(before) != 1:
        raise ValueError("missing or ambiguous paired-entry anchor")
    return source.replace(before, after)


def derive(parent, local):
    raw = (parent / "SOURCE_MANIFEST.json").read_bytes()
    if sha256(raw).hexdigest() != ORIGINAL:
        raise ValueError("original audited C4 source manifest changed")
    manifest = json.loads(raw)
    assert not any(manifest[k] for k in ("CPU_GO", "GPU_GO", "Model_GO"))
    files = {}
    for name, expected in manifest["files_SHA256"].items():
        path = (parent / name).resolve(strict=True)
        assert path.is_relative_to(parent.resolve()) and sha256(path.read_bytes()).hexdigest() == expected
        files[name] = path.read_bytes()
    entry = files["entry.py"].decode()
    entry = substitute(entry, "from profile_control import run_loaded,CapiTimer",
                       "from profile_control import CapiTimer\nfrom pair_control import run_loaded")
    entry = substitute(entry,
                       "c['requests']==5 and c['observer_schedule']==[False,False,False,True,False]and c['samples']==[11,12]",
                       "c['requests']==6 and c['compact_schedule']==[False,True,False,True,True,False]")
    entry = substitute(entry,
                       "  from existing_profile_consumer import bind_inner\n  bind_inner(binding.lib);report['profile_actual_providers']",
                       "  report['profile_actual_providers']")
    entry = substitute(entry,
                       "'deep_owner_read_r8900','deep_attention_inner_arm_r9193','deep_attention_inner_read_r9193','deep_c4_host_split_read_v1'",
                       "'deep_attention_inner_arm_r9193','deep_c4_compact_bank_set_v1','deep_c4_compact_bank_stats_v1'")
    entry = substitute(entry, "report['phase']='five_continuations'", "report['phase']='six_compact_continuations'")
    entry = substitute(entry, "run_loaded(context,prompt,out/'scalar-events',report,save,", "run_loaded(context,prompt,report,save,")
    files["entry.py"] = entry.encode()
    files["split_binding.py"] = files["candidate_binding.py"]
    files["candidate_binding.py"] = BINDING.format(native=NATIVE, library=LIBRARY, receipt=CPU_RECEIPT).lstrip().encode()
    files["pair_control.py"] = (local / "pair_control.py").read_bytes()
    cfg = json.loads(files["config.template.json"])
    assert not cfg["execute_authorized"] and cfg["actual_step"] is None and cfg["gpu_start_epoch"] is None
    del cfg["observer_schedule"]
    del cfg["samples"]
    cfg.update(Native_library=LIBRARY, Native_SHA256=NATIVE, requests=6, compact_schedule=SCHEDULE,
               output_root="/refactor/results/Deep-C4-compact-bank-pair/current-v1",
               run="c4-compact-bank-pair-a")
    files["config.template.json"] = encode(cfg)
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    files["SOURCE_MANIFEST.json"] = encode(dict(files_SHA256={n: sha256(b).hexdigest() for n,b in files.items()},
        parent_source_manifest_sha256=ORIGINAL, compact_cpu_receipt_sha256=CPU_RECEIPT,
        CPU_GO=False, GPU_GO=False, Model_GO=False))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    files = derive(args.parent, Path(__file__).resolve().parent)
    args.output.mkdir()
    for name, raw in files.items():
        with (args.output / name).open("xb") as stream:
            stream.write(raw)


if __name__ == "__main__":
    main()
