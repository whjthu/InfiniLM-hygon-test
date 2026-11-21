"""Source-bound four/five-field ABI compatibility; never changes attention math."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from importlib.machinery import PathFinder
import sys

from contract import NATIVE, NATIVE_ALIAS, ROOT, RUN, load, need, read, save, stamp

ORIGINAL_BYTES = 3909
ORIGINAL_SHA = "e9835a76bb2908a1160fec435a9b2e230b6db0588199ea48e6ae66309b2b5b67"
RELATIVE = "hyvideo/utils/flash_attn_no_pad.py"
PROVIDER_BYTES = 9535
PROVIDER_SHA = "31832efc383d01c9cce3e0f9e97d101b1a795a27aa274e39440bc2e243c5239c"
CPU_RECEIPT = ROOT / "results/training-hunyuanvideo-unpad-provider-CPU-v1/provider-result-private.json"
CPU_RECEIPT_BYTES = 1594
CPU_RECEIPT_SHA = "63abd0953abed4f895315fa7bfc9227e205458cbe3e510fc398ad4d6aceb04d1"
BACKING = RUN / "overlay/flash_attn_no_pad.py"
ALIAS = NATIVE_ALIAS / RELATIVE
HELPER = "_t09_unpad_first_four"
HELPER_SOURCE = '''def _t09_unpad_first_four(values):
    if type(values) not in (tuple, list) or len(values) not in (4, 5):
        raise ValueError("T09 unpad_input requires exactly four or five fields")
    return values[0], values[1], values[2], values[3]


'''


def assignments(tree, wrapped=False):
    rows = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name == HELPER:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            call = node.value
            name = HELPER if wrapped else "unpad_input"
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name:
                rows.append((fn, node))
    return rows


def patch_bytes(raw):
    need(len(raw) == ORIGINAL_BYTES and hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA,
         "Original exact flash helper differs; no patch fallback")
    need(HELPER.encode() not in raw, "Compatibility helper name already present")
    tree = ast.parse(raw)
    sites = assignments(tree)
    need(len(sites) == 4 and [fn.name for fn, _ in sites] ==
         ["flash_attn_no_pad", "flash_attn_no_pad_v3", "flash_attn_no_pad_v3", "flash_attn_no_pad_v3"],
         "Original four unpack sites differ")
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    def pos(line, column): return offsets[line - 1] + column
    edits = []
    for fn, node in sites:
        target = node.targets[0]
        need(isinstance(target, ast.Tuple) and len(target.elts) == 5
             and all(isinstance(x, ast.Name) for x in target.elts), "Original five-field tuple differs")
        fifth = target.elts[4].id
        need(not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == fifth
                     for n in ast.walk(fn)), "Original fifth field is used; compatibility not authorized")
        edits.append((pos(target.lineno, target.col_offset), pos(target.end_lineno, target.end_col_offset),
                      ", ".join(x.id for x in target.elts[:4]).encode()))
        call = node.value
        edits.append((pos(call.lineno, call.col_offset), pos(call.lineno, call.col_offset), (HELPER + "(").encode()))
        edits.append((pos(call.end_lineno, call.end_col_offset), pos(call.end_lineno, call.end_col_offset), b")"))
    first = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    edits.append((pos(first.lineno, 0), pos(first.lineno, 0), HELPER_SOURCE.encode()))
    patched = raw
    for start, end, value in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        patched = patched[:start] + value + patched[end:]
    # Inverse AST must restore the entire original module, not merely a few lines.
    after = ast.parse(patched)
    wrapped = assignments(after, wrapped=True)
    need(len(wrapped) == 4, "Generated wrapper site count differs")
    inverse = copy.deepcopy(after)
    inverse.body = [n for n in inverse.body if not isinstance(n, ast.FunctionDef) or n.name != HELPER]
    for (_fn, changed), (_orig_fn, original) in zip(assignments(inverse, wrapped=True), sites):
        need(len(changed.value.args) == 1 and not changed.value.keywords
             and len(changed.targets[0].elts) == 4, "Unexpected compatibility AST")
        changed.value = copy.deepcopy(changed.value.args[0])
        changed.targets[0].elts.append(copy.deepcopy(original.targets[0].elts[4]))
    need(ast.dump(inverse, include_attributes=False) == ast.dump(tree, include_attributes=False),
         "Inverse patch changed original source math")
    compile(patched, "<source-bound-T09-unpad-compat>", "exec")
    return patched


def descriptor():
    spec, _ = load(Path(RUN) / "src/COMPATIBILITY.json")
    need(spec["original"] == {"bytes": ORIGINAL_BYTES, "SHA256": ORIGINAL_SHA}, "Original patch contract differs")
    return spec


def prepare_overlay():
    _cpu, cpu_pin = load_cpu_receipt()
    original_path = Path(NATIVE) / RELATIVE
    before = stamp(original_path.lstat())
    raw, original_pin = read(original_path, ORIGINAL_BYTES)
    patched = patch_bytes(raw)
    spec = descriptor()
    need({"bytes": len(patched), "SHA256": hashlib.sha256(patched).hexdigest()} == spec["patched"],
         "Generated compatibility file identity differs")
    folder = Path(BACKING).parent
    folder.mkdir()
    with Path(BACKING).open("xb") as stream:
        stream.write(patched); stream.flush(); os.fsync(stream.fileno())
    os.chmod(BACKING, 0o444)
    need(stamp(original_path.lstat()) == before, "Original source changed during overlay preparation")
    result = {"original_source": original_pin, "original_source_untouched": True,
              "provider_CPU_receipt": cpu_pin,
              "overlay": {"path": str(BACKING), **spec["patched"]},
              "bind_target": str(ALIAS), "bind_mode": "ro", "inverse_AST_exact": True}
    save(Path(RUN) / "compatibility-prepared.json", result)
    return result


def load_cpu_receipt():
    receipt, pin = load(CPU_RECEIPT, 16384)
    need(pin["bytes"] == CPU_RECEIPT_BYTES and pin["SHA256"] == CPU_RECEIPT_SHA,
         "Actual .791 provider CPU receipt differs")
    need(receipt["passed"] is True and receipt["actual_return_arity"] == 4 and receipt["finite"] is True
         and receipt["padded_roundtrip_shape"] == [2, 3, 4]
         and all(receipt["first_four_semantic_identities"][name] is True for name in
                 ("unpadded_values", "indices", "cu_seqlens", "max_seqlen", "padded_roundtrip")),
         "Actual CPU first-four semantic proof incomplete")
    provider = receipt["actual_current_provider"]
    need(provider["bytes"] == PROVIDER_BYTES and provider["SHA256"] == PROVIDER_SHA,
         "Actual .791 provider source pin differs")
    return receipt, pin


def validate_execution_compatibility():
    receipt, pin = load_cpu_receipt()
    provider = receipt["actual_current_provider"]
    package = PathFinder.find_spec("flash_attn", sys.path)
    need(package is not None and package.submodule_search_locations, "Active provider package missing")
    active = PathFinder.find_spec("flash_attn.bert_padding", list(package.submodule_search_locations))
    need(active is not None and active.origin and Path(active.origin).resolve() == Path(provider["path"]).resolve(),
         "Current native provider resolution differs from actual CPU proof")
    _, actual = read(Path(active.origin), PROVIDER_BYTES)
    need(actual["bytes"] == PROVIDER_BYTES and actual["SHA256"] == PROVIDER_SHA, "Current provider bytes changed")
    spec = descriptor()
    raw, mounted = read(ALIAS, spec["patched"]["bytes"])
    _, backing = read(BACKING, spec["patched"]["bytes"])
    need(all(mounted[k] == spec["patched"][k] == backing[k] for k in ("bytes", "SHA256")),
         "Actual single-file overlay is not the frozen derived file")
    need(stamp(Path(ALIAS).lstat()) == stamp(Path(BACKING).lstat()), "Overlay alias and backing are different files")
    need(os.statvfs(ALIAS).f_flag & os.ST_RDONLY, "Compatibility file bind is not read-only")
    compile(raw, str(ALIAS), "exec")  # Actual container Python compile, without importing attention or models.
    _, original = read(Path(NATIVE) / RELATIVE, ORIGINAL_BYTES)
    need(original["SHA256"] == ORIGINAL_SHA, "Canonical original source was mutated")
    return {"provider_CPU_receipt": pin, "provider_current_source": actual,
            "overlay": mounted, "canonical_original_source": original,
            "read_only_bind_verified": True, "container_compile_passed": True,
            "actual_provider_function_replayed": False, "attention_math_changed": False}
