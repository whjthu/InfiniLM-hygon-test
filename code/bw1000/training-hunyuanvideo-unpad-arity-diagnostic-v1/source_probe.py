"""Inspect exact source text only; no provider imports, tensors, downloads or dispatch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

BINDING_SHA = "26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7"
FILES = {
    "transformer": "hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py",
    "token_refiner": "hyvideo/models/transformers/modules/token_refiner.py",
    "attention": "hyvideo/models/transformers/modules/attention.py",
    "flash_helper": "hyvideo/utils/flash_attn_no_pad.py",
}


def pinned(path, expected):
    raw = path.read_bytes()
    if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Exact source SHA/extent differs")
    return raw


def functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def mode_default(tree):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attention")
    defaults = dict(zip([x.arg for x in fn.args.args][-len(fn.args.defaults):], fn.args.defaults))
    value = defaults.get("attn_mode")
    return value.value if isinstance(value, ast.Constant) else "UNKNOWN"


def tuple_sites(tree):
    rows = []
    for fn in functions(tree):
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != 5:
                continue
            value = node.value
            kind = ("shape" if isinstance(value, ast.Attribute) and value.attr == "shape" else
                    "unpad_input" if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "unpad_input" else "other")
            fifth = target.elts[4]
            loads = (sum(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == fifth.id
                         for n in ast.walk(fn)) if isinstance(fifth, ast.Name) and fifth.id != "_" else None)
            rows.append({"function": fn.name, "line": node.lineno, "RHS": kind,
                         "target_names": [n.id if isinstance(n, ast.Name) else "OTHER" for n in target.elts],
                         "fifth_variable_Load_occurrences": loads})
    return rows


def return_arities(raw):
    tree = ast.parse(raw)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unpad_input")
    lengths = []
    unknown = 0
    # Do not count returns in nested functions/classes as this function's return contract.
    def visit(node):
        nonlocal unknown
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            return
        if isinstance(node, ast.Return):
            if isinstance(node.value, (ast.Tuple, ast.List)):
                lengths.append(len(node.value.elts))
            else:
                unknown += 1
        for child in ast.iter_child_nodes(node): visit(child)
    visit(fn)
    return {"function_line": fn.lineno, "literal_return_arities": sorted(set(lengths)),
            "unknown_return_expressions": unknown, "actual_CPU_result_observed": False}


def reproduce_assignment(arity):
    # Structural Python protocol fixture only. It is not the failed GPU stack.
    statement = ast.parse("a, b, c, d, e = unpad_input(x, mask)")
    namespace = {"unpad_input": lambda *_: tuple(range(arity)), "x": None, "mask": None}
    try:
        exec(compile(statement, "<synthetic-unpad-arity-contract>", "exec"), namespace)
        return {"synthetic_arity": arity, "passed": True}
    except ValueError as error:
        return {"synthetic_arity": arity, "passed": False, "error_type": "ValueError", "message": str(error)}


def inspect_sources(binding_path, source_root, provider_path=None, provider_sha=None, *, flat=False):
    binding = json.loads(pinned(binding_path, BINDING_SHA))
    parsed, pins = {}, {}
    for role, relative in FILES.items():
        rel = PurePosixPath(relative)
        path = source_root / rel.name if flat else source_root.joinpath(*rel.parts)
        declared = binding["files"][relative]
        raw = pinned(path, declared["SHA256"])
        if len(raw) != declared["bytes"]: raise ValueError("Original source length differs")
        parsed[role] = ast.parse(raw)
        pins[role] = {"relative_path": relative, "bytes": len(raw), "SHA256": declared["SHA256"]}
    refiner = next(n for n in parsed["token_refiner"].body if isinstance(n, ast.ClassDef)
                   and n.name == "IndividualTokenRefinerBlock")
    forward = next(n for n in refiner.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "attention"]
    route = [{"line": n.lineno, "explicit_attn_mode": next((ast.unparse(k.value) for k in n.keywords
              if k.arg == "attn_mode"), None)} for n in calls]
    set_mode = next(n for n in functions(parsed["transformer"]) if n.name == "set_attn_mode")
    touched = sorted({n.attr for n in ast.walk(set_mode) if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == "self"})
    result = {"source_pins": pins, "attention_default": mode_default(parsed["attention"]),
              "text_refiner_attention_calls": route, "set_attn_mode_self_attributes": touched,
              "five_value_sites": {role: tuple_sites(tree) for role, tree in parsed.items()},
              "actual_GPU_failure_site": "UNKNOWN_NO_ORIGINAL_TRACEBACK",
              "actual_unpad_provider_result": "NOT_EXECUTED",
              "actual_training_updates": 0, "production_patch_applied": False}
    if provider_path is not None:
        if not provider_sha: raise ValueError("Independent actual provider source SHA required")
        raw = pinned(provider_path, provider_sha)
        result["installed_provider_source"] = {"path": str(provider_path), "bytes": len(raw),
                                                "SHA256": provider_sha, **return_arities(raw)}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--flat-source", action="store_true", help="Exact root-exported four-file layout")
    p.add_argument("--provider-path", type=Path)
    p.add_argument("--provider-sha256")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = inspect_sources(a.binding, a.source_root, a.provider_path, a.provider_sha256, flat=a.flat_source)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps({"passed": True, "actual_GPU_failure_site": "UNKNOWN_NO_ORIGINAL_TRACEBACK"}))
