"""Bind only the verified C4 history alias DSO to the original R454 worker."""

import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType

BASELINE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
NATIVE = "5e6793ad220e2bb6501fadb3eae70f0b5270565759f2c6907997809ef1be3537"
LIBRARY = Path("/refactor/candidates/Deep-C4-history-bank-alias-CPU-v1/artifacts/libDeep_c4_history_bank_alias_v1.so")
RECEIPT_SHA = "74b40092a766ffd72aa7a0b89f2668a37ed557afdfa7481e44d037421025c35d"
PUBLICATION = "cb77f5d02f3c0fb5277f70e6a6edc071a55b9522"
OLD = {
    "/refactor/candidates/deep-attention-inner-r9193/deepseek_v4_csa_attention_batch_r784.cpp.o":
        ("282075cc69b2358f636336235e903b55fe364ffd83093c1652deee72b8522ba4",
         "bbc6f78463ce65b8a7e8ec56def9f7c672955863a2dd1b797e9821ccf432b4db", 997),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_receipt(original):
    descriptor, parent, normalized = original()
    assert parent["library_SHA256"] == BASELINE
    assert sha(parent["library"]) == BASELINE
    path = LIBRARY.with_name("CPU-build.json")
    assert sha(path) == RECEIPT_SHA and sha(LIBRARY) == NATIVE
    receipt = json.loads(path.read_bytes())
    assert receipt["completed"] is True and receipt["actual_step"] == "1087656.70"
    assert not receipt["Native_dlopen"] and not receipt["GPU_initialized"]
    assert receipt["publication_commit"] == PUBLICATION
    assert receipt["reused"] == 49 and receipt["old_strong_exports"] == 337
    assert receipt["new_strong_exports"] == ["deep_c4_history_bank_alias_set_v1", "deep_c4_history_bank_alias_stats_v1"]
    assert receipt["dynamic_NEEDED_unchanged"] is True
    assert receipt["library"] == str(LIBRARY) and receipt["library_SHA256"] == NATIVE
    source = normalized["link"]["full"]
    objects = dict(source["objects"])
    assert len(objects) == 50 and len(receipt["replaced"]) == 1
    assert {row["parent"] for row in receipt["replaced"]} == set(OLD)
    expected_link = list(source["argv"])
    for row in receipt["replaced"]:
        name = row["parent"]
        old, new, dependencies = OLD[name]
        assert objects.pop(name) == old and row["SHA256"] == new and row["dependencies"] == dependencies
        actual = LIBRARY.with_name(Path(name).name)
        assert sha(actual) == new
        objects[str(actual)] = new
        assert expected_link.count(name) == 1
        expected_link[expected_link.index(name)] = "/build/artifacts/" + actual.name
    assert expected_link.count(parent["library"]) == 1
    expected_link[expected_link.index(parent["library"])] = "/build/artifacts/" + LIBRARY.name
    actual_link, = [row["argv"] for row in receipt["commands"] if row["label"] == "link"]
    assert actual_link == expected_link
    link = [str(LIBRARY.parent / Path(a).name) if a.startswith("/build/artifacts/") else a for a in actual_link]
    norm = copy.deepcopy(normalized)
    norm["link"]["full"] = {"objects": objects, "argv": link}
    norm["libraries"]["full"] = {"path": str(LIBRARY), "SHA256": NATIVE}
    common = dict(library=str(LIBRARY), library_SHA256=NATIVE,
                  history_alias_CPU_receipt_SHA256=RECEIPT_SHA, baseline_library_SHA256=BASELINE)
    return dict(descriptor, **common), dict(parent, **common), norm


def install():
    from public_deep_model import checkpoint_loader, cli, native_binding, native_context, worker

    original = native_binding.binding
    validator = native_binding.validate_mode_functions
    assert worker.verify_native_chain is original
    assert native_binding.NATIVE_SHA == checkpoint_loader.NATIVE_SHA == cli.NATIVE_SHA == native_context.NATIVE_SHA == BASELINE
    alternate_globals = dict(validator.__globals__, NATIVE_SHA=NATIVE,
                             PIN=dict(native_binding.PIN, library=str(LIBRARY)))
    native_binding.validate_mode_functions = FunctionType(
        validator.__code__, alternate_globals, validator.__name__, validator.__defaults__, validator.__closure__)
    def binding():
        return bind_receipt(original)
    native_binding.binding = worker.verify_native_chain = binding
    checkpoint_loader.NATIVE_SHA = native_context.NATIVE_SHA = cli.NATIVE_SHA = NATIVE
