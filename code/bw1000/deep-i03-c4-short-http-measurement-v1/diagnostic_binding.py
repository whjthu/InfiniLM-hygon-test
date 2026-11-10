"""Bind only the verified short C4 measurement DSO to the original R454 worker."""

import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType

BASELINE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
NATIVE = "95625a4f8a87ab588c06611597bd320e2b96cabf5e4928e7b783b1b59aae1d8e"
LIBRARY = Path("/refactor/candidates/Deep-C4-short-host-split-CPU-v1/artifacts/libDeep_c4_short_host_split_v1.so")
RECEIPT_SHA = "b6e666faa2d06970bfe6fc811fabce3e29c23fb82563a8ca9e3dcf5de7d94ff3"
PUBLICATION = "7f27cff84c7e69642379d1445c20f62fdd967df4"
OLD = {
    "/refactor/candidates/deep-attention-inner-r9193/deepseek_v4_csa_attention_batch_r784.cpp.o":
        ("282075cc69b2358f636336235e903b55fe364ffd83093c1652deee72b8522ba4",
         "a433619794be24665e5923d8fadefa5d1a449327bcc88e22b48c385e0ca76df6", 997),
    "/refactor/candidates/deep-c4-smallt-r9252/deep_c4_smallt_r9252.cpp.o":
        ("3ea73a4acea21b2bf7cd74a55ff6b0616b995802cf915227bd939adb4a898156",
         "efebf0b88153a68a126b863937e68eadbd483152de42234785f5097865c4b27d", 992),
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
    assert receipt["completed"] is True and receipt["actual_step"] == "1087656.65"
    assert not receipt["Native_dlopen"] and not receipt["GPU_initialized"]
    assert receipt["publication_commit"] == PUBLICATION
    assert receipt["reused"] == 48 and receipt["old_strong_exports"] == 337
    assert receipt["new_strong_exports"] == ["deep_c4_short_host_split_read_v1"]
    assert receipt["dynamic_NEEDED_unchanged"] is True
    assert receipt["library"] == str(LIBRARY) and receipt["library_SHA256"] == NATIVE
    source = normalized["link"]["full"]
    objects = dict(source["objects"])
    assert len(objects) == 50 and len(receipt["replaced"]) == 2
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
                  short_c4_CPU_receipt_SHA256=RECEIPT_SHA, baseline_library_SHA256=BASELINE)
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
