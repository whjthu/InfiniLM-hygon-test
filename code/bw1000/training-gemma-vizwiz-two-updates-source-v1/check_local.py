"""Check real VizWiz source bytes and the renderer without native dependencies."""

from __future__ import annotations

import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import prepare


ROOT = Path(__file__).resolve().parents[3]
SEVEN = ROOT / "code/bw1000/training-gemma-vizwiz-reviewed-train-stage-v1"
FIVE = ROOT / "local-only/specialized06-vizwiz-native-target-binding-v1/train-swift.jsonl"
BASELINE = ROOT / "code/bw1000/training-gemma-usgs72-fresh-initial-domain25-v1"
RELEASE = ROOT / "local-only/specialized06-vizwiz-native-cpu-pilot-8-image-release-v1.json"


def seven_rows():
    spec = importlib.util.spec_from_file_location("published_vizwiz_stage7", SEVEN / "launch_once.py")
    builder = importlib.util.module_from_spec(spec)
    sys.modules.pop("receive_once", None)
    sys.path.insert(0, str(SEVEN))
    try:
        spec.loader.exec_module(builder)
    finally:
        sys.path.pop(0)
        sys.modules.pop("receive_once", None)
    claim = json.loads((SEVEN / "claim.json").read_bytes())
    pins, body = builder.prepare_packet(RELEASE, claim["quality_receipt_SHA256"])
    assert all(claim[key] == value for key, value in pins.items())
    start = claim["binding"]["bytes"]
    count = claim["rows"]["bytes"]
    return body[start:start + count]


def must_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("mutated frozen input was accepted")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected one absent private receipt path")
    five = FIVE.read_bytes()
    seven = seven_rows()
    view, lineage = prepare.build_view(five, seven)
    assert prepare.build_view(five, seven) == (view, lineage)
    assert len(view.splitlines()) == 800 and len(set(view.splitlines())) == 12
    assert lineage["published_scope_SHA256"] == prepare.APPROVED_SCOPE_SHA
    assert lineage["published_view_receipt"]["view_SHA256"] == prepare.APPROVED_VIEW_SHA == prepare.digest(view)
    derived = prepare.lineage_bytes(lineage)
    denoms = Counter(lineage["ordered_source_line_SHA256"])
    assert sorted(denoms.values()) == [66] * 4 + [67] * 8
    must_reject(lambda: prepare.build_view(five + b"\n", seven))
    must_reject(lambda: prepare.build_view(seven, five))
    source = (BASELINE / "overlay/vlm/plugins/gemma_domain_continuous.py").read_bytes()
    rendered = prepare.render_callback(source)
    must_reject(lambda: prepare.render_callback(source + b"\n"))
    compile(rendered, "<vizwiz rendered callback>", "exec")
    ast.parse(Path(__file__).with_name("gemma_vizwiz_two_updates_inputs.py").read_bytes())
    assert 'args.max_steps != 25' in rendered and 'len(kwargs["train_dataloader"]) != 100' in rendered
    assert "control.should_training_stop = True" in rendered and "control.should_save = False" in rendered
    assert "source_line = identify(kwargs, self.source_index)" in rendered and "check_update(step, counters, 476" in rendered
    assert 'update["adapter_before_SHA256"] = self.previous_adapter["SHA256"]' in rendered
    assert 'update["adapter_after_SHA256"] = adaptation["SHA256"]' in rendered
    assert 'update["learning_rates"] = rates' in rendered and "parameter_update_passed=bool(changing_steps)" in rendered
    assert "def fresh_config" in rendered and "must not consume a reference initial snapshot" in rendered
    scope = json.loads((BASELINE / "continuous-plan.json").read_bytes())
    command = scope["command"]
    values = {key: command[command.index(key) + 1] for key in (
        "--max_steps", "--per_device_train_batch_size", "--gradient_accumulation_steps",
        "--dataset_shuffle", "--warmup_ratio", "--dataloader_num_workers")}
    assert values == {"--max_steps": "25", "--per_device_train_batch_size": "1",
                      "--gradient_accumulation_steps": "4", "--dataset_shuffle": "true",
                      "--warmup_ratio": "0.03", "--dataloader_num_workers": "4"}
    receipt = {"schema": lineage["schema"] + "-local-verification", "view_SHA256": lineage["view_SHA256"],
               "view_rows": 800, "view_bytes": len(view), "unique_train_rows": 12, "rows_at_66_repeats": 4,
               "published_scope_SHA256": prepare.APPROVED_SCOPE_SHA, "published_builder_SHA256": prepare.APPROVED_BUILDER_SHA,
               "derived_lineage": {"bytes": len(derived), "SHA256": prepare.digest(derived)},
               "original_mounts_verified_by_this_run": False, "path_rewrite": False,
               "rows_at_67_repeats": 8, "nominal_scheduler_steps": 25, "target_updates_requested": 2,
               "original_trainer_flags": values, "frozen_label_counts": lineage["frozen_training_labels"],
               "effective_batch": 32, "target_microbatches_per_rank": 8,
               "frozen_callback_SHA256": prepare.CALLBACK, "rendered_callback_SHA256": prepare.digest(rendered.encode()),
               "source_streams": lineage["source_streams"], "actual_native_index_callable_present": True,
               "local_native_collator_calls": 0, "actual_sampler_exposures_observed": 0,
               "actual_optimizer_updates": 0, "actual_GPU": 0, "target_staged": False}
    raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    with Path(sys.argv[1]).open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"receipt_SHA256": prepare.digest(raw), "bytes": len(raw), "view_SHA256": prepare.digest(view)}))


if __name__ == "__main__":
    main()
