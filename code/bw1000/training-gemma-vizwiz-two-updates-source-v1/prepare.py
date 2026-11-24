"""Build the two-update Gemma VizWiz source view and render the frozen callback."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


FIVE = "be03f5443f6158f1b8358410632100b88cc15e0bba6649ecebd54648cbb88717"
SEVEN = "3f9051bb78a71f49d1e8ef51b06e706c164ff01a7650bc80a15154182d9e2c66"
CALLBACK = "167cd49754992a7a0d71c0f46a5013d59254a7485af458bf4a5191b4c428afea"
APPROVED = Path(__file__).resolve().parents[1] / "training-gemma-vizwiz-first-two-updates-v1"
APPROVED_BUILDER_SHA = "ccb8d64366330d0cb4a583ce138d9cf201d74e86349a51a352bf289f322ef180"
APPROVED_SCOPE_SHA = "7b494f529c5d202b522fb35cd5cf89af67bb97775534968518aee109668bd9d0"
APPROVED_VIEW_SHA = "cebeecad13db010a06f2ae7ef48260c6b26cc5ad42ea4993f6b8a60849ce1ad4"


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def build_view(five, reviewed):
    path = APPROVED / "build_view.py"
    raw = path.read_bytes()
    require(digest(raw) == APPROVED_BUILDER_SHA, "published VizWiz view builder differs")
    spec = importlib.util.spec_from_file_location("published_vizwiz_view", path)
    builder = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), vars(builder))
    scope = builder.load_scope(APPROVED / "TRAINING_SCOPE.json")
    require(builder.SCOPE_SHA256 == APPROVED_SCOPE_SHA, "published VizWiz scope differs")
    view, original = builder.build_view(scope, {"five": five, "reviewed": reviewed})
    require(original["view_SHA256"] == APPROVED_VIEW_SHA, "published VizWiz view differs")
    # This lineage is a derived member; the published train stream stays byte-for-byte.
    source_rows = [{"source": declared["name"], "ordinal": ordinal, "source_line_SHA256": sha}
                   for declared in scope["existing_frozen_rows"]
                   for ordinal, sha in enumerate(declared["ordered_line_SHA256"], 1)]
    return view, {
        "schema": "gemma-vizwiz-frozen-twelve-nominal25-two-updates-v1",
        "published_scope_SHA256": APPROVED_SCOPE_SHA, "published_view_receipt": original,
        "source_streams": {"five": FIVE, "reviewed": SEVEN}, "view_SHA256": APPROVED_VIEW_SHA,
        "view_rows": 800, "frozen_training_labels": original["label_class_counts"],
        "ordered_source_line_SHA256": [digest(line) for line in view.splitlines(keepends=True)],
        "source_rows": source_rows, "actual_sampler_order_observed": False, "native_optimizer_updates": 0,
    }


def once(text, old, new):
    require(text.count(old) == 1, "frozen callback source anchor differs")
    return text.replace(old, new, 1)


def render_callback(raw):
    require(digest(raw) == CALLBACK, "exact fresh-domain Gemma callback SHA differs")
    text = raw.decode().replace("\r\n", "\n")
    text = once(text, 'from vlm.runtime.vlm_update_contract import check_complete, check_update',
                'from vlm.runtime.vlm_update_contract import check_update\n'
                'from vlm.runtime.gemma_vizwiz_two_updates_inputs import load_index, identify')
    text = once(text, 'from vlm.runtime.resume_witness import optimizer_fingerprint, rng_fingerprint, rng_values',
                'from vlm.runtime.resume_witness import rng_fingerprint, rng_values')
    text = once(text, 'KEY = "gemma_domain_continuous_witness"', 'KEY = "gemma_vizwiz_two_updates_witness"')
    text = once(text, 'DATASET_SHA = "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"\n'
                'CHECKPOINTS = (10, 20, 25)', 'UPDATES = 2')
    text = text.replace("GemmaFreshDomainContinuous", "GemmaVizWizTwoUpdates")
    text = once(text, '        if (config["case"] != "T02.sft" or config["dataset_SHA256"] != DATASET_SHA',
                '        self.source_index = load_index(config)\n'
                '        if (config["case"] != "T02.sft"')
    text = once(text, '"Gemma continuous arm requires a fresh 25-step 100-batch run"',
                '"Gemma VizWiz requires the original nominal scheduler and fresh dataloader"')
    text = once(text, 'index >= 100 or step != index // 4', 'index >= UPDATES * 4 or step != index // 4')
    text = once(text, '        self.inputs.append(dict(ordinal=index + 1, global_step=step,',
                '        source_line = identify(kwargs, self.source_index)\n'
                '        self.inputs.append(dict(ordinal=index + 1, global_step=step, source_line_SHA256=source_line,')
    start = text.index('        if step >= 20:\n')
    end = text.index('        self.steps.append(update)\n', start)
    text = text[:start] + '''        update["learning_rates"] = rates
        update["adapter_before_SHA256"] = self.previous_adapter["SHA256"]
        update["adapter_after_SHA256"] = adaptation["SHA256"]
''' + text[end:]
    text = once(text, '        if step == 25:\n            control.should_save = True',
                '        if step == UPDATES:\n            control.should_save = False\n'
                '            control.should_training_stop = True')
    start = text.index('    def on_save(')
    end = text.index('\n\nif KEY in callbacks_map:', start)
    text = text[:start] + '''    def on_save(self, args, state, control, **kwargs):
        raise ValueError("unexpected checkpoint during the two-update VizWiz witness")

    def on_train_end(self, args, state, control, **kwargs):
        self.check_time()
        if state.global_step != UPDATES or len(self.steps) != UPDATES or self.saved or (
                len(self.inputs) != UPDATES * 4 or len(self.inputs) != len(self.outputs)):
            raise ValueError("incomplete Gemma VizWiz two-update witness")
        for hook in self.hooks:
            hook.remove()
        changing_steps = [step["global_step"] for step in self.steps
                          if not step["zero_learning_rate"] and step["adapter_changed"]]
        self.write("result.json", dict(rank=self.rank, world_size=8, completed_global_step=UPDATES,
            inputs=self.inputs, outputs=self.outputs, steps=self.steps, saved_checkpoints=self.saved,
            nominal_scheduler_steps=args.max_steps, actual_per_rank_microbatches=len(self.inputs),
            changing_steps_with_nonzero_rate=changing_steps,
            parameter_update_passed=bool(changing_steps)))
        return control
''' + text[end:]
    ast.parse(text)
    return text


def lineage_bytes(lineage):
    return (json.dumps(lineage, sort_keys=True, ensure_ascii=True) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    view = modes.add_parser("view")
    for name in ("five", "reviewed", "output", "lineage"):
        view.add_argument("--" + name, type=Path, required=True)
    overlay = modes.add_parser("render")
    overlay.add_argument("--source", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "view":
        result, lineage = build_view(args.five.read_bytes(), args.reviewed.read_bytes())
        with args.output.open("xb") as stream:
            stream.write(result)
        derived = lineage_bytes(lineage)
        with args.lineage.open("xb") as stream:
            stream.write(derived)
        print(json.dumps({"rows": lineage["view_rows"], "view_SHA256": lineage["view_SHA256"],
                          "derived_lineage": {"bytes": len(derived), "SHA256": digest(derived)}}))
    else:
        text = render_callback(args.source.read_bytes())
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(json.dumps({"callback_SHA256": digest(text.encode())}))


if __name__ == "__main__":
    main()
