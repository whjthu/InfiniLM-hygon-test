"""Render a disabled Gemma CP20-to-step21 diagnostic delta from exact prior sources."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

HERE = Path(__file__).resolve().parent
PINS = {
    "vlm/plugins/gemma_domain_continuous.py": "167cd49754992a7a0d71c0f46a5013d59254a7485af458bf4a5191b4c428afea",
    "vlm/plugins/gemma_domain_resume.py": "188e07812f9b3c9f5916da520f05785e6e7c6c34b95225b75440e78ce777fa01",
    "vlm/runtime/resume_witness.py": "4454f738caa07554caa5054234414941e07431416ac8ca3d6b5a3368a14bb02c",
    "vlm/runtime/forward_context_state.py": "df412bb2982914fb47705a7a90b4a0bf340e6e5551595e754caa474eee6eed76",
}
CONTINUOUS, RESUME, WITNESS, CONTEXT = PINS
PLAN_SHA = "939b112dcc16a3e8d3203f61203f4b0b3ba5572325c2e6814de0645693049537"
ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ORIGINAL = ROOT / "training-gemma-usgs72-fresh-initial-domain25-v1"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def exact(path, pin):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == pin, "ORIGINAL_SOURCE_HASH")
    return raw


def edit(source, old, new):
    need(source.count(old) == 1, "EXACT_ORIGINAL_SOURCE_ANCHOR")
    return source.replace(old, new, 1)


def derive_continuous(raw):
    s = raw.decode()
    s = edit(s, "from vlm.runtime.first_update_audit import fingerprint",
             "from vlm.runtime.first_update_audit import fingerprint\nfrom vlm.runtime.first_tail import for_rank")
    s = edit(s, "        self.previous_adapter = self.adapter()\n        self.progress()",
             """        self.previous_adapter = self.adapter()
        self.first_tail = for_rank(self.rank, "continuous", self.model, torch, rng_values, fingerprint, cpu)
        self.progress()""")
    s = edit(s, "        self.progress()\n\n    def after(self, module, argv, kwargs, output):",
             """        self.progress()
        if self.first_tail is not None:
            self.first_tail.before(step, index % 4 + 1, argv, kwargs, self.inputs[-1])

    def after(self, module, argv, kwargs, output):""")
    s = edit(s, """    def after(self, module, argv, kwargs, output):
        self.check_time()""", """    def after(self, module, argv, kwargs, output):
        if self.first_tail is not None:
            index = len(self.inputs) - 1
            step = self.trainer.state.global_step
            self.first_tail.after(step, index % 4 + 1)
        self.check_time()""")
    return edit(s, """        self.progress()

    def on_step_end""", """        self.progress()
        if self.first_tail is not None:
            self.first_tail.write(step, index % 4 + 1, self.outputs[-1], self.root / "first-tail-forward.json")

    def on_step_end""")


def derive_witness(raw):
    s = raw.decode()
    s = edit(s, "boundary_modules=None, allow_mid_epoch=False):",
             "boundary_modules=None, allow_mid_epoch=False, lightweight_first_tail=False):")
    s = edit(s, "        if (diagnostic_stop_step is not None", """        if lightweight_first_tail and (checkpoint_step != 20 or max_steps != 25 or diagnostic_stop_step != 21
                                       or entry_state_step is not None or boundary_modules is not None):
            raise ValueError('lightweight witness is restricted to CP20 first-tail observation')
        self.lightweight_first_tail = lightweight_first_tail
        if (diagnostic_stop_step is not None""")
    s = edit(s, "or entry_state_step is None or diagnostic_stop_step != entry_state_step + 1",
             """or (not lightweight_first_tail and
                    (entry_state_step is None or diagnostic_stop_step != entry_state_step + 1))""")
    return edit(s, "if (not self.entry_state_captured or not self.steps",
                "if ((not self.lightweight_first_tail and not self.entry_state_captured) or not self.steps")


def derive_resume(raw):
    s = raw.decode()
    s = edit(s, "from vlm.runtime.first_update_capture import remaining", """from vlm.runtime.first_update_capture import cpu, remaining
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_tail import for_rank""")
    s = edit(s, """class GemmaResumeWitness(ResumeWitness):
    def after(self, module, argv, kwargs, output):
        super().after(module, argv, kwargs, output)""", """class GemmaResumeWitness(ResumeWitness):
    first_tail = None

    def before(self, module, argv, kwargs):
        super().before(module, argv, kwargs)
        if self.first_tail is not None:
            micro = (len(self.inputs) - 1) % self.accumulation + 1
            self.first_tail.before(self.trainer.state.global_step, micro, argv, kwargs, self.inputs[-1])

    def after(self, module, argv, kwargs, output):
        if self.first_tail is not None:
            micro = (len(self.inputs) - 1) % self.accumulation + 1
            self.first_tail.after(self.trainer.state.global_step, micro)
        super().after(module, argv, kwargs, output)""")
    s = edit(s, '        self.outputs[-1]["loss_scalar"] = value', '''        self.outputs[-1]["loss_scalar"] = value
        if self.first_tail is not None:
            self.first_tail.write(self.trainer.state.global_step, micro, self.outputs[-1],
                                  self.root / "first-tail-forward.json")''')
    s = edit(s, "device_rng=torch.cuda.get_rng_state_all, allow_mid_epoch=True,",
             """device_rng=torch.cuda.get_rng_state_all, allow_mid_epoch=True,
            diagnostic_stop_step=21, lightweight_first_tail=True,""")
    s = edit(s, '        if self.witness.plan["expected_batches_to_skip_in_resumed_epoch"] != 80:',
             '''        self.witness.first_tail = for_rank(rank, "resumed", model, torch, rng_values, fingerprint, cpu)
        if self.witness.plan["expected_batches_to_skip_in_resumed_epoch"] != 80:''')
    return edit(s, """        if state.global_step == 25:
            control.should_save = True
        return control""", """        probe = self.witness.first_tail
        if probe is not None and not probe.written:
            raise ValueError("missing rank0/1 first-tail forward metadata")
        return self.witness.guarded(self.witness.request_diagnostic_stop)(control)""")


def overlay(original, context, output):
    source = {name: exact((context if name == CONTEXT else original / name), pin)
              for name, pin in PINS.items()}
    source[CONTINUOUS] = derive_continuous(source[CONTINUOUS]).encode()
    source[RESUME] = derive_resume(source[RESUME]).encode()
    source[WITNESS] = derive_witness(source[WITNESS]).encode()
    source["vlm/runtime/first_tail.py"] = (HERE / "first_tail.py").read_bytes()
    for name, raw in source.items():
        compile(raw, name, "exec")
    output.mkdir(mode=0o700)
    files = {}
    for name, raw in source.items():
        path = output / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        files[name] = dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())
    (output / "OVERLAY_FILES.json").write_text(json.dumps(files, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return files


def plan(original_plan, request, port):
    need(re.fullmatch(r"[a-f0-9]{32}", request) and not request.startswith(
         ("247fa3682a11", "66e39b4b1aeb")) and 20000 <= port <= 60999 and port != 29932,
         "EXACT_NEW_REQUEST_AND_PORT")
    original = json.loads(exact(original_plan, PLAN_SHA))
    command = original["command"].copy()
    framework = ROOT / ("training-gemma-usgs72-resume21-light-" + request) / "framework"
    run = framework.parent
    plugin = str(framework / RESUME)
    out = str(run / "resumed/train")
    command[0] = str(framework / "vlm/runtime/swift_entry.py")
    command[command.index("--external_plugins") + 1] = plugin
    command[command.index("--output_dir") + 1] = out
    env = dict(original["environment_overrides"], MASTER_PORT=str(port), OUTPUT_DIR=out)
    if "PYTHONPATH" in env:
        prefix = str(ORIGINAL / "framework")
        need(prefix in env["PYTHONPATH"].split(":"), "EXACT_ORIGINAL_UNIFIED_FRAMEWORK_PATH")
        env["PYTHONPATH"] = ":".join(str(framework) if part == prefix else part for part in env["PYTHONPATH"].split(":"))
    return dict(request=request, case="T02.sft", command=command, environment_overrides=env,
                original_plan_SHA256=PLAN_SHA, resume_from_checkpoint=original["resume_from_checkpoint"],
                restored_global_step=20, native_skip=80, nominal_max_steps=25, stop_after_update=21,
                optimizer_updates_planned=1, forwards_per_rank_planned=4, world_size=8,
                framework=str(framework), separate_output_root=str(run), training_ready=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    make = commands.add_parser("overlay")
    for name in ("original", "context", "output"):
        make.add_argument("--" + name, type=Path, required=True)
    render = commands.add_parser("plan")
    render.add_argument("--original-plan", type=Path, required=True)
    render.add_argument("--request", required=True)
    render.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    value = (overlay(args.original, args.context, args.output) if args.mode == "overlay"
             else plan(args.original_plan, args.request, args.port))
    print(json.dumps(value, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
