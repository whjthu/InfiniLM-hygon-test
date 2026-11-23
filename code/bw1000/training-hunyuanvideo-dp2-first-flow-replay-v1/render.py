"""Render the first-flow diagnostic into the exact common training candidate."""

import argparse
from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
BACKEND = "generation/src/gencuda/backends/hunyuanvideo_data_parallel.py"
ENTRY = "tools/hygon_training_recipe.py"
HELPER = "generation/src/gencuda/hunyuanvideo_dp2_flow_replay.py"
BASE = {
    BACKEND: "f21dfc2a5234c0882de9fadbc042f3fb4ab015a22dab8d02f66c3f4a6239fd0b",
    ENTRY: "6fda4ff1d3d87a165ace61b1ed4e347b22705bf983805c1f9182ac98d11f14fa",
}


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("original first-flow source boundary differs")
    return source.replace(old, new, 1)


def backend_changes(source):
    source = replace(source, 'from .. import hunyuanvideo_distributed_checkpoint as checkpoint\n',
                     'from .. import hunyuanvideo_distributed_checkpoint as checkpoint\n'
                     'from .. import hunyuanvideo_dp2_flow_replay as first_flow\n')
    source = replace(source, 'def validate_config(config: dict[str, Any]) -> None:\n',
                     'def validate_config(config: dict[str, Any]) -> None:\n'
                     '    if "first_flow_replay" in config:\n'
                     '        first_flow.validate_binding(config)\n')
    source = replace(source, '        forward_devices = []\n',
                     '        forward_devices = []\n'
                     '        diagnostic = "first_flow_replay" in config\n'
                     '        prediction = [None]\n'
                     '        diagnostic_artifacts = []\n\n'
                     '        def record_phase(phase: str, payload: Callable[[], Any]) -> None:\n'
                     '            artifact = collective_action(torch, device, "first-update " + phase,\n'
                     '                lambda: first_flow.write_update_artifact(torch, rank_dir, phase, payload()))\n'
                     '            diagnostic_artifacts.append(artifact)\n')
    source = replace(source, '            forward_devices.append(str(getattr(value[0], "device", "missing")))\n',
                     '            forward_devices.append(str(getattr(value[0], "device", "missing")))\n'
                     '            if diagnostic:\n'
                     '                prediction[0] = value[0].detach()\n')
    source = replace(source, '        trainable = [p for p in model.parameters() if p.requires_grad]\n',
                     '        if diagnostic:\n'
                     '            warmup, total, checkpoint_steps = 0, 1, set()\n'
                     '        trainable = [p for p in model.parameters() if p.requires_grad]\n')
    source = replace(source, '                loss = native.native_flow_mse(torch, model, flow_values, inputs)\n',
                     '                if diagnostic:\n'
                     '                    inputs = collective_action(torch, device, "sealed A100 rank input",\n'
                     '                        lambda: first_flow.replay_inputs(torch, config, rank, inputs, device, before))\n'
                     '                    record_phase("before", lambda: checkpoint.native.capture_model_state(model, optimizer))\n'
                     '                loss = native.native_flow_mse(torch, model, flow_values, inputs)\n'
                     '                if diagnostic:\n'
                     '                    record_phase("forward", lambda: {"prediction": prediction[0], "loss": loss.detach()})\n')
    source = replace(source, '                average_gradients(torch, model, optimizer, device)\n',
                     '                average_gradients(torch, model, optimizer, device)\n'
                     '                if diagnostic:\n'
                     '                    record_phase("preclip", lambda: {name: p.grad for name, p in model.named_parameters()\n'
                     '                                             if p.requires_grad})\n')
    source = replace(source, '                collective_action(torch, device, "native optimizer step", optimizer.step)\n',
                     '                if diagnostic:\n'
                     '                    record_phase("postclip", lambda: {"norm": norm, "gradients":\n'
                     '                        {name: p.grad for name, p in model.named_parameters() if p.requires_grad}})\n'
                     '                collective_action(torch, device, "native optimizer step", optimizer.step)\n')
    source = replace(source, '                fingerprint = checkpoint.replica_sha(payload)\n',
                     '                if diagnostic:\n'
                     '                    record_phase("updated", lambda: payload["state"])\n'
                     '                fingerprint = checkpoint.replica_sha(payload)\n')
    source = replace(source, '        checkpoint.write_json_atomic(rank_dir / "summary.json", summary)\n',
                     '        if diagnostic:\n'
                     '            summary.update(first_update_artifacts=diagnostic_artifacts, formal_performance_eligible=False,\n'
                     '                           measurement_wall_seconds=None, measurement_steps=0)\n'
                     '        checkpoint.write_json_atomic(rank_dir / "summary.json", summary)\n')
    source = replace(source, '    result["metrics"].update(rank_window_metrics(\n'
                     '        output_dir, result["logs"], warmup=config.get("warmup_steps", 5),\n'
                     '        measured=config.get("max_steps", 20), resumed=resumed,\n'
                     '    ))\n',
                     '    if "first_flow_replay" in config:\n'
                     '        result["metrics"].update(rank_measurement_wall_seconds=None,\n'
                     '            slowest_rank_measurement_wall_seconds=None, measurement_updates_per_second=None,\n'
                     '            timing_boundary=None, complete_5_warmup_20_measured_window=False,\n'
                     '            formal_performance_eligible=False)\n'
                     '    else:\n'
                     '        result["metrics"].update(rank_window_metrics(\n'
                     '            output_dir, result["logs"], warmup=config.get("warmup_steps", 5),\n'
                     '            measured=config.get("max_steps", 20), resumed=resumed,\n'
                     '        ))\n')
    return source


def entry_changes(source):
    source = replace(source, '    parser.add_argument("--asset-map", type=Path)\n',
                     '    parser.add_argument("--asset-map", type=Path)\n'
                     '    parser.add_argument("--dp2-first-flow-binding", type=vlm.absolute_path)\n'
                     '    parser.add_argument("--dp2-first-flow-binding-sha256")\n')
    source = replace(source, '    options = parser.parse_args(args)\n    if options.capture_first_policy_only:\n',
                     '    options = parser.parse_args(args)\n'
                     '    if (options.dp2_first_flow_binding is None) != (options.dp2_first_flow_binding_sha256 is None):\n'
                     '        parser.error("DP2 first-flow diagnosis requires both binding and SHA256")\n'
                     '    if options.dp2_first_flow_binding is not None and (options.case != "T09.dp2" or not options.run\n'
                     '            or options.dry_run or options.resume_from_checkpoint is not None or options.asset_map is None):\n'
                     '        parser.error("DP2 first-flow diagnosis requires a fresh T09.dp2 run with --asset-map")\n'
                     '    if options.capture_first_policy_only:\n')
    source = replace(source, '        plan["backend_options"] = ["--accelerator", generation_accelerator]\n',
                     '        plan["backend_options"] = ["--accelerator", generation_accelerator]\n'
                     '        if options.dp2_first_flow_binding is not None:\n'
                     '            with Path(options.dp2_first_flow_binding).open("rb") as stream:\n'
                     '                raw = stream.read(16385)\n'
                     '            if not 0 < len(raw) <= 16384 or hashlib.sha256(raw).hexdigest() != options.dp2_first_flow_binding_sha256:\n'
                     '                raise ValueError("DP2 first-flow binding bytes differ")\n'
                     '            plan["config"]["first_flow_replay"] = json.loads(raw)\n')
    return source


def render(base):
    sources = {}
    for name, update in ((ENTRY, entry_changes), (BACKEND, backend_changes)):
        raw = (base / name).read_bytes()
        if sha256(raw).hexdigest() != BASE[name]:
            raise ValueError(f"original common framework SHA256 differs: {name}")
        sources[name] = update(raw.decode("utf-8")).encode("utf-8")
    sources[HELPER] = (HERE / Path(HELPER).name).read_bytes()
    for name, raw in sources.items():
        compile(raw, name, "exec")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sources = render(args.base)
    args.output.mkdir()
    for name, raw in sources.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


if __name__ == "__main__":
    main()
