"""Verify the real plan and checkpoint evidence, then test source-only three-arm boundaries."""

import ast
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import comparison as c
from plan import CONFIG, digest, frozen, wire


def reject(operation, error):
    try:
        operation()
    except ValueError as caught:
        assert str(caught) == error, str(caught)
    else:
        raise AssertionError(error)


def checkpoint_fixture(root, actual_proof):
    cp = root / "four/continuous/checkpoint-25"
    cp.mkdir(parents=True)
    adapter, manifest = cp / c.MEMBER, cp / c.MANIFEST
    adapter.write_bytes(b"fixture-adapter")
    sha = digest(adapter.read_bytes())
    names = ("metal_tool_chest_original.png", "metal_tool_chest_side.png",
             "painted_wooden_stool_original.png", "painted_wooden_stool_side.png")
    records = [dict(id=i, path=name) for i, name in enumerate(names)]
    data = dict(records=records, ordered_sha256=digest(wire(records).rstrip(b"\n")))
    payload = dict(schema="sdxl-native-lora-full-step-hip-v1", boundary="after-complete-optimizer-scheduler-update",
                   step=25, next_step=26, contract=dict(hip=True, plan={"max_train_steps": 25}, data=data),
                   files={c.MEMBER: dict(bytes=adapter.stat().st_size, sha256=sha)}, state_sha256=c.CHECKPOINT_STATE)
    payload["integrity_sha256"] = digest(wire(payload).rstrip(b"\n"))
    manifest.write_bytes(wire(payload))
    pin = digest(manifest.read_bytes())
    args = SimpleNamespace(training_proof=actual_proof, adapter_sha=sha, checkpoint_manifest_sha=pin,
                           adapter=adapter, checkpoint_manifest=manifest)
    reject(lambda: c.checkpoint(args), "EXACT_2769_CONTINUOUS_CP25_AUTHORITY")
    with patch.multiple(c, CHECKPOINT_ROOT=cp, ADAPTER_SHA=sha, CHECKPOINT_SHA=pin, ADAPTER_BYTES=adapter.stat().st_size):
        assert c.checkpoint(args) == b"fixture-adapter"
        args.adapter = root / "resume" / c.MEMBER
        reject(lambda: c.checkpoint(args), "EXACT_2769_CONTINUOUS_CP25_AUTHORITY")
        args.adapter = adapter
        adapter.write_bytes(b"fixture-invalid")
        reject(lambda: c.checkpoint(args), "EXACT_PINNED_MEMBER")


def pairing_fixture(root, plan, plan_sha, plan_file):
    old, new = root / "old", root / "new"
    old.mkdir()
    new.mkdir()
    latents = old / "latents"
    latents.mkdir()
    pins, old_rows, new_rows = [], [], {}
    for arm, folder in (("base", old), ("cp25", old), (c.NEW_ARM, new)):
        for item in plan["samples"]:
            pair = item["pair"]
            name = f"{arm}-{pair:02d}.private.png"
            Image.new("RGB", (256, 256), (pair, len(arm), 5)).save(folder / name)
            body = (folder / name).read_bytes()
            noise = bytes([pair]) * 8192
            raw_pin = digest(noise)
            if arm == "base":
                latent_name = f"pair-{pair:02d}.bf16.private"
                (latents / latent_name).write_bytes(noise)
                pins.append(dict(pair=pair, file=latent_name, seed=item["seed"], dtype=CONFIG["latent_dtype"],
                                 shape=CONFIG["latent_shape"], bytes=8192, SHA256=raw_pin))
            row = dict(arm=arm, outcome="completed", image=name, image_bytes=len(body), image_SHA256=digest(body),
                       raw_latent_SHA256=raw_pin, prepared_latent_SHA256="a" * 64, timesteps_SHA256="b" * 64,
                       **{key: item[key] for key in c.KEYS})
            if arm == c.NEW_ARM:
                new_rows[arm, pair] = row
            else:
                old_rows.append(row)
    latent_raw = wire(pins)
    (latents / "LATENTS.json").write_bytes(latent_raw)
    latent_sha = digest(latent_raw)
    ledger = b"".join(map(wire, old_rows))
    (old / "generations.jsonl").write_bytes(ledger)
    summary = dict(request=c.OLD_REQUEST, sealed=True, passed=True, planned_slots=12, plan_SHA256=plan_sha,
                   generations_SHA256=digest(ledger), generations_bytes=len(ledger), complete_same_noise_pairs=6,
                   facts=dict(step="1194900.2762", latent_manifest_SHA256=latent_sha,
                              effective_scheduler_config_SHA256="c" * 64, actual_generation_calls=12))
    prior = wire(summary)
    (old / "result.json").write_bytes(prior)
    with patch.multiple(c, OLD_RESULT=digest(prior), OLD_LATENTS=latent_sha):
        _, rows = c.original(old, plan, plan_sha)
        assert len(c.old_noise(old, plan, rows)) == 6
        old_pins = {p: digest(p.read_bytes()) for p in old.rglob("*") if p.is_file()}
        facts = dict(step="1194900.9911", model_loads=1, actual_generation_calls=6,
                     latent_manifest_SHA256=latent_sha, effective_scheduler_config_SHA256="c" * 64,
                     adapter_manifest_SHA256=c.CHECKPOINT_SHA, adapter_SHA256=c.ADAPTER_SHA,
                     live_adapter=dict(active=c.NEW_ARM, strength=1.0, tensors=1120, SHA256=c.ADAPTER_SHA))
        request = "1" * 32
        c.seal(new, plan, plan_sha, request, new_rows, facts, None)
        terminal = root / "terminal.json"
        terminal.write_bytes(wire(dict(request=request, numbered_gpu_step=facts["step"], independent_post_clean=True,
                                       worker_result_SHA256=digest((new / "result.json").read_bytes()))))
        args = SimpleNamespace(plan=plan_file, original=old, new=new, terminal=terminal,
                               terminal_sha=digest(terminal.read_bytes()), output=root / "output")
        result = c.compose(args)
        assert (result["planned_slots"], result["reused_original_images"], result["completed_primary_pairs"]) == (18, 12, 6)
        assert result["blind_review_ready"] and len(list(args.output.iterdir())) == 2
        assert all(digest(p.read_bytes()) == sha for p, sha in old_pins.items())

        partial = root / "partial"
        partial.mkdir()
        fake = new_rows[c.NEW_ARM, 0]
        (partial / fake["image"]).write_bytes((new / fake["image"]).read_bytes())
        failure = {"phase": "fixture"}
        c.seal(partial, plan, plan_sha, request, {(c.NEW_ARM, 0): dict(fake, timesteps_SHA256="d" * 64)}, facts, failure)
        value = json.loads(terminal.read_bytes())
        value["worker_result_SHA256"] = digest((partial / "result.json").read_bytes())
        terminal.write_bytes(wire(value))
        args.new, args.output, args.terminal_sha = partial, root / "bad", digest(terminal.read_bytes())
        reject(lambda: c.compose(args), "THREE_ARM_NOISE_OR_TIMESTEP_MISMATCH")
        assert not args.output.exists()


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    plan_file = here / "PLAN.json"
    plan, plan_sha = frozen(plan_file)
    proof = repo / "results/bw1000/2026-11-28/specialized-sdxl-2769-four-native-rows-25-plus-5-exact-CP20-and-clean.json"
    receipt = repo / "local-only/sdxl-four-row-cp25-adapter-manifest-pins-0d0eb974acd54b2b9628660b990f8c8a.json"
    observed = json.loads(receipt.read_bytes())
    current = observed["response"]["artifacts"]["continuous"]
    assert observed["authority_verified"] is observed["remote_received"] is True
    assert (current["adapter_SHA256"], current["adapter_bytes"], current["manifest_SHA256"], current["state_SHA256"]) == (
        c.ADAPTER_SHA, c.ADAPTER_BYTES, c.CHECKPOINT_SHA, c.CHECKPOINT_STATE)
    pinned = repo / "code/bw1000/training-sdxl-cc0-cp25-three-family-native-fp32-vae-v2"
    assert (here / "PLAN.json").read_bytes() == (pinned / "PLAN.json").read_bytes()
    assert (here / "plan.py").read_bytes() == (pinned / "plan.py").read_bytes()
    source = ast.parse((here / "native.py").read_bytes())
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "pipe"
               for node in ast.walk(source)) == 1
    with TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        checkpoint_fixture(root, proof)
        pairing_fixture(root, plan, plan_sha, plan_file)
    print("exact .2769 observed pins, original plan bytes, same-noise composition and refusal fixtures PASS; models and GPU calls 0")


if __name__ == "__main__":
    main()
