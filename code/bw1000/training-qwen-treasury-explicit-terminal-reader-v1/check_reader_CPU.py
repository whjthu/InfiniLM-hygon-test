"""Exercise pure reader boundaries with explicitly synthetic CPU-only records."""
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import tempfile
import types


def check(reader_path, scratch):
    module = runpy.run_path(str(reader_path))
    def fp(key):
        return dict(SHA256=hashlib.sha256(key.encode()).hexdigest(), tensors=1, elements=1,
                    floating_elements=1, floating_l2=1.0, finite=True)
    steps = {phase: [handle, module["LABEL"] + "-" + phase, "COMPLETED", "0:0", "1", "a01r4n16",
                    "cpu=64,gres/dcu=8,mem=224000M,node=1" if phase == "gpu" else "cpu=2,mem=4G,node=1"]
             for phase, handle in module["HANDLES"].items()}
    intent = dict(request_id=module["REQUEST"], publication=module["PUBLICATION"], claim_SHA256=module["CLAIM_SHA"],
                  parent="1087656", node="a01r4n16", epoch=module["EPOCH"], deadline=module["EPOCH"] + 1620)
    stages = {phase: dict(row=row, terminal_visible=True, passed=True, rc=0) for phase, row in steps.items()}
    docs = dict(intent=intent, launch=copy.deepcopy(intent), controller=dict(passed=True, failure=None, stages=stages),
                native=dict(passed=True, rc=0, error=None, cleanup=dict(owned_processes_remaining=0, termination_needed=False), finished=[]),
                plan=dict(dataset_SHA256=module["DATA_SHA"], native_preflight_step="1087656.326", planned_updates=5,
                          domain_arm="continuous", resume_from_checkpoint=None),
                witnesses={}, initializations={}, rng={}, witness_SHA256={}, checkpoint_states={})
    docs.update({phase + "_accounting": value for phase, value in stages.items()})
    pci = [f"0000:{i:02x}:00.0" for i in range(8)]
    common = dict(pci=pci, boot_id="fixture-boot", kfd_holder_count=0, full_fd_inventory_performed=True)
    docs.update(before=dict(common, step=steps["pre"][0]), assigned=dict(common, step=steps["gpu"][0], visible_pci=pci),
                after=dict(common, step=steps["post"][0], owned_remaining=0))
    for rank in range(8):
        witness = dict(rank=rank, world_size=8, completed_global_step=5, saved_checkpoints=[2, 4, 5],
            frozen={"model.visual": dict(tensors=1, parameters=1)},
            inputs=[dict(ordinal=i, global_step=(i - 1)//4, full_input=fp(f"input:{rank}:{i}")) for i in range(1, 21)],
            outputs=[dict(ordinal=i, loss=fp(f"loss:{rank}:{i}"), loss_scalar=1.0) for i in range(1, 21)],
            steps=[dict(global_step=i, state={name: fp(f"{name}:{i}") for name in ("adapter", "optimizer", "scheduler")}) for i in range(1, 6)])
        docs["witnesses"][rank] = witness
        docs["initializations"][rank] = dict(module["INITIAL"])
        docs["rng"][rank] = dict(rank=rank, RNG=fp(f"RNG:{rank}"))
        digest = hashlib.sha256(json.dumps(witness, sort_keys=True).encode()).hexdigest()
        docs["witness_SHA256"][rank] = digest
        docs["native"]["finished"].append(dict(rank=rank, SHA256=digest))
    names = {"args.json", "adapter_config.json", "adapter_model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", *(f"rng_state_{i}.pth" for i in range(8))}
    checkpoints = {n: {name: dict(bytes=4, SHA256=hashlib.sha256(name.encode()).hexdigest()) for name in names} for n in (2, 4, 5)}
    docs["checkpoint_states"] = {n: dict(global_step=n, max_steps=5, epoch=n/5) for n in (2, 4, 5)}
    verified = module["verify_records"](docs, steps, checkpoints)
    assert verified["actual_optimizer_updates"] == 5 and verified["rank_microbatch_occurrences"] == 160
    assert verified["recovery_qualified"] is False and verified["data_order_matches_old_input_control"] is None
    assert module["verify_cleanup"](docs, steps)["clean"] is True
    cases = [
        ("wrong_request", lambda d,s,c: d["intent"].update(request_id="0"*32)),
        ("reset_epoch", lambda d,s,c: d["intent"].update(deadline=d["intent"]["deadline"] + 1)),
        ("RUNNING_nonterminal", lambda d,s,c: s["gpu"].__setitem__(2, "RUNNING")),
        ("foreign_actual_handle", lambda d,s,c: s["gpu"].__setitem__(0, "1087656.285")),
        ("missing_post", lambda d,s,c: s.pop("post")),
        ("collector_failure", lambda d,s,c: d["controller"].update(failure="fixture failure")),
        ("forced_cleanup", lambda d,s,c: d["native"]["cleanup"].update(termination_needed=True)),
        ("wrong_CPU_gate", lambda d,s,c: d["plan"].update(native_preflight_step="1103606.153")),
        ("wrong_dataset", lambda d,s,c: d["plan"].update(dataset_SHA256="0"*64)),
        ("resume_instead_of_fresh", lambda d,s,c: d["plan"].update(resume_from_checkpoint="old-checkpoint")),
        ("wrong_initial_adapter", lambda d,s,c: d["initializations"][0].update(named_parameters_SHA256="0"*64)),
        ("nonfresh_optimizer", lambda d,s,c: d["initializations"][0].update(optimizer_state_entries=1)),
        ("restored_RNG", lambda d,s,c: d["initializations"][0].update(RNG_restored=True)),
        ("missing_microbatch", lambda d,s,c: d["witnesses"][0]["inputs"].pop()),
        ("wrong_input_order", lambda d,s,c: d["witnesses"][0]["inputs"][0].update(ordinal=2)),
        ("wrong_accumulation_step", lambda d,s,c: d["witnesses"][0]["inputs"][0].update(global_step=1)),
        ("nonfinite_loss", lambda d,s,c: d["witnesses"][0]["outputs"][0].update(loss_scalar=float("nan"))),
        ("nonfinite_fingerprint", lambda d,s,c: d["witnesses"][0]["inputs"][0]["full_input"].update(finite=False)),
        ("missing_update", lambda d,s,c: d["witnesses"][0]["steps"].pop()),
        ("unchanged_adapter", lambda d,s,c: d["witnesses"][0]["steps"][-1]["state"].update(adapter=d["witnesses"][0]["steps"][0]["state"]["adapter"])),
        ("rank_state_disagrees", lambda d,s,c: d["witnesses"][1]["steps"][1]["state"]["optimizer"].update(SHA256="0"*64)),
        ("native_rank_digest", lambda d,s,c: d["native"]["finished"][0].update(SHA256="0"*64)),
        ("checkpoint4_missing_RNG", lambda d,s,c: c[4].pop("rng_state_7.pth")),
        ("checkpoint4_wrong_cursor", lambda d,s,c: d["checkpoint_states"][4].update(epoch=.4)),
        ("checkpoint_budget", lambda d,s,c: c[2]["optimizer.pt"].update(bytes=1024**3)),
    ]
    rejected = []
    for name, mutation in cases:
        d,s,c = copy.deepcopy((docs, steps, checkpoints))
        mutation(d,s,c)
        try:
            module["verify_records"](d,s,c)
        except (ValueError, KeyError, TypeError, IndexError):
            rejected.append(name)
        else:
            raise AssertionError("bad evidence accepted: " + name)
    cleanup_rejected = []
    for name, mutation in [("KFD_holder", lambda d:d["after"].update(kfd_holder_count=1)),
                           ("live_owned_PID", lambda d:d["after"].update(owned_remaining=1)),
                           ("foreign_PCI", lambda d:d["after"].update(pci=["foreign"])),
                           ("partial_FD_inventory", lambda d:d["after"].update(full_fd_inventory_performed=False))]:
        changed = copy.deepcopy(docs); mutation(changed)
        try:
            module["verify_cleanup"](changed, steps)
        except ValueError:
            cleanup_rejected.append(name)
        else:
            raise AssertionError("bad cleanup accepted: " + name)
    # Actual audit() early incomplete-accounting path retains the full denominator.
    with tempfile.TemporaryDirectory(dir=scratch, prefix="reader-accounting-fixture-") as name:
        fn = module["audit"]; env = fn.__globals__; previous_root, previous_subprocess = env["RUN"], env["subprocess"]
        try:
            env["RUN"] = Path(name)
            rows = [steps["pre"], list(steps["gpu"])]
            rows[1][2] = "RUNNING"
            env["subprocess"] = types.SimpleNamespace(check_output=lambda *a,**kw: "\n".join("|".join(row) for row in rows))
            incomplete = fn(Path(name))
            assert incomplete["passed"] is False and incomplete["planned_denominator"] == module["PLANNED"]
            assert len(incomplete["actual_accounting"]) == 2 and incomplete["collected_rank_microbatches"] == {}
        finally:
            env["RUN"], env["subprocess"] = previous_root, previous_subprocess
    return dict(passed=True, synthetic_CPU_records_only=True, accepted_complete_fixture=True,
                rejected_model_boundaries=rejected, rejected_cleanup_boundaries=cleanup_rejected,
                missing_post_RUNNING_retains_full_denominator=True, actual_original_run_consumed=False,
                actual_GPU_invocations=0, raw_model_or_checkpoint_tensors_loaded=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(check(Path(__file__).with_name("read_original.py"), options.scratch), allow_nan=False))
