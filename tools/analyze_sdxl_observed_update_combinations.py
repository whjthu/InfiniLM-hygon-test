"""Compare actual SDXL first-update residuals from recorded interventions."""

from collections import Counter, defaultdict
import hashlib
import json
import math

from analyze_sdxl_original596_update_budget import (
    COMPARISON, NAME, POLICY, REFERENCE, ROOT, RUN, TENSOR_COUNT,
    pinned_json, unique_object, validate_metric,
)


RUNS = {
    "1087656.417": (
        "training-sdxl-first-groupnorm-update-v1-2d945ec0fcbf4bdb9bf1632710c01b64",
        "902ebe6a1c6f345947700d8416167e089fa5cfaa8a5d43cebd528154283d8b32", None,
        "first GroupNorm intervention", 0.20054350477735272,
    ),
    "1087656.455": (
        "training-sdxl-time-projection-full-update-v1-cad19b9c131d487dabafe26088a78ed4",
        "f35b7ef2b6a6fd4ab8d55e81f514d91584fdf7b3a419d0f235a3479b852a2536", None,
        "matching first time projection", 0.19819117386511054,
    ),
    "1087656.475": (
        "training-sdxl-a100-loss-adjoint-full-update-v1-c59d3373987648b8baf181deb9f37ea9",
        None, "b7b82fda4bb1927d835190bae706ad8725038c4d63a33b1274cfe7dc41a2fb3b",
        "matching final loss adjoint", 0.20210564870992906,
    ),
    "1087656.480": (
        "training-sdxl-lora-b-fp32-gradient-full-update-v1-34f4502eb63d42ccae1425b15b5b7475",
        "106e5491cf9db4d4d3e04c5c08b16d3b696170568125ad1af583eb173ec1ced2", None,
        "FP32 local LoRA B weight gradients", 0.19271896545609946,
    ),
    "1087656.540": (
        "training-sdxl-frozen-conv-dgrad-fp32-v2-9aede86323d74badb2f00ecdbf88b5c3",
        None, "7547fb7b94d22b0dd290bdf5ec75186761aed52e5e1f69958e7f938c37c42f56",
        "FP32 frozen convolution input gradients", 0.19477395241396306,
    ),
    "1087656.596": (
        RUN.name, COMPARISON, None, "FP32 attention adjoints", 0.19399272635879333,
    ),
}


def read_bounded(path, digest, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"recorded digest or size differs: {path.parent.name}/{path.name}")
    return json.loads(raw, object_pairs_hook=unique_object), len(raw)


def summarize(comparisons, threshold):
    common = next(iter(comparisons.values()))["metrics"]["parameter_update"]["per_tensor"]
    if len(common) != TENSOR_COUNT:
        raise ValueError("wrong full trainable population")
    by_step = {}
    for step, packet in comparisons.items():
        if (packet["stage_id"] != "sdxl-sft" or packet["boundary"] != "fixed_realized_inputs"
                or packet["accelerator"] != "hip" or packet["reference_sha256"] != REFERENCE
                or packet["policy_sha256"] != POLICY or packet["passed"] is not False
                or packet["checks"]["parameter_update"] is not False or "structural_failure" in packet):
            raise ValueError(f"incomparable original update: {step}")
        metric = packet["metrics"]["parameter_update"]
        norms = validate_metric(metric, common.keys())
        rows = metric["per_tensor"]
        if any(rows[n]["reference_l2_squared"] != base["reference_l2_squared"] for n, base in common.items()):
            raise ValueError(f"per-tensor reference changed: {step}")
        if not math.isclose(metric["relative_l2"], RUNS[step][-1], rel_tol=1e-10):
            raise ValueError(f"published update verdict changed: {step}")
        by_step[step] = {n: row["difference_l2_squared"] for n, row in rows.items()}

    reference_energy = norms["reference_l2_squared"]
    scopes = {key: defaultdict(list) for key in ("whole_model", "unet_block", "attention_module", "tensor")}
    for name in common:
        match = NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected trainable layout: {name}")
        module = name[:match.end("attention")]
        for key, scope in zip(("all", match["block"], module, name), scopes.values()):
            scope[key].append(name)

    def select(groups):
        chosen = Counter()
        tied = 0
        minima = []
        for names in groups.values():
            energies = sorted((math.fsum(rows[n] for n in names), step) for step, rows in by_step.items())
            energy, step = energies[0]
            minima.append(energy)
            if len(energies) > 1 and energies[1][0] == energy:
                tied += 1
            else:
                chosen[step] += 1
        error = math.fsum(minima)
        return {
            "partitions": len(groups),
            "winning_partitions_without_ties": dict(sorted(chosen.items())),
            "tied_partitions": tied,
            "minimum_squared_error": error,
            "minimum_relative_l2": math.sqrt(error / reference_energy),
            "mathematical_selector_meets_gate": error <= threshold**2 * reference_energy,
        }

    result = {scope: select(groups) for scope, groups in scopes.items()}
    minima = [item["minimum_squared_error"] for item in result.values()]
    if any(fine > coarse * (1 + 1e-12) for coarse, fine in zip(minima, minima[1:])):
        raise ValueError("a finer partition must permit at least as good a selection")
    return result


def main():
    policy = pinned_json(ROOT / "training-t06-sdxl-first-reference-r2/policy.json", 2677, POLICY)
    threshold = policy["metrics"]["parameter_update_relative_l2_max"]
    if threshold != 0.05 or policy["reference_payload_sha256"] != REFERENCE:
        raise ValueError("wrong first-update gate")
    comparisons, sources = {}, {}
    for step, (directory, digest, summary_digest, label, metric) in RUNS.items():
        run = ROOT / directory
        if summary_digest:
            summary, _ = read_bounded(run / "comparison-summary.json", summary_digest, 16384)
            if summary["numeric_passed"] is not False or summary["parameter_counts"]["parameter_update"] != TENSOR_COUNT:
                raise ValueError(f"missing original comparison: {step}")
            digest = summary["full_private_comparison_SHA256"]
        packet, size = read_bounded(run / "comparison.json", digest, 3 * 1024**2)
        comparisons[step] = packet
        sources[step] = {
            "comparison_SHA256": digest, "comparison_bytes": size,
            "summary_SHA256": summary_digest, "intervention": label,
            "relative_l2": metric, "observed_archive_SHA256": packet["observed_sha256"],
        }
    if len({row["observed_archive_SHA256"] for row in sources.values()}) != len(sources):
        raise ValueError("duplicate observed first-update archive")
    print(json.dumps({
        "schema": "bw1000.training.sdxl.observed_update_combinations.v1",
        "tensors": TENSOR_COUNT, "sources": sources, "policy_SHA256": POLICY, "reference_SHA256": REFERENCE,
        "parameter_update_relative_l2_gate": threshold, "selections": summarize(comparisons, threshold),
        "new_GPU_steps": 0, "new_CPU_steps": 0, "training_qualified": False,
        "scope": "The selector chooses recorded updates for disjoint whole tensors or larger groups. "
                 "It cannot predict interactions or prove the outcome of joint changes in a new backward pass.",
    }, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
