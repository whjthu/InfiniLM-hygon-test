"""Check the actual archived controls and the per-slot scoring contract offline."""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import score_cpu as s


def reject(fn, expected):
    try:
        fn()
    except ValueError as error:
        assert str(error) == expected, str(error)
    else:
        raise AssertionError(expected)


def variants(root, plan, plan_sha):
    rows = []
    for arm in s.ARMS:
        for item in plan["samples"]:
            pair = item["pair"]
            rows.append(dict(arm=arm, outcome="completed", image=f"{arm}-{pair:02d}.private.png",
                             image_bytes=1024, image_SHA256="a" * 64, raw_latent_SHA256="b" * 64,
                             prepared_latent_SHA256="c" * 64, timesteps_SHA256="d" * 64,
                             **{key: item[key] for key in s.KEYS}))
    for i, (name, arms, request, step, part) in enumerate((
            ("old", s.ARMS[:2], s.OLD_REQUEST, "1194900.2762", rows[:12]),
            ("new", s.ARMS[2:], s.NEW_REQUEST, "1194900.2821", rows[12:]))):
        folder = root / name
        folder.mkdir()
        raw = b"".join(map(s.wire, part))
        ledgers = dict(request=request, plan_SHA256=plan_sha, facts=dict(step=step), sealed=True,
                       planned_slots=len(part), planned_pairs=6, generations_SHA256=s.digest(raw),
                       generations_bytes=len(raw))
        (folder / "result.json").write_bytes(s.wire(ledgers))
        (folder / "generations.jsonl").write_bytes(raw)
        assert s.native_ledger(folder, s.digest(s.wire(ledgers)), request, step, arms, plan_sha) == part
    composition = root / "composition"
    composition.mkdir()

    def certify(values, expected):
        scoped = [dict(row, artifact_scope="new" if row["arm"] == s.NEW else "original") for row in values]
        raw = b"".join(map(s.wire, scoped))
        pin = s.digest(raw)
        doc = dict(index_SHA256=pin, planned_slots=18, planned_pairs=6, reused_original_images=12, new_slots=6,
                   original_worker_result_SHA256=s.ORIGINAL_WORKER, four_row_worker_result_SHA256=s.NEW_WORKER,
                   score_computed=False)
        (composition / "result.json").write_bytes(s.wire(doc))
        (composition / "three-arm-index.private.jsonl").write_bytes(raw)
        with patch.multiple(s, COMPOSITION=s.digest(s.wire(doc)), INDEX=pin):
            if expected:
                reject(lambda: s.index_slots(composition, values, plan), expected)
            else:
                assert len(s.index_slots(composition, values, plan)) == 6
                reject(lambda: s.index_slots(composition, rows[1:] + rows[:1], plan), "EXACT_SLOT_PROMPT_SEED_AND_IMAGE_LINEAGE")

    certify(rows, None)
    foreign = [*rows[:-1], dict(rows[-1], seed=100)]
    certify(foreign, "EXACT_SLOT_PROMPT_SEED_AND_IMAGE_LINEAGE")
    drift = [*rows[:-1], dict(rows[-1], timesteps_SHA256="e" * 64)]
    certify(drift, "EXACT_SAME_NOISE_BOTH_CONTRASTS")


def summaries(controls):
    rows = [dict(pair=i, family=item["family"], status="scored", base=item["base"], two_row_cp25=item["cp25"],
                 four_row_cp25=item["cp25"] + .05 * (i // 2))
            for i, item in enumerate(controls)]
    full = s.summarize(rows)
    assert full["counts"] == dict(scored=6, error=0, missing=0)
    assert abs(full["primary_family_equal_mean"] - .05) < 1e-9
    rows[0]["status"] = "error"
    assert s.summarize(rows)["incomplete_pairs"] == 1
    rows[1]["status"] = "missing"
    scores = s.summarize(rows)
    assert scores["counts"] == dict(scored=4, error=1, missing=1)
    assert scores["primary_family_equal_mean"] is scores["secondary_family_equal_mean"] is None


def main():
    repo = Path(__file__).resolve().parents[3]
    published = repo / "results/bw1000/2026-11-28"
    worker = repo / "code/bw1000/training-sdxl-cc0-four-row-cp25-three-arm-v2"
    assert (worker / "plan.py").read_bytes() == Path(__file__).with_name("plan.py").read_bytes()
    plan, sha = s.frozen(worker / "PLAN.json")
    controls = s.original_controls(published / "specialized-sdxl-cp25-2765-three-cc0-families-clip-score-report.json", plan, sha)
    assert len(controls) == 6
    s.original_terminal(published / "specialized-sdxl-2821-four-row-cp25-six-images-and-three-arm-index.json")
    s.original_scorer(repo / "code/bw1000/training-sdxl-cc0-cp25-clip-score-v2/score_cpu.py")
    tree = ast.parse(Path(s.__file__).read_bytes())
    score, = (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score")
    calls = [n for n in ast.walk(score) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Attribute) and n.func.attr == "encode_image" for n in calls) == 1
    assert all(n.args[0].attr == "new" for n in calls if isinstance(n.func, ast.Name) and n.func.id == "input_bytes")
    with TemporaryDirectory() as temp:
        variants(Path(temp).resolve(), plan, sha)
    summaries(controls)
    print("actual .2821/.2822/.2765 archives and frozen scorer verified; slot, seed, noise and denominator fixtures PASS; no image encoding")


if __name__ == "__main__":
    main()
