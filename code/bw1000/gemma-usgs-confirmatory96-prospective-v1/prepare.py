"""Freeze a prospective Gemma catalog query plan and private exclusion identities."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from urllib.parse import urlencode


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / "code/bw1000/gemma-usgs-body12-closure-audit-v1/audit.py"
STYLE18 = ROOT / "local-only/training-zimage-USGS-style18-dataset-v1-original/dataset/data-binding-private.json"
OUTPUT = ROOT / "local-only/gemma-usgs-confirmatory96-prospective-v1"
GRIDS = (6, 7, 8, 11, 12, 13, 16, 17, 19, 22, 23, 24, 26, 27, 28, 29)
YEARS = (2008, 2009, 2018, 2019, 2020, 2021, 2022, 2025)


def audited_module():
    raw = AUDIT.read_bytes()
    assert sha256(raw).hexdigest() == "67048a168c2f36e13c2cb880cfffb6351433aec7b384fe921ed728fe7023cdb6"
    return runpy.run_path(str(AUDIT))


def request_rows(c):
    intervals = [
        dict(starttime=f"{year}-{start}T00:00:00Z", endtime_exclusive=f"{year}-{end}T00:00:00Z")
        for year in YEARS for start, end in (("01-05", "06-26"), ("07-05", "12-26"))
    ]
    rows = []
    for position, grid in enumerate(GRIDS):
        for slot in (position, (position + 7) % 16):
            span = intervals[slot]
            query = c.params(grid, span)
            rows.append(dict(query_id=f"G{grid:02d}-S{slot:02d}", grid_id=grid, slot_id=slot,
                             half_open_bounds=c.grid_bounds(grid), **span, query_params=query,
                             url=c.COUNT_ENDPOINT + "?" + urlencode(dict(query, format="text"))))
    assert len(rows) == len({row["url"] for row in rows}) == 32
    return rows


def exclusion(q, audited):
    routes, historical, _, _ = q.preflight(audited["COUNTS"])
    receipts = json.loads(q.read(q.OUTPUT / "BODY_RECEIPTS.json", 16384, audited["RECEIPTS_SHA"]))
    audit_path = ROOT / "local-only/gemma-usgs-confirmatory96-body12-v1/offline-closure-fdc3daba-original.json"
    actual = json.loads(q.read(audit_path, 32768, "36cfea307f3cd965f96484718d99c67bb186a2a062c569e641249c72bd61c9d9"))
    summary = actual["result"]
    assert actual["publication"] == "fdc3dabaab51aeaf532d30fbbbf4c2d25a421136"
    assert len(receipts) == len(routes) == summary["original_bodies_verified"] == 12
    prior = []
    body_proofs = {}
    for route, receipt in zip(routes, receipts, strict=True):
        query_id = route["query_id"]
        assert query_id == receipt["query_id"]
        recorded = receipt["body"]
        source = q.read(q.OUTPUT / (query_id + ".geojson"), q.CAP, recorded["SHA256"])
        assert q.C.pin(source) == recorded == summary["per_query"][query_id]["body"]
        rows, dispositions = audited["audit_body"](q, source, route)
        assert len(rows) == summary["per_query"][query_id]["raw_events"]
        assert dispositions["not_reviewed"] == summary["per_query"][query_id]["closure_only_not_reviewed"]
        start = len(prior)
        for row in rows:
            prior.append({key: row[key] for key in ("id", "aliases", "time_UTC_epoch_ms")} |
                         {"values": {key: row["values"][key] for key in ("latitude", "longitude")}})
        body_proofs[query_id] = dict(body=recorded, projected_slice=[start, len(prior)],
                                    unreviewed_identity_vertices=dispositions["not_reviewed"])
    assert len(historical) == 3832 and len(prior) == summary["capacity"]["raw_events"] == 3003
    return historical, prior, body_proofs


def style18_identities(q, known_ids):
    raw = q.read(STYLE18, 14457, "2152981e81882985de57a9419a34f895440d7965cc3b277c762f15304bc51c35")
    binding = json.loads(raw)
    assert binding["blind_body_read"] is binding["QA_or_gold_read"] is False
    splits = {}
    excluded_ids = set()
    for split, expected in (("train", (12, 72, 3)), ("dev", (6, 36, 2))):
        rows = [row for row in binding["images"] if row["split"] == split]
        ids = {event for row in rows for event in row["event_ids"]}
        groups = {row["group"] for row in rows}
        assert (len(rows), len(ids), len(groups)) == expected
        excluded_ids.update(ids)
        splits[split] = dict(images=len(rows), event_identities=len(ids), published_groups=len(groups),
                             ids_already_in_catalog_history=len(ids & known_ids))
    return sorted(excluded_ids), dict(binding=q.C.pin(raw), scope=splits,
                                      previously_reserved_blind_identity_coverage=False,
                                      usable_source_explicit_ID_exclusion_required=True)


def prepare():
    audited = audited_module()
    q = audited["load_policy"]()
    historical, prior, body_proofs = exclusion(q, audited)
    known_ids = {alias for row in historical + prior for alias in row["aliases"]}
    style_ids, style_scope = style18_identities(q, known_ids)
    private = dict(original_history=historical, previous_body_identities=prior,
                   explicit_style18_event_exclusions=style_ids)
    private_pin = q.C.pin(q.C.canonical(private))
    summary = dict(private_index=private_pin, original_history_SHA256=q.HISTORY_SHA,
                   original_history_rows=len(historical), prior_body_projection_rows=len(prior),
                   catalog_aliases=len(known_ids), all_known_catalog_alias_SHA256=q.digest(q.C.canonical(sorted(known_ids))),
                   style18_train_dev=style_scope, exact_prior_bodies=body_proofs)
    queries = request_rows(q.C)
    plan = dict(state="FROZEN_PROSPECTIVE_COUNT_ONLY", source="official USGS", years=list(YEARS),
                selected_grids=list(GRIDS), queries=queries, planned_queries=32, selected_body_limit=12,
                eligible_count_range=[30, 999], body_cap_bytes=4194304, primary_groups=8,
                questions_per_group=12, prediction_arms=3, official_reviewed_only_for_window=True,
                unreviewed_complete_identity_for_closure_only=True, missing_identity_stops_body=True,
                historical_exclusions=q.C.pin(q.C.canonical(summary)), historical_manifest=summary,
                labeled_or_media_work_blocked_until_independent_identity_screen=True, actual_HTTP=0)
    OUTPUT.mkdir()
    assert q.C.write(OUTPUT / "historical-identities-private.json", private) == private_pin
    q.C.write(OUTPUT / "HISTORIC_SUMMARY.json", summary)
    result = q.C.write(OUTPUT / "COUNT_PLAN.json", plan)
    return dict(plan=result, historical_private_index=private_pin, projected_prior_rows=len(prior),
                historical_rows=len(historical), explicit_style18_exclusion_ids=len(style_ids),
                style18_scope=style_scope, planned_count_slots=len(queries), actual_HTTP=0)


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True, allow_nan=False))
