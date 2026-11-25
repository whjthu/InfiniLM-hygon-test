"""Fetch the fixed public catalog once and qualify it where the audit index can be read."""

import argparse
import json
from pathlib import Path
import sys
from urllib.request import build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify


COUNTS = qualify.ROOT / "local-only/gemma-usgs-confirmatory96-body12-v1/original-public-counts"
ROUTES_SHA = "b1174560297f47887dc4d6de9391759db5a6abf4ed6643577237dc9d213d7477"
SOURCE_SHA = "a6f90231b8200dbb67b7e881126f31aa43333f28a74ec9ab7aabd624fc8fb0a1"


def inputs():
    plan = json.loads(qualify.read(qualify.COUNT / "FROZEN/COUNT32_PLAN.json", 65536, qualify.PLAN_SHA))
    counts = json.loads(qualify.read(COUNTS / "COUNTS.json", 20514, qualify.COUNT_SHA))
    match = json.loads(qualify.read(COUNTS / "MATCH.json", 7318, qualify.MATCH_SHA))
    routes = qualify.verify_match(plan, counts, match)
    qualify.need(qualify.digest(qualify.C.canonical(routes)) == ROUTES_SHA, "FROZEN_ROUTE_ORDER_CHANGED")
    return plan, routes


def fetch(routes, authority):
    folder = qualify.OUTPUT
    folder.mkdir()
    qualify.C.write(folder / "RUN_INTENT.json", dict(
        routes=routes, publication=authority["published_commit"], ordered_routes_SHA256=ROUTES_SHA,
        policy="only the frozen twelve official body queries, one attempt each",
    ))
    opener = build_opener(qualify.NoRedirect)
    reports = []
    for route in routes:
        _, report = qualify.one_route(route, folder, opener)
        reports.append(report)
    receipt = qualify.C.write(folder / "BODY_RECEIPTS.json", reports)
    return dict(body_receipts=receipt, planned=12, terminal=len(reports),
                completed=sum(row["status"] == "COMPLETE" for row in reports), gold_reads=0)


def analyze(plan, routes, authority, receipt_sha):
    folder = qualify.OUTPUT
    intent = json.loads(qualify.read(folder / "RUN_INTENT.json", 32768))
    qualify.need(intent["routes"] == routes and intent["ordered_routes_SHA256"] == ROUTES_SHA
                 and intent["publication"] == authority["published_commit"], "EXACT_BODY_RUN_INTENT")
    raw = qualify.read(folder / "BODY_RECEIPTS.json", 32768, receipt_sha)
    reports = json.loads(raw)
    qualify.need(len(reports) == len(routes), "FULL_BODY_RECEIPT_DENOMINATOR")
    new = []
    completed = 0
    for route, report in zip(routes, reports, strict=True):
        qualify.need(report["query_id"] == route["query_id"] and report["url"] == route["body_url"]
                     and report["planned_count"] == route["count"] and report["request_attempts"] == 1,
                     "BODY_RECEIPT_DOES_NOT_BIND_FROZEN_ROUTE")
        saved = qualify.read(folder / (route["query_id"] + "-receipt.json"), 8192)
        qualify.need(saved == qualify.C.canonical(report), "BODY_RECEIPT_AGGREGATE_MISMATCH")
        if report["status"] == "COMPLETE":
            expected = report["body"]
            original = qualify.read(folder / (route["query_id"] + ".geojson"), qualify.CAP, expected["SHA256"])
            rows = qualify.body(original, route)
            qualify.need(qualify.C.pin(original) == expected and len(rows) == report["body_rows"],
                         "BODY_RECHECK_MISMATCH")
            new.extend(rows)
            completed += 1
        else:
            qualify.need(report["status"] == "FAILED", "BODY_RECEIPT_UNKNOWN_STATE")
    result = dict(planned=12, terminal=12, completed=completed, observed_features=len(new),
                  gold_reads=0, gold_generated=0, predictions=0, formal_delivery_increment=0,
                  qualified_groups=None, selected_charts=0, body_receipts=qualify.C.pin(raw))
    if completed != 12:
        result["state"] = "INCOMPLETE_BODY12_STOP_NO_REPLACEMENT"
    else:
        history, legacy, selection = qualify.history_inputs(plan)
        windows, diagnostics = qualify.qualify(history, new, legacy, selection)
        result.update(diagnostics)
        if windows:
            result["state"] = "EIGHT_GROUPS_PRELABEL_SELECTION_PENDING_MEDIA_CROSSCHECK"
            result["selection"] = qualify.C.write(folder / "selection-private.json", windows)
        else:
            result["state"] = "FEWER_THAN_EIGHT_GROUPS_STOP_NO_REPLACEMENT"
    return dict(receipt=qualify.C.write(folder / "QUALIFICATION.json", result), **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fetch", "analyze"))
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--body-receipts-sha")
    args = parser.parse_args()
    if bool(args.body_receipts_sha) != (args.phase == "analyze"):
        parser.error("only analyze requires the exact aggregate body receipt SHA-256")
    qualify.read(Path(qualify.__file__), 65536, SOURCE_SHA)
    plan, routes = inputs()
    authority = qualify.approval(args.authority, args.authority_sha, routes)
    qualify.need(authority.get("entry_SHA256") == qualify.digest(Path(__file__).read_bytes()),
                 "EXACT_BODY_ENTRY_REQUIRED")
    if args.phase == "fetch":
        result = fetch(routes, authority)
    else:
        result = analyze(plan, routes, authority, args.body_receipts_sha)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
