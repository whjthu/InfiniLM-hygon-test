"""Fetch the twelve prospectively matched official catalogs and audit their identities."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from types import FunctionType


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = ROOT / "local-only/gemma-usgs-confirmatory96-prospective-v1"
COUNTS = STAGE / "public-counts-for-body-v1"
OUTPUT = STAGE / "body12-after-counts-778c0c2e-original"
PLAN_SHA = "148dd36fd682d5f99217f5035deac96cda2e6bbed4137b6793aec92fb76bc49e"
COUNTS_SHA = "83ce92d149b4baa82669a6260ef82c060be64febeebeccf45b8a765655ea4f66"
MATCH_SHA = "778c0c2e45bde107432d4a590ec16e9c28432377726427d7c73acd2bf5a8a3be"
HISTORY_SHA = "15816f22037e9085b0c8b7d1e65b2794f3e78f6e5f9357179fa94f05ab9ddb48"


def sources():
    path = ROOT / "code/bw1000/gemma-usgs-body12-closure-audit-v1/audit.py"
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != "67048a168c2f36e13c2cb880cfffb6351433aec7b384fe921ed728fe7023cdb6":
        raise ValueError("ORIGINAL_PROVENANCE_RULE_CHANGED")
    audit = runpy.run_path(str(path))
    return audit["load_policy"](), audit["audit_body"]


def inputs(q):
    plan = json.loads(q.read(HERE / "COUNT_PLAN.json", 65536, PLAN_SHA))
    counts = json.loads(q.read(COUNTS / "COUNTS.json", 20514, COUNTS_SHA))
    match = json.loads(q.read(COUNTS / "MATCH.json", 7324, MATCH_SHA))
    routes = q.verify_match(plan, counts, match)
    q.need(len({r["grid_id"] for r in routes}) == len({r["slot_id"] for r in routes}) == 12,
           "EXACT_PROSPECTIVE_ROUTE_CARDINALITY")
    return routes


def check_authority(q, routes, path, expected_sha):
    raw = q.read(path, 4096, expected_sha)
    value = json.loads(raw)
    q.need(value.get("plan_SHA256") == PLAN_SHA and value.get("counts_SHA256") == COUNTS_SHA
           and value.get("match_SHA256") == MATCH_SHA and value.get("history_SHA256") == HISTORY_SHA
           and value.get("routes_SHA256") == q.digest(q.C.canonical(routes))
           and value.get("source_SHA256") == q.digest(Path(__file__).read_bytes())
           and re.fullmatch("[0-9a-f]{40}", value.get("publication", "")) is not None
           and value.get("BODY_GO") is True and value.get("no_labels_or_predictions") is True,
           "EXACT_PUBLISHED_PROSPECTIVE_BODY_AUTHORITY")
    return value


def fetch(q, audit_body, routes, authority, opener):
    OUTPUT.mkdir()
    q.C.write(OUTPUT / "RUN_INTENT.json", dict(routes_SHA256=authority["routes_SHA256"],
              publication=authority["publication"], planned=12, max_attempts_per_route=1))
    # Keep the original bounded HTTP transport and receipt writer. The prospective
    # rule distinguishes events eligible for windows from identity-only vertices.
    def parse(raw, route):
        return audit_body(q, raw, route)[0]
    request = FunctionType(q.one_route.__code__, dict(q.one_route.__globals__, body=parse))
    reports = [request(route, OUTPUT, opener)[1] for route in routes]
    pin = q.C.write(OUTPUT / "BODY_RECEIPTS.json", reports)
    return dict(body_receipts=pin, terminal=len(reports), completed=sum(r["status"] == "COMPLETE" for r in reports),
                planned=12, emitted_prompts=0, gold_generated=0)


def analyze(q, audit_body, routes, authority, receipt_sha):
    intent = json.loads(q.read(OUTPUT / "RUN_INTENT.json", 4096))
    q.need(intent == dict(routes_SHA256=authority["routes_SHA256"], publication=authority["publication"],
           planned=12, max_attempts_per_route=1), "EXACT_PROSPECTIVE_BODY_INTENT")
    receipt_bytes = q.read(OUTPUT / "BODY_RECEIPTS.json", 32768, receipt_sha)
    reports = json.loads(receipt_bytes)
    q.need(len(reports) == len(routes) == 12, "ALL_PROSPECTIVE_BODY_SLOTS_REQUIRED")
    events = []
    bodies = {}
    completed = 0
    for rule, receipt in zip(routes, reports, strict=True):
        ident = rule["query_id"]
        q.need(receipt["query_id"] == ident and receipt["url"] == rule["body_url"]
               and receipt["planned_count"] == rule["count"] and receipt["request_attempts"] == 1,
               "ORIGINAL_PROSPECTIVE_ROUTE_RECEIPT")
        original = q.read(OUTPUT / (ident + "-receipt.json"), 8192)
        q.need(original == q.C.canonical(receipt), "EXACT_PER_ROUTE_BODY_RECEIPT")
        if receipt["status"] == "COMPLETE":
            bound = receipt["body"]
            raw = q.read(OUTPUT / (ident + ".geojson"), q.CAP, bound["SHA256"])
            rows, disposition = audit_body(q, raw, rule)
            q.need(q.C.pin(raw) == bound and len(rows) == receipt["body_rows"]
                   and disposition["window_eligible"] == receipt["preferred_US"], "RAW_BODY_OR_CLASSIFICATION_DIFFERS")
            events.extend(rows)
            completed += 1
            bodies[ident] = dict(body=bound, raw_events=len(rows), dispositions=dict(disposition))
        else:
            q.need(receipt["status"] == "FAILED", "UNRECOGNIZED_BODY_TERMINAL")
            bodies[ident] = {k: receipt[k] for k in ("status", "http_status", "error_type", "gate", "body") if k in receipt}
    result = dict(body_receipts=q.C.pin(receipt_bytes), bodies=bodies, terminal=12, planned=12, completed=completed,
                  qualified_groups=None, selected_charts=0, selected_events=0, emitted_prompts=0, gold_generated=0,
                  actual_model_predictions=0, blind_EFFECT_GO=False, formal_delivery_increment=0)
    if completed != 12:
        result["state"] = "PROSPECTIVE_BODY_FAILURE_STOP_NO_REPLACEMENT"
    else:
        index = json.loads(q.read(STAGE / "historical-identities-private.json", 934320, HISTORY_SHA))
        h = index["original_history"] + index["previous_body_identities"]
        q.need(len(h) == 6835 and len(index["explicit_style18_event_exclusions"]) == 108
               and set(index["explicit_style18_event_exclusions"]) <= {a for row in h for a in row["aliases"]},
               "HISTORICAL_CATALOG_AND_STYLE18_EVENT_COVERAGE")
        legacy = json.loads(q.read(q.LEGACY, 225451, q.C.IDENTITY_PIN["SHA256"]))
        prereg = json.loads(q.read(q.PREREG, 65536, q.PREREG_SHA))
        windows, closure = q.qualify(h, events, legacy, prereg["selection"])
        result.update(closure)
        result["state"] = "FEWER_THAN_EIGHT_GROUPS_STOP_NO_REPLACEMENT"
        if windows:
            q.need(len(windows) == 16 and len({w["group"] for w in windows}) == 8
                   and len({row["id"] for w in windows for row in w["rows"]}) == 96, "EXACT_PRELABEL_DENOMINATORS")
            result.update(state="EIGHT_PROSPECTIVE_GROUPS_PRELABEL_MEDIA_AND_IDENTITY_REVIEW_PENDING",
                          selected_events=96,
                          selection_private=q.C.write(OUTPUT / "PRELABEL_SELECTION_PRIVATE.json", windows))
    result["reserved_style18_blind_identity_coverage_verified"] = False
    pin = q.C.write(OUTPUT / "QUALIFICATION.json", result)
    return dict(receipt=pin, **{k: v for k, v in result.items() if k != "bodies"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "fetch", "analyze"))
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--authority-sha")
    parser.add_argument("--receipt-sha")
    args = parser.parse_args()
    q, audit_body = sources()
    routes = inputs(q)
    if args.phase == "preflight":
        result = dict(routes_SHA256=q.digest(q.C.canonical(routes)), planned=12, HTTP=0, gold_generated=0)
    else:
        if not (args.authority and args.authority_sha and (bool(args.receipt_sha) == (args.phase == "analyze"))):
            parser.error("fetch and analyze require exact authority, and analyze requires the body receipt")
        authority = check_authority(q, routes, args.authority, args.authority_sha)
        if args.phase == "fetch":
            result = fetch(q, audit_body, routes, authority, q.build_opener(q.NoRedirect))
        else:
            result = analyze(q, audit_body, routes, authority, args.receipt_sha)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
