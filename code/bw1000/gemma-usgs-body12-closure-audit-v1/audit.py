"""Audit capacity of the original catalog bodies without producing evaluation inputs."""

from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA = "a6f90231b8200dbb67b7e881126f31aa43333f28a74ec9ab7aabd624fc8fb0a1"
RECEIPTS_SHA = "4288fe90202febbb314e5b48fc6198f7945f507b3486fa1d0c32eb2c5f678245"
ORIGINAL_QUALIFICATION_SHA = "35902fac942804249520fffc1d0bac9738b88361ce08a69ea3a462b65331d078"
SOURCE = ROOT / "code/bw1000/gemma-usgs-confirmatory96-body12-v1/qualify.py"
COUNTS = ROOT / "local-only/gemma-usgs-confirmatory96-body12-v1/original-public-counts"


def load_policy():
    raw = SOURCE.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("ORIGINAL_BODY_POLICY_CHANGED")
    module = ModuleType("original_body_policy")
    module.__file__ = str(SOURCE)
    exec(compile(raw, str(SOURCE), "exec"), module.__dict__)
    return module


def identity(q, feature):
    props = feature["properties"]
    q.need(feature["type"] == "Feature" and feature["geometry"]["type"] == "Point"
           and isinstance(props["ids"], str) and props["ids"].strip(","), "IDENTITY_REQUIRED")
    lon, lat, _ = feature["geometry"]["coordinates"]
    us_preferred = props["net"] == "us" and "us" in props["sources"].strip(",").split(",")
    if us_preferred and props["status"] != "reviewed":
        q.need(isinstance(props["status"], str) and props["status"], "REVIEW_STATUS_REQUIRED")
        magnitude = props["mag"]
        q.need(props["type"] == "earthquake" and type(magnitude) in (int, Decimal)
               and Decimal(magnitude).is_finite() and magnitude >= Decimal("4.5")
               and isinstance(props["magType"], str) and props["magType"], "OTHER_PROVENANCE_REJECTED")
        aliases = [name for name in props["ids"].split(",") if name]
        row = q.C.event(dict(id=feature["id"], aliases=aliases, time_UTC_epoch_ms=props["time"],
                             values=dict(latitude=lat, longitude=lon)))
        return row, "not_reviewed"
    row, preferred = q.C.body_record(feature)
    return row, "window_eligible" if preferred else "not_us_preferred"


def audit_body(q, raw, route):
    doc = json.loads(raw, parse_float=Decimal, parse_constant=q.rejects_nonfinite,
                     object_pairs_hook=q.strict_object)
    features = doc["features"]
    meta = doc["metadata"]
    q.need(doc["type"] == "FeatureCollection" and isinstance(features, list)
           and 0 < len(features) < 1000, "BODY_EXTENT")
    q.need(meta.get("status") == 200 and ("count" not in meta
           or (type(meta["count"]) is int and meta["count"] == len(features))), "BODY_METADATA")
    rows = []
    dispositions = Counter()
    last = None
    for feature in features:
        row, disposition = identity(q, feature)
        lon, lat, depth = feature["geometry"]["coordinates"]
        stamp = row["time_UTC_epoch_ms"]
        q.need(all(type(v) in (int, Decimal) for v in (lat, lon, depth))
               and q.C.contains(route, lat, lon, stamp)
               and (last is None or last <= stamp), "EXACT_SOURCE_BOUNDS_AND_ORDER")
        last = stamp
        dispositions[disposition] += 1
        preferred = disposition == "window_eligible"
        row.update(preferred_US=preferred, route=route["query_id"])
        if preferred:
            props = feature["properties"]
            magnitude = props["mag"]
            q.need(type(magnitude) in (int, Decimal) and -100 <= depth <= 1000
                   and type(props["updated"]) is int and isinstance(props["magType"], str),
                   "WINDOW_NUMERIC_PROVENANCE")
            row.update(preferred_net="us", magType=props["magType"], updated_epoch_ms=props["updated"],
                       values=dict(latitude=str(lat), longitude=str(lon), depth_km=str(depth),
                                   magnitude=str(magnitude)))
        rows.append(row)
    return rows, dispositions


def audit(q):
    routes, history, legacy, selection = q.preflight(COUNTS)
    old = json.loads(q.read(q.OUTPUT / "QUALIFICATION.json", 8192, ORIGINAL_QUALIFICATION_SHA))
    q.need(old["state"] == "INCOMPLETE_BODY12_STOP_NO_REPLACEMENT"
           and old["planned"] == old["terminal"] == 12 and old["completed"] == 11
           and old["qualified_groups"] is None and old["gold_reads"] == old["predictions"] == 0,
           "ORIGINAL_REJECTION_MUST_BE_PRESERVED")
    receipts = json.loads(q.read(q.OUTPUT / "BODY_RECEIPTS.json", 16384, RECEIPTS_SHA))
    q.need(len(receipts) == len(routes) == 12, "ORIGINAL_ROUTES_DENOMINATOR")
    rows = []
    total = Counter()
    by_query = {}
    for route, receipt in zip(routes, receipts, strict=True):
        query_id = route["query_id"]
        q.need(receipt["query_id"] == query_id and receipt["url"] == route["body_url"]
               and receipt["request_attempts"] == 1 and receipt["http_status"] == 200
               and receipt["planned_count"] == route["count"], "ORIGINAL_ONE_HTTP_RECEIPT")
        original_receipt = q.read(q.OUTPUT / (query_id + "-receipt.json"), 8192)
        q.need(original_receipt == q.C.canonical(receipt), "ORIGINAL_AGGREGATE_RECEIPT_MISMATCH")
        pin = receipt["body"]
        raw = q.read(q.OUTPUT / (query_id + ".geojson"), q.CAP, pin["SHA256"])
        q.need(q.C.pin(raw) == pin, "ORIGINAL_BODY_SIZE_OR_HASH")
        features, dispositions = audit_body(q, raw, route)
        if receipt["status"] == "COMPLETE":
            q.need(len(features) == receipt["body_rows"] and not dispositions["not_reviewed"],
                   "ORIGINAL_COMPLETE_ROUTE_DIFFERENCE")
        else:
            q.need(receipt["status"] == "FAILED" and receipt["gate"] == "MATCH_PREVIOUS_PROVENANCE_FILTER"
                   and dispositions["not_reviewed"] > 0, "UNREPRODUCED_OR_UNSCOPED_ORIGINAL_REJECTION")
        rows.extend(features)
        total.update(dispositions)
        by_query[query_id] = dict(body=pin, original_status=receipt["status"],
                                  raw_events=len(features), window_eligible=dispositions["window_eligible"],
                                  closure_only_not_reviewed=dispositions["not_reviewed"],
                                  closure_only_other_network=dispositions["not_us_preferred"])
    windows, capacity = q.qualify(history, rows, legacy, selection)
    q.need(len(rows) == sum(total.values()) and capacity["preferred_US"] == total["window_eligible"]
           and all(row["preferred_US"] for window in windows for row in window["rows"]), "CAPACITY_DENOMINATOR")
    capacity.pop("selected_charts")
    return dict(state="ORIGINAL_BODIES_REUSED_FOR_OFFLINE_CAPACITY_ONLY",
                original_gate_preserved=old["state"], original_accepted_bodies=old["completed"],
                original_body_receipts_SHA256=RECEIPTS_SHA, original_bodies_verified=len(by_query),
                historical_identity_rows=len(history), per_query=by_query, dispositions=dict(total),
                capacity=capacity, illustrative_window_count=len(windows),
                new_HTTP_requests=0, prompts_generated=0, gold_generated=0, predictions=0,
                independent_confirmatory_effect_qualified=False, formal_delivery_increment=0)


if __name__ == "__main__":
    print(json.dumps(audit(load_policy()), sort_keys=True, allow_nan=False))
