"""Acquire the four preregistered official snapshots and select input-only windows."""

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import runpy
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, build_opener

from freeze_gemma_usgs_confirmatory96_identity_v1 import PLAN, PLAN_SHA, OUTPUT as IDENTITY, read, verified


OUTPUT = Path("local-only/gemma-usgs-confirmatory96-v1-four-month-source")
IDENTITY_SHA = "d0bc19d20e0fa09df6db253fb5b679a943c0ed1a6bf9c73ab715b74d09a11b89"
OLD_ALGORITHM = Path("code/bw1000/training-USGS-multievent-source-v2/prepare.py")
OLD_ALGORITHM_SHA = "c0b063f0b48dbfe24f2546e390e8ff705aa05cd71605f029552346c19f8e0858"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def save(name, raw):
    if not isinstance(raw, bytes):
        raw = canonical(raw) + b"\n"
    path = OUTPUT / name
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": path.as_posix(), "bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def month(plan, start, end, opener):
    fixed = plan["acquisition"]
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    params = dict(fixed["static_query"], starttime=lo.isoformat(timespec="seconds").replace("+00:00", "Z"),
                  endtime=(hi - timedelta(milliseconds=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z"))
    url = fixed["endpoint"] + "?" + urlencode(params)
    meta = {"url": url, "query_started_UTC": now(), "attempts": 1}
    save(start + "-intent.json", meta)
    rows = []
    try:
        cap = fixed["body_cap_bytes"]
        with opener.open(url, timeout=14) as response:
            raw = response.read(cap + 1)
            meta.update(status=response.status, retrieved_UTC=now(), server_date=response.headers.get("Date"),
                        content_type=response.headers.get_content_type(), content_bytes=len(raw))
        if meta["status"] != 200 or not 0 < len(raw) <= cap:
            raise ValueError("source body limit or HTTP status")
        meta["body"] = save(start + ".geojson", raw)
        data = json.loads(raw, parse_float=Decimal)
        features = data["features"]
        meta["received_features"] = len(features)
        if data["type"] != "FeatureCollection" or not 0 < len(features) < 1000:
            raise ValueError("empty, non-catalog, or possibly truncated query")
        if "count" in data["metadata"] and data["metadata"]["count"] != len(features):
            raise ValueError("catalog metadata count differs")
        earliest, latest = round(lo.timestamp() * 1000), round(hi.timestamp() * 1000)
        inventories = Counter()
        seen = set()
        last = earliest
        for feature in features:
            properties = feature["properties"]
            key, stamp = feature["id"], properties["time"]
            if not isinstance(key, str) or not key or key in seen or type(stamp) is not int or not last <= stamp < latest:
                raise ValueError("catalog IDs or chronological UTC order differ")
            seen.add(key)
            last = stamp
            net = properties["net"]
            inventories[net] += 1
            contributors = set(properties["sources"].strip(",").split(","))
            if net != "us" or "us" not in contributors:
                continue
            geom = feature["geometry"]
            if properties["type"] != "earthquake" or properties["status"] != "reviewed" or geom["type"] != "Point":
                raise ValueError("catalog type or reviewed status differs")
            x, y, d = geom["coordinates"]
            mag = Decimal(str(properties["mag"]))
            vals = [Decimal(str(v)) for v in (x, y, d, mag)]
            if (not all(v.is_finite() for v in vals) or not (-180 <= vals[0] <= 180 and -90 <= vals[1] <= 90
                    and -100 <= vals[2] <= 1000 and vals[3] >= Decimal("4.5")) or not properties["magType"]):
                raise ValueError("invalid source numeric record")
            aliases = sorted({key, *(x for x in properties["ids"].split(",") if x)})
            rows.append({"id": key, "aliases": aliases, "time_UTC_epoch_ms": stamp, "preferred_net": net,
                         "magType": properties["magType"], "updated_epoch_ms": properties["updated"],
                         "values": dict(zip(("longitude", "latitude", "depth_km", "magnitude"), map(str, vals)))})
        meta.update(preferred_network_counts=dict(inventories), eligible_numeric_rows=len(rows),
                    query_catalog=data["metadata"].get("api"), query_catalog_generated=data["metadata"].get("generated"),
                    state="SOURCE_COMPLETE")
        if not rows:
            raise ValueError("no provenance-qualified events")
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as exc:
        meta.update(state="SOURCE_FAILED", error_type=type(exc).__name__, error_summary=str(exc)[:180])
        rows.clear()
    return rows, save(start + "-receipt.json", meta), meta


def select(old, new, plan, methods):
    distance, cell, display = (methods[k] for k in ("distance", "cell", "display"))
    rows = old + new
    nold = len(old)
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i, j):
        a, b = root(i), root(j)
        if a != b:
            parents[max(a, b)] = min(a, b)

    aliases = {}
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["time_UTC_epoch_ms"])
    left = 0
    for position, i in enumerate(ordered):
        a = rows[i]
        for alias in a["aliases"]:
            join(aliases.setdefault(alias, i), i)
        while a["time_UTC_epoch_ms"] - rows[ordered[left]]["time_UTC_epoch_ms"] > 72 * 3600000:
            left += 1
        for j in ordered[left:position]:
            if root(i) != root(j) and distance(a, rows[j]) <= 500:
                join(i, j)
    old_roots = {root(i) for i in range(nold)}
    eligible = [i for i in range(nold, len(rows)) if root(i) not in old_roots]
    cells = {}
    for i in eligible:
        join(cells.setdefault(cell(rows[i]), i), i)
    groups = defaultdict(list)
    for i in eligible:
        groups[root(i)].append(rows[i])
    seed = plan["selection"]["seed"]
    capacities = []
    for group in groups.values():
        ordered_group = sorted(group, key=lambda row: (row["time_UTC_epoch_ms"], row["id"]))
        group_id = sha256(canonical(sorted(r["id"] for r in group))).hexdigest()
        order = sha256((seed + group_id).encode()).hexdigest()
        windows = []
        for offset in range(0, len(group) - 5, 6):
            window = ordered_group[offset:offset + 6]
            if any(len({display(r["values"][f]) for r in window}) < 2 for f in ("magnitude", "depth_km")):
                continue
            wid = sha256(canonical([r["id"] for r in window])).hexdigest()
            windows.append({"id": wid, "group": group_id, "rows": window})
        capacities.append((order, group_id, windows))
    usable = sorted((order, gid, windows) for order, gid, windows in capacities if len(windows) >= 2)
    required_groups = plan["selection"]["required_groups"]
    report = {"new_events": len(new), "excluded_via_legacy_closure": len(new) - len(eligible),
              "post_exclusion_events": len(eligible), "new_groups": len(groups),
              "groups_with_two_full_windows": len(usable), "required_groups": required_groups}
    if len(usable) < required_groups:
        return [], report
    return [w for _, _, windows in usable[:required_groups] for w in windows[:2]], report


def main():
    plan, plan_pin = read(PLAN, PLAN_SHA)
    evidence, evidence_pin = read(IDENTITY / "receipt.json", IDENTITY_SHA, 3489)
    if evidence["plan"] != plan_pin or evidence["gold_reads"] or evidence["acquisitions"]:
        raise ValueError("exact pre-acquisition identity witness required")
    ip = evidence["index"]
    old, _ = read(Path(ip["path"]), ip["SHA256"], ip["bytes"])
    verified(OLD_ALGORITHM, OLD_ALGORITHM_SHA)
    methods = runpy.run_path(str(OLD_ALGORITHM))
    OUTPUT.mkdir(exist_ok=False)
    opener = build_opener(NoRedirect)
    months = [month(plan, a, b, opener) for a, b in plan["acquisition"]["utc_month_windows"]]
    report = {"plan": plan_pin, "historical_witness": evidence_pin, "month_receipts": [pin for _, pin, _ in months],
              "actual_queries": len(months), "qualified_months": sum(m["state"] == "SOURCE_COMPLETE" for _, _, m in months),
              "gold_reads": 0, "gold_generated": 0, "actual_predictions": 0, "formal_delivery_credit": 0}
    if report["qualified_months"] != 4:
        report.update(state="SOURCE_INCOMPLETE_STOP_NO_ALTERNATE_MONTH")
    else:
        new = [row for rows, _, _ in months for row in rows]
        if len({r["id"] for r in new}) != len(new):
            report.update(state="DUPLICATE_PRIMARY_EVENT_ACROSS_MONTHS_STOP")
        else:
            charts, metrics = select(old["rows"], new, plan, methods)
            report.update(metrics)
            if charts:
                if ({c["id"] for c in charts} & set(old["chart_ids"]) or {c["group"] for c in charts} & set(old["group_ids"])
                        or len({r["id"] for c in charts for r in c["rows"]}) != 96):
                    raise ValueError("selected input identity differs from the frozen design")
                report.update(state="INPUT_WINDOWS_FROZEN_PENDING_RENDER_MEDIA_CROSSCHECK_AND_LABELS",
                              selected_charts=len(charts), selection=save("selection-private.json", charts))
            else:
                report.update(state="INSUFFICIENT_EVENT_GROUPS_STOP_NO_REFILL", selected_charts=0)
    pin = save("receipt.json", report)
    print(json.dumps({"receipt": pin, **{k: v for k, v in report.items() if isinstance(v, (str, int))}}, sort_keys=True))


if __name__ == "__main__":
    main()
