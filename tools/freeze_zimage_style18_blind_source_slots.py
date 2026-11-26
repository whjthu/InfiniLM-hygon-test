"""Bind the original Style18 blind slots to unused, verified USGS sources."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import runpy
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local-only"
STYLE = LOCAL / "t07-style18-heldout-prelabel-v1"
PUBLIC = LOCAL / "gemma-usgs-confirmatory96-prospective-v1"
BODIES = PUBLIC / "body12-after-counts-778c0c2e-original"
HISTORICAL = LOCAL / "gemma-usgs-confirmatory96-v1-historic-identity/metadata-only-identity.json"
TRANSPORT = ROOT / "code/bw1000/gemma-usgs-confirmatory96-prospective-v1/body_once.py"
PINS = {
    "plan": "a1e69159b7c39619eeecb64ebd5ac7fa66bb94b4a37a70bcec5bf2ed6e20c367",
    "original_source": "c2fe1f911bd6ff673533e8b18a21a5c850e930755e4e327931972f49214fa5a1",
    "original_gemma_selection": "49253df040cfb9648b1372e67816409bf6bc38e348fea359b6f71e0d34dabb5c",
    "historical_full_identity_only": "e22d89b8c58aea49e2291395b47a04b3c27aacf8b18e489cf945fe3a4c2e2885",
    "historical_catalog_identity_only": "15816f22037e9085b0c8b7d1e65b2794f3e78f6e5f9357179fa94f05ab9ddb48",
    "style18_reserved_crosswalk": "8d3277cf6f4d5d0bfa96b5bd06420331a3828ae9e83ab7d0a656a444fdf8f57c",
    "body_receipts": "fc28e739fcefb55efba04d038a45b259ac274f7569ab5b8b6285c84a9578db0f",
    "body_transport_source": "84909e12de31e3febee943c2167e47b871b95662884f603056b5e4a755eece4e",
}
UNUSED_ROUTES = ("G17-S14", "G22-S00", "G23-S10", "G28-S05")
CONTROL_HASHES = (
    "6f57f413b30a25d988d0efa932cb888679d546e663ea3d845ea70830e7da33b5",
    "7df18ff9b1abd020ef82e6c0feb9d9f3f09d6d53169119b83397171fb8b76afc",
    "a4d362e2da8294fd4f48875a3cf00f24c91115a78661fa566894bafd2825ceea",
    "6ddbabfa8a05245dc193e96e5a7c14d50263a74bf3a69004e40120954a9738df",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read(path, limit, expected):
    require(path.resolve(strict=True) == path, "Original evidence path must be canonical")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit and digest(data) == expected, "Original evidence bytes differ")
    return data


def identity_closure(history, candidates, distance):
    """Find transitive alias or 500 km/72 h connections without reading prior images."""
    rows = history + candidates
    cutoff = len(history)
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        a, b = sorted((root(left), root(right)))
        parents[b] = a

    aliases = {}
    chronological = sorted(range(len(rows)), key=lambda i: rows[i]["time_UTC_epoch_ms"])
    start = 0
    for i, idx in enumerate(chronological):
        row = rows[idx]
        for alias in sorted({row["id"], *row["aliases"]}):
            join(idx, aliases.setdefault(alias, idx))
        while row["time_UTC_epoch_ms"] - rows[chronological[start]]["time_UTC_epoch_ms"] > 72 * 3_600_000:
            start += 1
        for previous in chronological[start:i]:
            if root(previous) != root(idx) and distance(rows[previous], row) <= 500:
                join(previous, idx)

    old_roots = {root(i) for i in range(cutoff)}
    groups = defaultdict(list)
    rejected = 0
    for i, row in enumerate(candidates, cutoff):
        if root(i) in old_roots:
            rejected += 1
        else:
            groups[root(i)].append(row)
    return groups, rejected


def load_sources():
    read(TRANSPORT, 65536, PINS["body_transport_source"])
    transport = runpy.run_path(str(TRANSPORT))
    policy, verify_body = transport["sources"]()
    routes = {route["query_id"]: route for route in transport["inputs"](policy)}
    manifest = json.loads(read(BODIES / "BODY_RECEIPTS.json", 16384, PINS["body_receipts"]))
    receipts = {row["query_id"]: row for row in manifest}
    require(len(routes) == len(manifest) == len(receipts) == 12 and routes.keys() == receipts.keys(),
            "Expected the original twelve official receipts")
    selected = json.loads(read(BODIES / "PRELABEL_SELECTION_PRIVATE.json", 32768, PINS["original_gemma_selection"]))
    gemma = [row for chart in selected for row in chart["rows"]]
    used = {row["route"] for row in gemma}
    require(len(gemma) == 96 and len(used) == 8 and sorted(routes.keys() - used) == list(UNUSED_ROUTES),
            "A proposed source was used in the Gemma confirmation")
    crosswalk = json.loads(read(PUBLIC / "style18-reserved-identity-original-readback-v1.json", 4096,
                               PINS["style18_reserved_crosswalk"]))
    require(crosswalk["prospective_selection"]["SHA256"] == PINS["original_gemma_selection"]
            and crosswalk["original_historical_identity"]["SHA256"] == PINS["historical_full_identity_only"]
            and crosswalk["full_Style18_counts"] == dict(charts=24, events=144, groups=7, media=24)
            and crosswalk["actual_old_image_reads"] == crosswalk["actual_gold_reads"] == 0,
            "The reserved Style18 identity-only exclusion is incomplete")
    legacy = json.loads(read(HISTORICAL, 225451, PINS["historical_full_identity_only"]))
    index = json.loads(read(PUBLIC / "historical-identities-private.json", 934320,
                             PINS["historical_catalog_identity_only"]))
    history = legacy["rows"] + index["original_history"] + index["previous_body_identities"] + gemma
    raw_rows = []
    evidence = {}
    for ident in UNUSED_ROUTES:
        route = routes[ident]
        receipt = receipts[ident]
        url = urlsplit(receipt["url"])
        require(receipt["url"] == route["body_url"] and receipt["status"] == "COMPLETE"
                and receipt["http_status"] == 200 and receipt["request_attempts"] == 1
                and (url.scheme, url.netloc, url.path) == ("https", "earthquake.usgs.gov", "/fdsnws/event/1/query"),
                "An unused snapshot no longer binds its original official request")
        body = receipt["body"]
        original = read(BODIES / f"{ident}.geojson", policy.CAP, body["SHA256"])
        parsed, flags = verify_body(policy, original, route)
        require(len(original) == body["bytes"] and len(parsed) == receipt["body_rows"]
                and flags["window_eligible"] == receipt["preferred_US"], "Official snapshot scope differs")
        evidence[ident] = {"url": receipt["url"], "body": body, "source_rows": len(parsed),
                           "eligible_USGS_rows": flags["window_eligible"]}
        raw_rows.extend(parsed)
    return history, raw_rows, legacy, selected, routes, evidence, policy.M


def pick_windows(clean, legacy, gemma, methods):
    roster = defaultdict(list)
    for closure_group in clean.values():
        owners = {row["route"] for row in closure_group}
        if len(owners) == 1:
            for row in closure_group:
                if row["preferred_US"]:
                    roster[row["route"]].append(row)
    excluded_charts = set(legacy["chart_ids"]) | {row["id"] for row in gemma}
    excluded_groups = set(legacy["group_ids"]) | {row["group"] for row in gemma}
    windows = {}
    for route in UNUSED_ROUTES:
        order = sorted(roster[route], key=lambda row: (row["time_UTC_epoch_ms"], row["id"]))
        group = digest(methods.canonical(sorted(row["id"] for row in order)))
        if group in excluded_groups:
            continue
        for start in range(0, len(order) - 5, 6):
            rows = order[start:start + 6]
            chart = digest(methods.canonical([row["id"] for row in rows]))
            varied = all(len({methods.display(row["values"][key]) for row in rows}) >= 2
                         for key in ("magnitude", "depth_km"))
            if chart not in excluded_charts and varied:
                windows[route] = (group, chart, rows)
                break
    return windows, {name: len(roster[name]) for name in UNUSED_ROUTES}


def prompt(rows, methods):
    days = [datetime.fromtimestamp(row["time_UTC_epoch_ms"] / 1000, timezone.utc).date() for row in rows]
    values = [
        f"{label} {methods.display(row['values']['magnitude'])}/{methods.display(row['values']['depth_km'])}"
        for label, row in zip("ABCDEF", rows, strict=True)
    ]
    return (f"Create a two-panel scientific figure for six USGS-reviewed earthquake records, {min(days)} to "
            f"{max(days)} UTC. Use bars for reported magnitude above and dot markers for depth in km below. "
            f"Keep A–F in chronological order. Magnitude/depth pairs: {', '.join(values)}. "
            "Add clear panel titles, labels and units. Credit the U.S. Geological Survey.")


def build():
    original = json.loads(read(STYLE / "PRELABEL.json", 16384, PINS["plan"]))
    read(STYLE / "freeze.py", 16384, PINS["original_source"])
    require(original["schema"] == "STYLE18_HELDOUT_PRELABEL_V1" and original["planned_images"] == 32
            and original["generation_GO"] is False, "The frozen 16-pair design changed")
    history, raw_rows, legacy, gemma, routes, evidence, methods = load_sources()
    clean, rejected = identity_closure(history, raw_rows, methods.distance)
    windows, capacities = pick_windows(clean, legacy, gemma, methods)
    domains = []
    for i, route in enumerate(UNUSED_ROUTES, 1):
        item = {"slot": f"D{i}", "source": evidence[route], "grid": routes[route]["grid_id"],
                "eligible_identity_rows": capacities[route], "source_attested": route in windows}
        if route in windows:
            group, chart, rows = windows[route]
            text = prompt(rows, methods)
            item.update(group=group, chart=chart, events=rows, prompt=text, prompt_SHA256=digest(text.encode()),
                        content_checklist=["Two panels in the stated order", "Labels A–F in chronological order",
                                           "Correct magnitude and depth coordinates", "Depth unit shown in km",
                                           "Source credit present", "No unsupported events"])
        domains.append(item)
    require(len({item["grid"] for item in domains}) == 4
            and len({group for group, _chart, _rows in windows.values()}) == len(windows)
            and len({row["id"] for _group, _chart, rows in windows.values() for row in rows}) == 6 * len(windows),
            "Independent group or event counts differ")
    controls = []
    for i, (entry, expected) in enumerate(zip(original["prompts"][4:], CONTROL_HASHES, strict=True), 1):
        require(entry["slot"] == f"C{i}" and entry["uses_external_asset"] is False
                and entry["prompt_SHA256"] == digest(entry["proposed_prompt"].encode()) == expected,
                "A reviewed original synthetic control differs")
        controls.append({"slot": entry["slot"], "prompt": entry["proposed_prompt"],
                         "prompt_SHA256": expected, "content_checklist": entry["content_checklist"],
                         "source_attested": True, "external_images_or_labels": False})
    prompts = domains + controls
    summary = {
        "state": "STYLE18_BLIND_SOURCE_SLOTS_FROZEN_AWAITING_GENERATION_INPUT_AND_REVIEW",
        "official_unused_snapshots_verified": len(evidence),
        "source_candidate_events": len(raw_rows),
        "historical_identity_vertices": len(history),
        "candidate_events_rejected_by_prior_transitive_closure": rejected,
        "domain_sources_attested": len(windows),
        "independent_control_origins_checked": len(controls),
        "planned_images": original["planned_images"],
        "planned_pairs": len(original["pairs"]),
        "planned_updates": 0,
        "new_HTTP_acquisitions": 0,
        "old_blind_bodies_images_answers_predictions_read": 0,
        "new_model_generations": 0,
        "generation_GO": False,
        "formal_delivery_increment": 0,
        "source_credit": "U.S. Geological Survey",
        "evidence_SHA256": PINS,
        "remaining_before_generation": [
            "Independent review of exact new domain prompts and their source-grounded checklist",
            "Current target verification of the original initializer and continuous CP25",
            "Independent blinded assessment custody and unique execution claim",
        ],
    }
    private = dict(summary, prompts=prompts, seeds=original["pairs"], arms=original["arms"],
                   generation_settings=original["original_generation_settings"])
    return private, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_output", type=Path)
    args = parser.parse_args()
    require(args.private_output.resolve().is_relative_to(LOCAL), "Keep private prompt bytes under local-only")
    private, summary = build()
    raw = canonical(private)
    with args.private_output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(summary, private_source_packet_SHA256=digest(raw)),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
