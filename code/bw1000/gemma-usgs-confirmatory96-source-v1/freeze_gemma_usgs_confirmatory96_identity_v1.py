"""Freeze the known historical event identities without opening any question or gold file."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path


PLAN = Path("coordination/tasks/training-t02-gemma-usgs-confirmatory96-preregistration-v1.json")
PLAN_SHA = "b5a9a308dd50c2a940da9f9eb2a6b2bfc23872cecd37ab0b5a07816d6790f1a6"
TABLE = Path("local-only/training-domain-usgs-catalog-table-reading-v1")
TABLE_VIEW = Path("local-only/training-domain-usgs-catalog-table-reading-v1-qwen-view/binding-private.json")
QWEN_SOURCE = Path("code/bw1000/training-qwen-usgs-dev80-base-adapter-v1")
T04_SOURCE = Path("code/bw1000/training-USGS-T04-blind25-native-CPU-v1/PLAN.json")
OUTPUT = Path("local-only/gemma-usgs-confirmatory96-v1-historic-identity")


def verified(path, wanted, size=None):
    with path.open("rb") as stream:
        raw = stream.read(1048577)
    if not 0 < len(raw) <= 1048576 or sha256(raw).hexdigest() != wanted:
        raise ValueError(f"identity differs for {path.name}")
    if size is not None and size != len(raw):
        raise ValueError(f"extent differs for {path.name}")
    return raw, {"path": path.as_posix(), "bytes": len(raw), "SHA256": wanted}


def read(path, wanted, size=None):
    raw, pin = verified(path, wanted, size)
    return json.loads(raw), pin


def event(event_id, aliases, epoch, lat, lon):
    latitude, longitude = float(lat), float(lon)
    if (
        not isinstance(event_id, str) or not event_id
        or not isinstance(aliases, (list, tuple)) or any(not isinstance(x, str) or not x for x in aliases)
        or type(epoch) is not int or not math.isfinite(latitude) or not math.isfinite(longitude)
        or not (-90 <= latitude <= 90 and -180 <= longitude <= 180)
    ):
        raise ValueError("invalid historical event identity")
    return {"id": event_id, "aliases": sorted({event_id, *aliases}), "time_UTC_epoch_ms": epoch,
            "values": {"latitude": latitude, "longitude": longitude}}


def legacy_event(row):
    return event(row["id"], row["aliases"], row["time_UTC_epoch_ms"],
                 row["values"]["latitude"], row["values"]["longitude"])


def encode(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def save(name, obj):
    raw = encode(obj)
    path = OUTPUT / name
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": path.as_posix(), "bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def main():
    plan, plan_pin = read(PLAN, PLAN_SHA)
    anchors = []
    proof = []
    for spec in plan["legacy_anchors"]:
        doc, pin = read(Path(spec["path"]), spec["SHA256"], spec["bytes"])
        anchors.append(doc)
        proof.append(pin)
    events = [legacy_event(row) for doc in anchors[:3] for row in doc["rows"]]
    windows, render, r26 = anchors[3:]
    recorded = {row["id"]: row for row in render["charts"]}
    if len(recorded) != 24 or set(recorded) != {row["id"] for row in windows["charts"]}:
        raise ValueError("render144 chart inventory mismatch")
    blind_events, blind_groups = set(), set()
    chart_ids, group_ids, media_hashes = set(), set(), set()
    for w in windows["charts"]:
        m = recorded[w["id"]]
        ids = [r["id"] for r in w["rows"]]
        if (w["group"], w["split"], ids) != (m["group"], m["split"], m["event_ids"]):
            raise ValueError("render144 chart identity or split mismatch")
        events.extend(map(legacy_event, w["rows"]))
        chart_ids.add(m["id"])
        group_ids.add(m["group"])
        media_hashes.add(render["outputs"][m["image"]]["SHA256"])
        if w["split"] == "blind":
            blind_groups.add(w["group"])
            blind_events.update(ids)
    if (len(blind_groups), len(blind_events)) != (2, 36) or len(media_hashes) != 24:
        raise ValueError("first blind or render144 identity count differs")
    for row in r26["features"]:
        p = row["properties"]
        lon, lat, _ = row["geometry"]["coordinates"]
        events.append(event(row["id"], tuple(x for x in p["ids"].split(",") if x), p["time"], lat, lon))
    if len(r26["features"]) != 216:
        raise ValueError("R26 source extent differs")

    t04 = json.loads(T04_SOURCE.read_bytes())
    t04_root = Path("local-only/training-USGS-reported-event-charts-v1-original/dataset")
    t04_manifest, pin = read(t04_root / "manifest-private.json", "9423f4b9b3b40d64260f6d8a43c7c50e4595bcc887767fc47fba5db3d9145390", 10105)
    if t04["dataset_manifest"] != {k: pin[k] for k in ("bytes", "SHA256")}:
        raise ValueError("T04 blind25 does not bind the original render")
    proof.append(pin)
    slots = {slot for row in t04_manifest["charts"] for slot in row["source_slots"]}
    if slots != {row["slot"] for row in anchors[0]["rows"]} or len(slots) != 50:
        raise ValueError("T04 render does not map to all fifty historical event slots")
    t04_blind = [x for x in t04_manifest["charts"] if x["split"] == "blind"]
    if len(t04_blind) != 5 or t04_manifest["outputs"]["blind/inputs.jsonl"]["SHA256"] != t04["blind_inputs"]["SHA256"]:
        raise ValueError("T04 bound five-chart blinded view differs")
    for row in t04_manifest["charts"]:
        chart_ids.add(row["id"])
        group_ids.add(row["group"])
        media_hashes.add(row["image"]["SHA256"])

    provenance, pin = read(TABLE / "source-provenance.json",
                           "40854b6e734e37fc9d0455fbc3abaf0d0e466b366faffd693d2837acbdc8ac84", 2431)
    proof.append(pin)
    view, pin = read(TABLE_VIEW, "762898a4d7da69935dc8bd92d1db16f16f5cc791d7e0379e2e6fe519d91e0567", 5252)
    if view["source_SHA256"] != proof[-1]["SHA256"]:
        raise ValueError("Qwen catalog generation provenance differs")
    proof.append(pin)
    qwen_package, pin = read(QWEN_SOURCE / "SOURCE_FILES.json",
                              "e2368f122d3ba4cb03b74ef1964cc417ee02eadaa3b16e0412386cda2bc491d4", 1118)
    proof.append(pin)
    dev_pin = qwen_package.get("files", qwen_package)["DEV_INPUTS.json"]
    qwen, pin = read(QWEN_SOURCE / "DEV_INPUTS.json", dev_pin["SHA256"], dev_pin["bytes"])
    proof.append(pin)
    bound_dev = view["data"]["dev"]
    media = {Path(k).name: v for k, v in bound_dev["media"].items()}
    if qwen["dataset_SHA256"] != bound_dev["inputs"]["SHA256"] or media != qwen["media"] or len(media) != 20:
        raise ValueError("Qwen Dev80 input or media does not bind the rendered view")
    for split, count in (("train", 130), ("dev", 121), ("blind", 127)):
        raw, pin = verified(TABLE / f"{split}-public-events.jsonl",
                            provenance["split"][split]["qualified_SHA256"])
        proof.append(pin)
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != count or (split == "dev" and bound_dev["events"] != count):
            raise ValueError("catalog table source event count differs")
        for row in rows:
            stamp = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
            if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(None):
                raise ValueError("table event lacks UTC")
            events.append(event(row["id"], [], round(stamp.timestamp() * 1000), row["latitude"], row["longitude"]))
    for split in view["data"].values():
        media_hashes.update(split["media"].values())

    if any(not isinstance(x, str) or len(x) != 64 or set(x) - set("0123456789abcdef") for x in media_hashes):
        raise ValueError("invalid historical media digest")

    unique_events = {encode(row): row for row in events}
    OUTPUT.mkdir(exist_ok=False)
    index_pin = save("metadata-only-identity.json", {
        "rows": list(unique_events.values()), "chart_ids": sorted(chart_ids), "group_ids": sorted(group_ids),
        "media_SHA256": sorted(media_hashes), "first_blind_groups": sorted(blind_groups)
    })
    result = {"state": "HISTORICAL_IDENTITY_CROSSWALK_FROZEN_BEFORE_ACQUISITION", "plan": plan_pin,
              "source_pins": proof, "index": index_pin, "source_extent": len(events),
              "deduplicated_event_records": len(unique_events), "first_blind_events": 36, "first_blind_groups": 2,
              "T04_source_slots": 50, "T04_blind_charts": 5, "Qwen_Dev80_source_events": 121,
              "Qwen_Dev80_media": 20, "R26_donor_events": 216, "acquisitions": 0, "gold_reads": 0,
              "raw_train_or_blind_prompt_reads": 0, "legacy_alias_coverage_complete": False}
    receipt = save("receipt.json", result)
    print(json.dumps({"receipt": receipt, "deduplicated_event_records": len(unique_events),
                      "Qwen_Dev80_source_events": 121, "T04_source_slots": 50, "first_blind_events": 36}))


if __name__ == "__main__":
    main()
