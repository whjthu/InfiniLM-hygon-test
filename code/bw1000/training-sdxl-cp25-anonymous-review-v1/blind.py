"""Prepare and score a sealed, arm-blind review of the twelve CP25 product images."""

import argparse
import json
from pathlib import Path
import secrets
import statistics
import struct
import zlib

from plan import ARMS, digest, frozen, need, put, read, wire
from score_cpu import final_pairs


CRITERIA = {
    "single_complete_target_object": "One recognizable target object, with no duplicate and no part cut by the image edge.",
    "plain_clean_background": "Uncluttered and visually plain background, with no other object or prominent scenery.",
}


def clean_png(raw):
    """Validate the original RGB pixels and omit ancillary chunks carrying metadata."""
    signature = b"\x89PNG\r\n\x1a\n"
    need(raw.startswith(signature), "EXPECTED_ORIGINAL_PNG")
    offset = len(signature)
    chunks = []
    while offset + 12 <= len(raw):
        size = struct.unpack_from(">I", raw, offset)[0]
        end = offset + size + 12
        need(end <= len(raw), "TRUNCATED_PNG_CHUNK")
        tag = raw[offset + 4:offset + 8]
        payload = raw[offset + 8:end - 4]
        crc = struct.unpack_from(">I", raw, end - 4)[0]
        need(zlib.crc32(tag + payload) == crc, "ORIGINAL_PNG_CHUNK_CRC")
        if tag in (b"IHDR", b"IDAT", b"IEND"):
            chunks.append((tag, payload, raw[offset:end]))
        else:
            need(tag[0] & 32 and tag not in (b"acTL", b"fcTL", b"fdAT", b"tRNS"),
                 "UNSUPPORTED_PNG_DISPLAY_SEMANTICS")
        offset = end
    types = [tag for tag, _, _ in chunks]
    need(offset == len(raw) and types[:1] == [b"IHDR"] and types[-1:] == [b"IEND"]
         and len(types) >= 3 and set(types[1:-1]) == {b"IDAT"}
         and chunks[0][1] == struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0)
         and not chunks[-1][1], "EXACT_STATIC_RGB_256_PNG")
    return signature + b"".join(chunk for _, _, chunk in chunks)


def prepare(args):
    plan, plan_sha = frozen(args.plan)
    terminal_raw = read(args.terminal, 64 << 10, args.terminal_sha)
    terminal = json.loads(terminal_raw)
    worker, pairs = final_pairs(args.generation, plan, plan_sha, terminal)
    need(len(pairs) == 6 and worker["passed"] is True, "EXACT_COMPLETE_SIX_PAIRS_FOR_REVIEW")

    staged = []
    for item, left, right in pairs:
        for row in (left, right):
            expected = f"{row['arm']}-{row['pair']:02d}.private.png"
            need(row["image"] == expected, "EXACT_SEALED_PNG_NAME")
            raw = read(args.generation / expected, 2 << 20, row["image_SHA256"])
            need(len(raw) == row["image_bytes"], "EXACT_PNG_EXTENT")
            staged.append((item, row, clean_png(raw)))
    secrets.SystemRandom().shuffle(staged)
    identifiers = [secrets.token_hex(12) for _ in staged]
    need(len(set(identifiers)) == len(staged), "DISTINCT_OPAQUE_REVIEW_IDS")
    args.out.mkdir(mode=0o700)
    viewer = args.out / "reviewer"
    viewer.mkdir(mode=0o700)
    slots, mapping = [], []
    for ident, (item, row, png) in zip(identifiers, staged, strict=True):
        put(viewer / (ident + ".png"), png)
        slots.append(dict(id=ident, image=ident + ".png", prompt=item["prompt"], SHA256=digest(png)))
        mapping.append(dict(id=ident, arm=row["arm"], pair=row["pair"], family=item["family"],
                            original_image_SHA256=row["image_SHA256"]))
    queue = wire(dict(criteria=CRITERIA, slots=slots))
    put(viewer / "queue.json", queue)
    template = wire(dict(queue_SHA256=digest(queue), reviewer_id="",
                         ratings=[dict(id=ident, **dict.fromkeys(CRITERIA)) for ident in identifiers]))
    put(viewer / "ratings-template.json", template)
    key = wire(dict(queue_SHA256=digest(queue), terminal_SHA256=args.terminal_sha,
                    original_worker_SHA256=terminal["worker_result_SHA256"], mapping=mapping))
    put(args.out / "unblind.private.json", key)
    receipt = dict(queue_SHA256=digest(queue), unblind_commitment_SHA256=digest(key),
                   terminal_SHA256=args.terminal_sha, images=len(slots), ratings_sealed=0)
    put(args.out / "receipt.json", wire(receipt))
    return receipt


def score(args):
    receipt = json.loads(read(args.bundle / "receipt.json", 8 << 10, args.receipt_sha))
    ratings = json.loads(read(args.ratings, 32 << 10, args.ratings_sha))
    queue_raw = read(args.bundle / "reviewer/queue.json", 32 << 10, receipt["queue_SHA256"])
    queue = json.loads(queue_raw)
    need(ratings["queue_SHA256"] == receipt["queue_SHA256"] and isinstance(ratings["reviewer_id"], str)
         and bool(ratings["reviewer_id"].strip()), "INDEPENDENT_REVIEWER_AND_QUEUE_IDENTITY")
    expected = [s["id"] for s in queue["slots"]]
    rows = ratings["ratings"]
    need(len(rows) == len(expected) == len(set(expected)) == 12 and {r["id"] for r in rows} == set(expected)
         and all(set(r) == {"id", *CRITERIA} and all(type(r[k]) is bool for k in CRITERIA) for r in rows),
         "TWELVE_EXPLICIT_BLIND_RATINGS")
    # No key is read until the exact independent rating bytes and queue are verified.
    key_raw = read(args.bundle / "unblind.private.json", 32 << 10, receipt["unblind_commitment_SHA256"])
    key = json.loads(key_raw)
    need(key["queue_SHA256"] == receipt["queue_SHA256"] and [r["id"] for r in key["mapping"]] == expected,
         "EXACT_FROZEN_UNBLINDING_KEY")
    for slot in queue["slots"]:
        need(slot["image"] == slot["id"] + ".png", "OPAQUE_REVIEW_IMAGE_NAME")
        read(args.bundle / "reviewer" / slot["image"], 2 << 20, slot["SHA256"])
    by_id = {r["id"]: r for r in rows}
    pairs = {}
    for slot in key["mapping"]:
        pair = pairs.setdefault((slot["family"], slot["pair"]), {})
        need(slot["arm"] in ARMS and slot["arm"] not in pair, "EXACT_ONCE_PER_ARM")
        pair[slot["arm"]] = all(by_id[slot["id"]][k] for k in CRITERIA)
    need(len(pairs) == 6 and all(set(pair) == set(ARMS) for pair in pairs.values()), "SIX_COMPLETE_REVIEWS")
    families = {}
    for (family, _), pair in pairs.items():
        families.setdefault(family, []).append(int(pair["cp25"]) - int(pair["base"]))
    need(len(families) == 3 and all(len(values) == 2 for values in families.values()),
         "THREE_INDEPENDENT_PRODUCT_FAMILIES")
    result = dict(ratings_SHA256=args.ratings_sha, images=12, pairs=6, independent_families=3,
                  combined_success={arm: sum(p[arm] for p in pairs.values()) for arm in ARMS},
                  criterion_success={arm: {k: sum(by_id[x["id"]][k] for x in key["mapping"] if x["arm"] == arm)
                                           for k in CRITERIA} for arm in ARMS},
                  family_mean_difference={family: statistics.fmean(values) for family, values in families.items()},
                  equal_family_difference=statistics.fmean(statistics.fmean(values) for values in families.values()),
                  formal_FID=False, formal_domain_increment=0)
    put(args.output, wire(result))
    return result


def main():
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="stage", required=True)
    pack = subparsers.add_parser("prepare")
    for arg in ("plan", "terminal", "generation", "out"):
        pack.add_argument("--" + arg, type=Path, required=True)
    pack.add_argument("--terminal-sha", required=True)
    reveal = subparsers.add_parser("score")
    for arg in ("bundle", "ratings", "output"):
        reveal.add_argument("--" + arg, type=Path, required=True)
    for arg in ("receipt-sha", "ratings-sha"):
        reveal.add_argument("--" + arg, required=True)
    args = root.parse_args()
    print(json.dumps(prepare(args) if args.stage == "prepare" else score(args), sort_keys=True))


if __name__ == "__main__":
    main()
