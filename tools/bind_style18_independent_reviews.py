"""Freeze a private two-reviewer Style18 allocation and score only sealed votes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import hmac
import json
from pathlib import Path
import re
import secrets


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local-only"
SOURCE_PIN = "f71913a6a78e1e6f109baf99d23cea2e1ada9ed0b40a2684c30c629303ecef7a"
ORIGINAL_PIN = "a1e69159b7c39619eeecb64ebd5ac7fa66bb94b4a37a70bcec5bf2ed6e20c367"
HEX = re.compile("[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def private_json(path, expected):
    require(path.resolve(strict=True).is_relative_to(LOCAL) and not path.is_symlink(),
            "Private review metadata must stay under local-only")
    with path.open("rb") as stream:
        raw = stream.read(131073)
    require(len(raw) <= 131072 and digest(raw) == expected, "Exact private review evidence required")
    return json.loads(raw)


def write_private(path, obj):
    require(path.resolve().is_relative_to(LOCAL), "Private review metadata must stay under local-only")
    raw = canonical(obj)
    with path.open("xb") as stream:
        stream.write(raw)
    return digest(raw)


def pair_index(source):
    pairs = source["seeds"]
    prompts = {row["slot"] for row in source["prompts"] if row["source_attested"]}
    view = {row["pair_slot"]: row["prompt_slot"] for row in pairs}
    require(prompts == {f"{kind}{index}" for kind in "DC" for index in range(1, 5)}
            and len(pairs) == len(view) == len({row["seed"] for row in pairs}) == 16,
            "The independently reviewed eight-prompt, sixteen-pair plan differs")
    require(view == {f"{prompt}-{repeat}": prompt for prompt in prompts for repeat in (1, 2)},
            "An expected review pair is missing or not at its original prompt")
    return view


def allocate(source, original, custody, secret):
    pairs = pair_index(source)
    rubric = original["independent_human_rubric"]
    require(rubric["required_independent_raters"] == 2 and rubric["primary_vote_values"] == ["A", "Tie", "B"],
            "The original two-person, three-choice rubric differs")
    rubric_pin = digest(canonical(rubric))
    raters = custody["reviewers"]
    require(custody["status"] == "ROOT_VERIFIED_TWO_INDEPENDENT_REVIEWERS"
            and custody["source_packet_SHA256"] == SOURCE_PIN and custody["rubric_SHA256"] == rubric_pin
            and len(raters) == len({row["owner_id"] for row in raters}) == 2
            and all(re.fullmatch("[a-zA-Z0-9._-]{1,64}", row["owner_id"])
                    and HEX.fullmatch(row["attestation_SHA256"]) for row in raters)
            and len({row["attestation_SHA256"] for row in raters}) == 2,
            "Root must supply attestations for two distinct independent reviewers")
    require(len(secret) == 32, "Use a private 32-byte allocation seed")
    mappings = {}
    for rater in raters:
        owner = rater["owner_id"]
        order = {}
        for pair, prompt in pairs.items():
            keyed = hmac.digest(secret, canonical([owner, prompt, SOURCE_PIN]), "sha256")
            first = keyed[0] % 2
            side = first if pair.endswith("-1") else 1 - first
            order[pair] = ("before", "after")[side]
        mappings[owner] = order
    return dict(state="PRIVATE_STYLE18_REVIEW_ALLOCATION_BEFORE_GENERATION", source_packet_SHA256=SOURCE_PIN,
                rubric_SHA256=rubric_pin, pair_to_prompt=pairs, A_arm_by_reviewer=mappings,
                reviewers=raters, custody_SHA256=digest(canonical(custody)))


def summarize(allocation, allocation_pin, sealed_reviews, asset_manifest):
    require(digest(canonical(allocation)) == allocation_pin, "The pre-generation allocation commitment differs")
    expected = allocation["pair_to_prompt"]
    reviewers = allocation["A_arm_by_reviewer"]
    require(set(sealed_reviews) == set(reviewers) == set(asset_manifest["reviewer_image_manifests"]),
            "Both independent review receipts and their image manifests are required")
    require(asset_manifest["allocation_SHA256"] == allocation_pin,
            "The root image manifests refer to a different allocation")
    votes = defaultdict(list)
    for owner, review in sealed_reviews.items():
        rows = review["judgments"]
        require(review["state"] == "BLINDED_REVIEW_SEALED_BEFORE_UNBLIND"
                and review["owner_id"] == owner and review["rubric_SHA256"] == allocation["rubric_SHA256"]
                and review["allocation_SHA256"] == allocation_pin
                and review["image_manifest_SHA256"] == asset_manifest["reviewer_image_manifests"][owner]
                and HEX.fullmatch(review["image_manifest_SHA256"])
                and len(rows) == len({row["pair_slot"] for row in rows}) == len(expected)
                and {row["pair_slot"] for row in rows} == set(expected),
                "A review receipt or its complete denominator differs")
        for row in rows:
            pair, value = row["pair_slot"], row["vote"]
            require(value in {"A", "Tie", "B", None} and
                    (value is not None or isinstance(row.get("missing_reason"), str) and row["missing_reason"].strip()),
                    "An omitted comparison must have the original failure reason")
            arm = reviewers[owner][pair]
            if value == "B":
                arm = "after" if arm == "before" else "before"
            votes[pair].append("missing" if value is None else "tie" if value == "Tie"
                               else "candidate" if arm == "after" else "baseline")
    strata = {}
    for label, prefix in (("domain", "D"), ("control", "C")):
        grouped = defaultdict(list)
        for pair, prompt in expected.items():
            if prompt.startswith(prefix):
                grouped[prompt].append(votes[pair])
        rows = [vote for pairs in grouped.values() for vote in pairs]
        require(len(grouped) == 4 and len(rows) == 8 and all(len(pair) == 2 for pair in rows),
                "The per-stratum denominators differ")
        strata[label] = dict(independent_prompts=4, paired_seed_repetitions=8,
                            missing_pairs=sum("missing" in pair for pair in rows),
                            disagreements=sum(a != b and "missing" not in (a, b) for a, b in rows),
                            votes=dict(Counter(vote for pair in rows for vote in pair)),
                            per_prompt={prompt: dict(Counter(vote for pair in pairs for vote in pair))
                                        for prompt, pairs in sorted(grouped.items())})
    return dict(state="STYLE18_TWO_SEALED_REVIEWS_UNBLINDED", strata=strata,
                aggregate_complete=all(row["missing_pairs"] == 0 for row in strata.values()),
                formal_delivery_increment=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("original", type=Path)
    prepare.add_argument("custody", type=Path)
    prepare.add_argument("custody_sha")
    prepare.add_argument("allocation", type=Path)
    finish = commands.add_parser("summarize")
    finish.add_argument("allocation", type=Path)
    finish.add_argument("allocation_sha")
    finish.add_argument("images", type=Path)
    finish.add_argument("images_sha")
    finish.add_argument("--review", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    args = parser.parse_args()
    if args.command == "prepare":
        allocation = allocate(private_json(args.source, SOURCE_PIN), private_json(args.original, ORIGINAL_PIN),
                              private_json(args.custody, args.custody_sha), secrets.token_bytes(32))
        result = dict(allocation_commitment_SHA256=write_private(args.allocation, allocation),
                      reviewers=2, pairs=16, new_generations=0)
    else:
        reviews = [private_json(Path(path), pin) for path, pin in args.review]
        owners = {review["owner_id"]: review for review in reviews}
        require(len(reviews) == len(owners) == 2, "Provide exactly two unique sealed review receipts")
        result = summarize(private_json(args.allocation, args.allocation_sha), args.allocation_sha,
                           owners, private_json(args.images, args.images_sha))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
