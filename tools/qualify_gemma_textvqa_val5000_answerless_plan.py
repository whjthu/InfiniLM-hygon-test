"""Bind the actual 5,000 official question texts to the independently sealed image roster."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


QUESTION_SHA = "1d247d3e670aebfd02508383239c48d3a97514e2f0dc37712ae7946f18c4387c"
MEDIA_SHA = "59a2d28e626385f97885357606f97f798f963602296297218f952cbee768898f"
CONTRACT_SHA = "85268183847c9af8a5a9d444eaf4fe19cd8df19149ffd77d046c3298132752c7"
ROOT = Path(__file__).resolve().parents[1]


def read(path, pin, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit or sha256(raw).hexdigest() != pin:
        raise ValueError(f"Frozen input differs: {path.name}")
    return raw


def save(path, blob):
    with path.open("xb") as stream:
        stream.write(blob)
    return sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    local = ROOT / "local-only"
    output = local / args.out
    if not output.resolve().parent == local or output.exists():
        raise ValueError("Use one new private result directory")
    contract_path = local / "gemma-textvqa-val5000-adapter-preflight-v1/contract.py"
    source = read(contract_path, CONTRACT_SHA, 32 << 10)
    spec = importlib.util.spec_from_file_location("frozen_gemma_contract", contract_path)
    contract = importlib.util.module_from_spec(spec)
    exec(compile(source, str(contract_path), "exec"), vars(contract))
    published = local / "training-qwen-generic-public-holdout-v1"
    identity = contract.identities(read(published / "TEXTVQA_VAL_051_IDENTITIES.json", contract.IDENTITIES, 1 << 20))
    questions = contract.document(read(published / "TEXTVQA_VAL_051_QUESTIONS_WITHOUT_ANSWERS.json",
                                       QUESTION_SHA, 1 << 20))
    original = contract.document(read(local / "gemma-textvqa-val5000-complete-reread-v1/retention-media.private.json",
                                      MEDIA_SHA, 2 << 20))
    if (original["all_local_asset_bytes_verified"] is not True or original["identities_SHA256"] != contract.IDENTITIES
            or original["annotation_SHA256"] != contract.ANN or any(row["readable"] is not True for row in original["rows"])):
        raise ValueError("Original publisher media readback differs")
    # This exact on-target comparison re-decoded the 64 original training inputs against the local validation seal.
    overlap_path = ROOT / "results/bw1000/2026-11-28/training-gemma-textvqa-val5000-target-native-MPO-and-cross-exact-RGB.json"
    overlap = contract.document(read(overlap_path, "a4ee4b0a331afbc563fc2696a9b5d2667a093304a35caadec7a870031dcd23f5", 32 << 10))
    cross = overlap["cross_exact_RGB"]
    if (cross["exact_train_media_reread"], cross["exact_validation_pixel_identities"],
            cross["overlapping_exact_RGB_distinct"]) != (64, 3166, 0):
        raise ValueError("The original image sets lack cross-split content qualification")
    media = dict(schema="TEXTVQA_V051_VAL_PRESENT_MEDIA_FOR_INFERENCE_V1",
                 annotations_SHA256=contract.ANN, identity_SHA256=contract.IDENTITIES, original_64_content_overlap=0,
                 rows=[dict(image_id=row["image_id"], relative_path=row["image_id"] + ".jpg",
                            bytes=row["bytes"], SHA256=row["SHA256"]) for row in original["rows"]])
    plan = contract.input_plan(identity, questions, media)
    result = dict(schema="GEMMA_TEXTVQA_VAL5000_ACTUAL_ANSWERLESS_PLAN_V1", ordered_questions=len(plan),
                  distinct_images=len(media["rows"]), native_contract_SHA256=CONTRACT_SHA,
                  identity_SHA256=contract.IDENTITIES, questions_SHA256=QUESTION_SHA,
                  sealed_local_media_SHA256=MEDIA_SHA, plan_SHA256=sha256(contract.packed(plan)).hexdigest(),
                  answer_rows_emitted=0, target_media_completed=False, target_evaluation_launched=False)
    output.mkdir()
    result["native_media_contract_SHA256"] = save(output / "media-for-inference.private.json", contract.packed(media))
    save(output / "result.json", contract.packed(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
