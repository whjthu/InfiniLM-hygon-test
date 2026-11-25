"""Build the Qwen3-VL street-scene SFT slice from pinned TextVQA and TextOCR train assets."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unicodedata


TEXTVQA_ASSETS_SHA = "a3246f1eec4a0d3fb1618c5c12b57d196004b20d1811e28665a8ce21a7e0fce0"
TEXTOCR_JOIN_SHA = "d323cb093fa55c9dcd19ff429f3747ecc597ce14a00c63ecbc8e2fd87ef17b10"
PROMPT = (
    "Transcribe the readable words in the image. For each word, return its text and "
    "bounding rectangle [left, top, right, bottom] in coordinates normalized to 0–1000."
)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("original dataset receipt changed")
    return json.loads(raw)


def normalize(answer):
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())


def consensus(answers):
    normalized = [normalize(answer) for answer in answers]
    votes = Counter(value for value in normalized if value)
    best = votes.most_common(2)
    if not best or best[0][1] < 3 or (len(best) > 1 and best[0][1] == best[1][1]):
        return None
    return answers[normalized.index(best[0][0])]


def item(image, question, answer):
    if "<image>" in question or "<image>" in answer or not question.strip() or not answer.strip():
        raise ValueError("unexpected role delimiter or empty example")
    return {"images": [image], "messages": [
        {"role": "user", "content": "<image>\n" + question},
        {"role": "assistant", "content": answer},
    ]}


def boxes(words, width, height):
    result = []
    rejected = 0
    for row in words:
        xs, ys = row["points"][::2], row["points"][1::2]
        quantized = [round(min(xs)*1000/width), round(min(ys)*1000/height),
                     round(max(xs)*1000/width), round(max(ys)*1000/height)]
        left, top, right, bottom = quantized
        if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
            rejected += 1
            continue
        result.append({"text": row["word"], "box": quantized})
    result.sort(key=lambda entry: (entry["box"][1], entry["box"][0], entry["box"][3], entry["box"][2], entry["text"]))
    return result, rejected


def build(media_path, ocr_path):
    media = pinned(media_path, TEXTVQA_ASSETS_SHA)
    ocr = pinned(ocr_path, TEXTOCR_JOIN_SHA)
    if (media["split"] != "train" or media["blind_labels_read"] or ocr["blind_labels_read"]
            or ocr["input_train_assets_SHA256"] != TEXTVQA_ASSETS_SHA
            or set(media["images"]) != set(ocr["records"])):
        raise ValueError("training-only image binding changed")
    rows, lineage = [], []
    excluded_questions = []
    target_polygons = rejected_boxes = 0
    for name, image in sorted(media["images"].items()):
        basename = image["relative_file"]
        path = media_path.parent / basename
        if (path.name != basename or path.resolve(strict=True) != path.absolute()
                or digest(path.read_bytes()) != image["SHA256"]):
            raise ValueError("frozen original image bytes changed")
        for qa in sorted(image["train_questions"], key=lambda row: row["question_id"]):
            answer = consensus(qa["answers"])
            if answer is None:
                excluded_questions.append(qa["question_id"])
                continue
            rows.append(item(basename, qa["question"], answer))
            lineage.append({"task": "TextVQA", "image_id": name, "question_id": qa["question_id"]})
        words, rejected = boxes(ocr["records"][name]["qualified"], image["width"], image["height"])
        rejected_boxes += rejected
        if words:
            rows.append(item(basename, PROMPT, json.dumps(words, ensure_ascii=False, separators=(",", ":"))))
            lineage.append({"task": "TextOCR", "image_id": name, "words": len(words)})
            target_polygons += len(words)
    expected_qa = sum(len(r["train_questions"]) for r in media["images"].values())
    if sum(row["task"]=="TextVQA" for row in lineage) + len(excluded_questions) != expected_qa:
        raise ValueError("quality denominator changed")
    proof = {"images": len(media["images"]), "original_VQA_questions": expected_qa,
             "VQA_rows": sum(row["task"]=="TextVQA" for row in lineage), "OCR_rows": sum(row["task"]=="TextOCR" for row in lineage),
             "OCR_words": target_polygons, "OCR_geometry_rejected_before_builder": ocr["rejected_polygons"],
             "OCR_boxes_rejected_after_quantization": rejected_boxes, "excluded_ambiguous_questions": excluded_questions,
             "lineage": lineage, "TextVQA_assets_SHA256": TEXTVQA_ASSETS_SHA, "TextOCR_join_SHA256": TEXTOCR_JOIN_SHA,
             "native_processor_rows": 0, "optimizer_updates": 0, "training_ready": False}
    return rows, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("ocr", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    directory = args.media.parent.resolve(strict=True)
    if args.train.parent.resolve(strict=True) != directory or args.receipt.parent.resolve(strict=True) != directory:
        raise ValueError("renderer expects media and outputs in the same private corpus directory")
    if any(path.exists() or path.is_symlink() for path in (args.train, args.receipt)):
        raise ValueError("output already exists")
    rows, proof = build(args.media, args.ocr)
    train = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in rows).encode()
    proof.update(train_SHA256=digest(train), train_bytes=len(train))
    receipt = (json.dumps(proof, ensure_ascii=False, sort_keys=True)+"\n").encode()
    with args.train.open("xb") as out:
        out.write(train)
    with args.receipt.open("xb") as out:
        out.write(receipt)
    print(json.dumps({**{key:proof[key] for key in ("images","VQA_rows","OCR_rows","OCR_words","train_SHA256","training_ready")},
                      "ambiguous_questions": len(proof["excluded_ambiguous_questions"]),"receipt_SHA256":digest(receipt)}))


if __name__ == "__main__":
    main()
