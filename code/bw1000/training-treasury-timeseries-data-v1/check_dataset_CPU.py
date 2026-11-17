"""Audit source arithmetic, grouped splits and answer-free holdout inputs without model use."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
from build_dataset import FIELDS, SPLITS, TOPICS, SNAPSHOT_SHA256, canonical, questions, select_rows, sha


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.dataset/"manifest.json").read_bytes())
    selected=select_rows(args.source.read_bytes())
    source=json.loads((args.dataset/"selected-source-rows.json").read_bytes())
    assert source["source_snapshot_SHA256"]==SNAPSHOT_SHA256
    assert source["data"]==[row for rows in selected.values() for row in rows] and len(source["data"])==120
    for name,pin in manifest["files_SHA256"].items():
        assert sha((args.dataset/name).read_bytes())==pin
    dates=set()
    windows=set()
    media=set()
    counts={}
    for split in SPLITS:
        records=lines(args.dataset/split/"inputs.jsonl")
        labels=lines(args.dataset/split/"labels.jsonl")
        groups=lines(args.dataset/split/"windows.jsonl")
        assert len(groups)==8 and len(records)==len(labels)==144
        answer_by_id={row["id"]:row for row in labels}
        assert len(answer_by_id)==144
        ground={}
        for group in groups:
            rows=group["source_rows"]
            assert len(rows)==5 and group["dates"]==[r["record_date"] for r in rows]
            assert group["id"]==sha(canonical(rows)) and group["id"] not in windows
            windows.add(group["id"])
            assert not set(group["dates"])&dates
            dates.update(group["dates"])
            for topic,field,question,answer in questions(rows):
                ground[(group["id"],topic,field)]=(question,answer)
        topics=Counter()
        modalities=Counter()
        by_window=Counter()
        for row in records:
            target=answer_by_id[row["id"]]
            question,answer=ground[(row["window_id"],row["topic"],target["field"])]
            assert target["window_id"]==row["window_id"] and target["topic"]==row["topic"] and target["answer"]==answer
            assert row["messages"][0]["content"]=="<image>\n"+question+" Return only the answer, without currency symbols, grouping separators or explanation."
            if split=="train":
                assert row["messages"][1:]==[dict(role="assistant",content=answer)]
            else:
                assert len(row["messages"])==1 and row["messages"][0]["role"]=="user"
            assert row["images"]==[f"{split}/{row['window_id']}-{row['modality']}.png"]
            topics[row["topic"]]+=1
            modalities[row["modality"]]+=1
            by_window[row["window_id"]]+=1
        assert dict(topics)==dict.fromkeys(TOPICS,48) and dict(modalities)=={"table":72,"chart":72}
        assert set(by_window.values())=={18}
        for relative in {row["images"][0] for row in records}:
            raw=(args.dataset/relative).read_bytes()
            pin=sha(raw)
            assert pin not in media
            media.add(pin)
            with Image.open(io.BytesIO(raw)) as image:
                assert image.mode=="RGB" and image.size in ((1440,360),(1440,1020))
                image.verify()
        counts[split]=dict(dates=40,windows=8,images=16,examples=144,topics=dict(topics),
                           input_supervision_present=split=="train")
    assert len(dates)==120 and len(windows)==24 and len(media)==48
    try:
        select_rows(args.source.read_bytes()+b" ")
    except ValueError:
        pass
    else:
        raise AssertionError("Changed source admitted")
    result=dict(state="CPU_DATA_CONTRACT_PASS",manifest_SHA256=sha((args.dataset/"manifest.json").read_bytes()),
                source_snapshot_SHA256=SNAPSHOT_SHA256,splits=counts,raw_dates=120,windows=24,images=48,examples=432,
                cross_split_date_overlap=0,cross_split_window_overlap=0,cross_split_media_overlap=0,
                automated_gold_recomputation=True,manual_blind_label_inspection_or_model_tuning=False,
                selected_source_rows_bytes=(args.dataset/"selected-source-rows.json").stat().st_size,
                new_official_data_download_bytes=0,model_or_GPU_executed=False,training_ready=False)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2)
        stream.write("\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
