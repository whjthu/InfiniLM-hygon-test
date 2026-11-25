"""Independent verification of signed depth from frozen train PNGs and public ComCat."""
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
from hashlib import sha256
import io,json
from pathlib import Path

local=Path.cwd().resolve()/"local-only"
root=local/"usgs-r26-train12-depth-change-candidate"
def digest(raw):
    return sha256(raw).hexdigest()
def exact(path,expected,limit=300000):
    assert path.resolve(strict=True)==path and 0<path.stat().st_size<=limit
    data=path.read_bytes()
    assert len(data)<=limit and digest(data)==expected
    return data
def aliases(feature):
    p=feature["properties"]
    return {a.lower() for a in p["ids"].split(",") if a}|{feature["id"].lower(),(p["net"]+p["code"]).lower()}
def round_depth(feature):
    return Decimal(feature["geometry"]["coordinates"][2]).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
visual_path=local/"usgs-r26-frozen-independent-12-png-visual-depth-endpoints.json"
visual_pin="699069081135e5e4182e909031d396a41d6f81aa2d0859ee625c4cb7ccc5a9d2"
v=json.loads(exact(visual_path,visual_pin))
assert not v["opened_train_answer_file"] and not v["opened_original_public_geojson_values"] and v["dev_blind_file_body_reads"]==0
assert len(v["rows"])==12
visual={r["path"]:r for r in v["rows"]}
assert len(visual)==12
candidate_pin="da46986c6799414edff4a57de0112bc72e1fc64e21334141b83c14d26b2b93e1"
index=json.loads(exact(root/"CANDIDATE_INDEX.json",candidate_pin))
assert v["candidate_index_SHA256"]==candidate_pin
assert index["schema"]=="USGS_R26_PUBLIC_JULY_2024_TRAIN_ONLY_DEPTH_CHANGE_CANDIDATE_V1"
assert set(index["files"])==set(visual)|{"train-depth-change.jsonl"}
train_pin=index["files"]["train-depth-change.jsonl"]["SHA256"]
rows=[json.loads(line) for line in exact(root/"train-depth-change.jsonl",train_pin).splitlines()]
assert len(rows)==len(index["chart_identity"])==12
source_pin="cb8d47d716603e935da6f36d28a5286d2017d9ed908d6332e164fd5f862a7ca2"
features=json.loads(exact(local/"usgs-r26-public-comcat-2024-07-original.geojson",source_pin),parse_float=Decimal)["features"]
assert len(features)==216
source_order=sorted(range(len(features)),key=lambda j:(features[j]["properties"]["time"],features[j]["id"]))
all_aliases=set()
for feature in features:
    strict_ids=aliases(feature)
    assert not (all_aliases & strict_ids)
    all_aliases.update(strict_ids)
old_pin="4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc"
old=json.loads(exact(local/"training-USGS-multievent-render144-v2-original/dataset/manifest-private.json",old_pin))
old_by_split={split:[chart for chart in old["charts"] if chart["split"]==split] for split in ("train","dev","blind")}
assert {k:len(v) for k,v in old_by_split.items()}=={"train":12,"dev":6,"blind":6}
assert len({e.lower() for chart in old["charts"] for e in chart["event_ids"]})==144
source_windows=[source_order[i:i+6] for i in range(0,len(source_order),6)]
target_signs=Counter()
expected_orders=[]
for ordinal,window in enumerate(source_windows,1):
    delta=round_depth(features[window[-1]])-round_depth(features[window[0]])
    sign="positive" if delta>0 else "negative" if delta<0 else "zero"
    if sign!="zero" and target_signs[sign]<6: expected_orders.append(ordinal)
    target_signs[sign]+=1
assert target_signs=={"positive":11,"negative":18,"zero":7}
assert [x["chart_ordinal"] for x in index["chart_identity"]]==expected_orders
ids,charts,images,used_aliases=set(),set(),set(),set()
signs=Counter()
for row,entry in zip(rows,index["chart_identity"],strict=True):
    path=entry["image"]
    observed=visual[path]
    assert path.startswith("train/media/") and row["images"]==[path] and observed["panel"]=="Reported depth" and observed["unit"]=="km"
    assert row["id"] not in ids and row["chart_id"]==row["group"]==entry["chart_id"] and row["chart_id"] not in charts
    assert row["kind"]=="depth_change"
    ids.add(row["id"]);charts.add(row["chart_id"])
    imgsha=index["files"][path]["SHA256"]
    assert imgsha==observed["image_SHA256"] and imgsha not in images
    raw=exact(root/path,imgsha,1048576)
    assert len(raw)==index["files"][path]["bytes"]
    assert raw.startswith(b"\x89PNG\r\n\x1a\n") and raw[12:16]==b"IHDR" and int.from_bytes(raw[16:20],"big")==int.from_bytes(raw[20:24],"big")==512
    images.add(imgsha)
    slots=[n-1 for n in entry["event_source_slots"]]
    assert slots==source_windows[entry["chart_ordinal"]-1]
    raw_events=[features[n] for n in slots]
    assert entry["event_primary_ids"]==[e["id"] for e in raw_events]
    for e in raw_events:
        e_aliases=aliases(e)
        assert not (used_aliases & e_aliases)
        used_aliases.update(e_aliases)
    first,last=Decimal(observed["E01"]),Decimal(observed["E06"])
    assert first==round_depth(raw_events[0]) and last==round_depth(raw_events[-1])
    diff=last-first
    assert diff!=0
    sign="positive" if diff>0 else "negative"
    assert entry["sign"]==sign
    assert row["messages"]==[
       {"role":"user","content":"<image>\nCompute DISPLAYED E06 depth minus DISPLAYED E01 depth, in km. "
        "Use exactly two decimal places and a minus sign if negative; retain zero. "
        "Answer FINAL: followed by the answer."},
       {"role":"assistant","content":f"FINAL: {diff:.2f}"}]
    signs[sign]+=1
assert signs=={"positive":6,"negative":6}
overlap={}
for split,prior in old_by_split.items():
    overlap[split]=dict(event_or_alias=len(used_aliases&{e.lower() for c in prior for e in c["event_ids"]}),
       chart_or_group=len(charts&{ident for c in prior for ident in (c["id"],c["group"])}),
       PNG=len(images&{old["outputs"][c["image"]]["SHA256"] for c in prior}))
assert all(v==0 for metrics in overlap.values() for v in metrics.values())
record=dict(state="INDEPENDENT_PUBLIC_SOURCE_TO_PNG_TO_TRAIN_LABEL_PASS",query_response_SHA256=source_pin,
    prior_identity_manifest_SHA256=old_pin,candidate_index_SHA256=candidate_pin,train_jsonl_SHA256=train_pin,
    frozen_image_transcription_SHA256=visual_pin,public_rows_checked=len(features),selected_public_primary_events=72,
    verified_new_charts=len(charts),verified_new_rows=len(ids),label_matches=12,endpoint_source_matches=24,
    pair_capacity=6,counts_by_sign=dict(signs),prior_identity_overlap_by_split=overlap,
    old_reverse_alias_table_available=False,train_only=True,opened_dev_blind_body_files=0,training_ready=False,
    actual_optimizer_updates=0,domain_models_delivered=0)
path=local/"usgs-r26-independent-public-source-png-train-verification.json"
output=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
with path.open("xb") as f:f.write(output)
print(json.dumps(dict(file=path.name,SHA256=digest(output),bytes=len(output),**record),sort_keys=True))
