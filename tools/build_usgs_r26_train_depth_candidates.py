"""Project the one frozen ComCat response into a train-only signed-depth candidate."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/"local-only"
SOURCE = PRIVATE/"usgs-r26-public-comcat-2024-07-original.geojson"
SOURCE_SHA = "cb8d47d716603e935da6f36d28a5286d2017d9ed908d6332e164fd5f862a7ca2"
SOURCE_BYTES = 154907
OUTPUT = PRIVATE/"usgs-r26-train12-depth-change-candidate"
OLD = PRIVATE/"training-USGS-multievent-render144-v2-original/dataset/manifest-private.json"
OLD_PIN = "4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc"
FROZEN = ROOT/"code/bw1000/training-USGS-multievent-render144-v2"
RENDER_SHA = "fd4a6af93018e629e7c047ff0a56b095e84fc38ea758c6062f05b661b8615938"
CONTRACT_SHA = "17a2802a78a2fb3b45a9918660774d25718433a7fb477feaf4a31987faf3e68e"
OVERLAY_ROOT = PRIVATE/"workspace-matplotlib-overlay-completion-fsid-fix-v2"
OVERLAY_RESULT_SHA = "d4642ac075a3681e2b13611d5c43858f591f36919ccd921afbf431447f6b0c2a"
RENDER_PLAN = {"renderer_versions":{"Pillow":"13.0.0","matplotlib":"3.10.7","numpy":"2.3.5"},
               "font_SHA256":"3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22",
               "Software_metadata":"training-USGS-r26-2024-07-train-depth-candidate","PNG_max_bytes":1048576}
START = int(datetime(2024,7,1,tzinfo=timezone.utc).timestamp()*1000)
END = int(datetime(2024,8,1,tzinfo=timezone.utc).timestamp()*1000)
QUOTA = 6
LIMIT = 216
CAP = 2*1024**2
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}")
HEX = re.compile(r"[0-9a-f]{64}")
SPLITS = {"train":12,"dev":6,"blind":6}


class Rejected(ValueError):
    """Stable bounded rejection with no input values."""


def need(condition, code):
    if not condition:
        raise Rejected(code)


def digest(raw):
    return sha256(raw).hexdigest()


def document(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def pin(path, expected, extent=None, cap=CAP):
    before=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and
         0<before.st_size<=cap and (extent is None or before.st_size==extent),"BOUNDED_CANONICAL_INPUT")
    with path.open("rb") as file:
        raw=file.read(cap+1)
    after=path.lstat()
    def identity(info):
        return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
    need(len(raw)==before.st_size and identity(before)==identity(after) and digest(raw)==expected,
         "INPUT_IDENTITY_MISMATCH")
    return raw


def split_identities(manifest):
    need(manifest["Software"]=="training-USGS-multievent-render144-v2"
         and manifest["actual_split_QA"]=={key:count*6 for key,count in SPLITS.items()}
         and len(manifest["charts"])==24,"OLD_SPLIT_METADATA")
    result={}
    seen_charts,seen_media,seen_groups=set(),set(),{}
    counts=Counter()
    for chart in manifest["charts"]:
        split=chart["split"]
        events=chart["event_ids"]
        media=manifest["outputs"][chart["image"]]["SHA256"]
        need(split in SPLITS and chart["id"] not in seen_charts and media not in seen_media
             and chart["image"].startswith(split+"/media/") and len(events)==6 and chart["QA"]==6
             and seen_groups.setdefault(chart["group"],split)==split,"DISJOINT_OLD_CHART_GROUP_MEDIA")
        seen_charts.add(chart["id"])
        seen_media.add(media)
        counts[split]+=1
        for event in events:
            need(isinstance(event,str) and ID.fullmatch(event) and event.casefold() not in result,
                 "UNIQUE_OLD_EVENT_IDENTITIES")
            result[event.casefold()]=split
    need(dict(counts)==SPLITS and len(result)==144,"OLD_SPLIT_DENOMINATORS")
    return result


def identifiers(value):
    need(isinstance(value,str) and 0<len(value)<=8192,"BOUNDED_IDENTIFIER_FIELD")
    tokens=[item for item in value.split(",") if item]
    need(bool(tokens) and len(tokens)<=64 and all(ID.fullmatch(item) for item in tokens),"EXACT_IDENTIFIER_LIST")
    return tokens


def event(feature):
    q=feature["properties"]
    xyz=feature["geometry"]["coordinates"]
    eid=feature["id"]
    need(feature["type"]=="Feature" and feature["geometry"]["type"]=="Point" and len(xyz)==3
         and q["type"]=="earthquake","NON_EARTHQUAKE_OR_POINT")
    need(isinstance(eid,str) and ID.fullmatch(eid) and q.get("code") and q.get("net")
         and ID.fullmatch(q["net"]) and ID.fullmatch(q["code"]),"PRIMARY_EVENT_ID")
    need(type(q["time"]) is int and START<=q["time"]<END,"OUTSIDE_FROZEN_UTC_WINDOW")
    numeric={"magnitude":q["mag"],"longitude":xyz[0],"latitude":xyz[1],"depth_km":xyz[2]}
    need(all(type(number) in (Decimal,int) for number in numeric.values()),"MISSING_OR_NON_NUMERIC")
    need(all(Decimal(number).is_finite() for number in numeric.values())
         and Decimal("4.5")<=numeric["magnitude"]<=Decimal(12)
         and -180<=numeric["longitude"]<=180 and -90<=numeric["latitude"]<=90
         and -100<=numeric["depth_km"]<=1000,"SOURCE_NUMERIC_RANGE")
    aliases={item.casefold() for item in identifiers(q["ids"])}
    preferred=(q["net"]+q["code"]).casefold()
    aliases.update((eid.casefold(),preferred))
    need(isinstance(q["magType"],str) and ID.fullmatch(q["magType"]),"MAGNITUDE_TYPE")
    return {"id":eid,"aliases":sorted(aliases),"time_UTC_epoch_ms":q["time"],
            "preferred_net":q["net"],"sources":sorted(set(identifiers(q["sources"]))),
            "magType":q["magType"],"values":{key:str(number) for key,number in numeric.items()}}


def project(raw, old):
    src=json.loads(raw,parse_float=Decimal)
    features=src["features"]
    metadata=src["metadata"]
    need(src["type"]=="FeatureCollection" and isinstance(features,list) and 0<len(features)<=LIMIT
         and ("count" not in metadata or metadata["count"]==len(features))
         and ("limit" not in metadata or metadata["limit"] in (LIMIT,str(LIMIT)))
         and ("offset" not in metadata or metadata["offset"] in (1,"1"))
         and ("status" not in metadata or metadata["status"] in (200,"200")),
         "FROZEN_FEATURE_DENOMINATOR")
    valid=[]
    bad=Counter()
    last=-1
    for slot,item in enumerate(features,1):
        stamp=item.get("properties",{}).get("time") if isinstance(item,dict) and isinstance(item.get("properties"),dict) else None
        if type(stamp) is int:
            need(stamp>=last,"CATALOG_TIME_ASCENDING")
            last=stamp
        try:
            valid.append((slot,event(item)))
        except Rejected as error:
            bad[str(error)]+=1
        except (KeyError,TypeError,AttributeError):
            bad["MALFORMED_FEATURE"]+=1
    ordered=sorted(valid,key=lambda member:(member[1]["time_UTC_epoch_ms"],member[1]["id"],member[0]))
    parents=list(range(len(ordered)))
    def leader(i):
        while i!=parents[i]:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    owners={}
    for i,(_,item) in enumerate(ordered):
        for alias in item["aliases"]:
            j=owners.setdefault(alias,i)
            a,b=leader(i),leader(j)
            parents[max(a,b)]=min(a,b)
    groups={}
    for i,member in enumerate(ordered):
        groups.setdefault(leader(i),[]).append(member)
    eligible=[]
    duplicates=0
    overlaps=0
    by_split=Counter()
    duplicate_primary=0
    for members in groups.values():
        combined=sorted({alias for _,item in members for alias in item["aliases"]})
        splits={old[alias] for alias in combined if alias in old}
        if splits:
            overlaps+=len(members)
            by_split.update({split:len(members) for split in splits})
            continue
        slot,item=members[0]
        duplicates+=len(members)-1
        duplicate_primary+=len(members)-len({data["id"].casefold() for _,data in members})
        eligible.append({**item,"aliases":combined,"source_slot":slot})
    need(len(features)==len(eligible)+duplicates+overlaps+sum(bad.values()),"EVENT_DENOMINATOR_RECONCILIATION")
    need(len({alias for data in eligible for alias in data["aliases"]})==
         sum(len(data["aliases"]) for data in eligible),"DISJOINT_NEW_ALIAS_COMPONENTS")
    signs=Counter()
    charts=[]
    windows,remainder=divmod(len(eligible),6)
    for ordinal in range(windows):
        rows=eligible[ordinal*6:(ordinal+1)*6]
        def display(row):
            return Decimal(row["values"]["depth_km"]).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
        delta=display(rows[-1])-display(rows[0])
        sign="positive" if delta>0 else "negative" if delta<0 else "zero"
        signs[sign]+=1
        chart_id=digest(document([row["id"] for row in rows]))
        charts.append({"ordinal":ordinal+1,"id":chart_id,"group":chart_id,"split":"train","rows":rows,"sign":sign,
                       "expected_delta":format(Decimal(0) if delta==0 else delta,".2f")})
    selected={name:[c for c in charts if c["sign"]==name][:QUOTA] for name in ("positive","negative")}
    chosen=sorted(selected["positive"]+selected["negative"],key=lambda c:c["ordinal"])
    counts={"query_limit":LIMIT,"catalog_response_features":len(features),"query_limit_hit":len(features)==LIMIT,
            "qualified_before_overlap_and_dedup":len(valid),"rejections":dict(bad),"old_overlap_excluded":overlaps,
            "old_overlap_touched_by_split":{key:by_split[key] for key in SPLITS},"new_duplicate_excluded":duplicates,
            "duplicate_primary_within_eligible_components":duplicate_primary,"eligible_distinct_events":len(eligible),
            "six_event_windows":windows,"tail_events_not_windowed":remainder,
            "windows_by_sign":{key:signs[key] for key in ("positive","negative","zero")},
            "strict_target_each_sign":QUOTA,"selected_by_sign":{key:len(items) for key,items in selected.items()},
            "quota_shortfall_by_sign":{key:QUOTA-len(items) for key,items in selected.items()},
            "other_nonzero_windows":signs["positive"]+signs["negative"]-len(chosen),"zero_windows_excluded":signs["zero"],
            "selected_events":6*len(chosen)}
    return chosen, counts


def modules():
    import types
    contract=types.ModuleType("contracts")
    exec(compile(pin(FROZEN/"contracts.py",CONTRACT_SHA,2873),"frozen_usgs_contract","exec"),contract.__dict__)
    raw=pin(FROZEN/"render.py",RENDER_SHA)
    title=b"June 2026: reported earthquakes"
    need(raw.count(title)==1 and b"(.14,.27,.82,.23),'depth','Reported depth','km'" in raw,
         "EXACT_ORIGINAL_V2_RENDER_TEMPLATE")
    rendered=raw.replace(title,b"July 2024: reported earthquakes")
    module={}
    previous=sys.modules.get("contracts")
    sys.modules["contracts"]=contract
    try:
        exec(compile(rendered,"frozen_usgs_layout_with_2024_07_title","exec"),module)
    finally:
        if previous is None:
            del sys.modules["contracts"]
        else:
            sys.modules["contracts"]=previous
    return contract.questions,module,digest(rendered)


def png_valid(png):
    from PIL import Image
    with Image.open(io.BytesIO(png)) as image:
        need(image.format=="PNG" and image.size==(512,512) and
             image.info.get("Software")==RENDER_PLAN["Software_metadata"],"CANDIDATE_PNG_IDENTITY_AND_SHAPE")
        image.verify()


def write_once(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
    return {"SHA256":digest(raw),"bytes":len(raw)}


def materialize(selected, summary):
    need(not OUTPUT.exists(),"EXCLUSIVE_CANDIDATE_OUTPUT")
    outcome=json.loads(pin(OVERLAY_ROOT/"outcome.json",OVERLAY_RESULT_SHA,13615))
    need(outcome["passed"] is True and outcome["job"]["returncode"]==0 and outcome["job"]["timed_out"] is False,
         "ORIGINAL_RENDER_OVERLAY_QUALIFICATION")
    overlay=OVERLAY_ROOT/"overlay"
    need(overlay.resolve(strict=True)==overlay and overlay.is_dir(),"EXACT_LOCAL_MATPLOTLIB_OVERLAY")
    sys.path.insert(0,str(overlay))
    need(all(importlib.metadata.version(name)==version for name,version in RENDER_PLAN["renderer_versions"].items()),
         "EXACT_RENDERER_RUNTIME_VERSIONS")
    questions,renderer,derived_sha=modules()
    images={}
    rows=[]
    pins=[]
    with TemporaryDirectory(prefix="usgs-r26-mplconfig-",dir=PRIVATE) as cache:
        old_cache=os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"]=cache
        try:
            runtime=renderer["preflight"](RENDER_PLAN)
            for chart in selected:
                [task]=[q for q in questions(chart) if q["kind"]=="depth_change"]
                need(task["answer"]==chart["expected_delta"],"INDEPENDENT_AND_ORIGINAL_DISPLAY_ARITHMETIC")
                name=f"train/media/{chart['ordinal']:03d}.png"
                image=renderer["draw"](chart,RENDER_PLAN,runtime)
                png_valid(image)
                images[name]=image
                rows.append({"id":task["id"],"chart_id":chart["id"],"group":chart["group"],"kind":"depth_change",
                    "event_ids":task["event_ids"],"images":[name],
                    "messages":[{"role":"user","content":"<image>\n"+task["question"]},
                                {"role":"assistant","content":"FINAL: "+task["answer"]}]})
                pins.append({"chart_id":chart["id"],"chart_ordinal":chart["ordinal"],"sign":chart["sign"],
                             "event_primary_ids":[item["id"] for item in chart["rows"]],
                             "event_source_slots":[item["source_slot"] for item in chart["rows"]],
                             "event_all_identity_SHA256":digest(document([item["aliases"] for item in chart["rows"]])),
                             "image":name,"display_rows_SHA256":task["display_rows_SHA256"]})
        finally:
            if old_cache is None:
                os.environ.pop("MPLCONFIGDIR",None)
            else:
                os.environ["MPLCONFIGDIR"]=old_cache
    need(len(images)==len({digest(value) for value in images.values()})==12 and len(rows)==12,
         "DISTINCT_NEW_IMAGE_AND_ROW_DENOMINATOR")
    OUTPUT.mkdir(mode=0o700)
    (OUTPUT/"train/media").mkdir(parents=True)
    files={name:write_once(OUTPUT/name,data) for name,data in images.items()}
    text=(b"\n".join(document(row) for row in rows)+b"\n")
    files["train-depth-change.jsonl"]=write_once(OUTPUT/"train-depth-change.jsonl",text)
    bundle={"schema":"USGS_R26_PUBLIC_JULY_2024_TRAIN_ONLY_DEPTH_CHANGE_CANDIDATE_V1","input":summary,"files":files,
            "chart_identity":pins,"render":{"original_layout_SHA256":RENDER_SHA,"date_only_derived_SHA256":derived_sha,
                    "font_SHA256":RENDER_PLAN["font_SHA256"],"renderer_versions":RENDER_PLAN["renderer_versions"]},
            "semantic_visual_review":"PENDING_INDEPENDENT_READBACK","training_ready":False,"actual_optimizer_updates":0}
    receipt=write_once(OUTPUT/"CANDIDATE_INDEX.json",document(bundle)+b"\n")
    return {"candidate_index":receipt,"actual_candidate_training_rows":len(rows),"actual_candidate_images":len(images),
            "derived_date_only_renderer_SHA256":derived_sha,"semantic_visual_review":"PENDING"}


def run(source_sha, build):
    need(source_sha==SOURCE_SHA and HEX.fullmatch(source_sha),"EXPLICIT_FROZEN_SOURCE_SHA256")
    raw=pin(SOURCE,source_sha,SOURCE_BYTES)
    old=split_identities(json.loads(pin(OLD,OLD_PIN,14864)))
    chosen,counts=project(raw,old)
    complete=all(value==0 for value in counts["quota_shortfall_by_sign"].values())
    summary={"state":"COMPLETE_TRAIN_SELECTION" if complete else "TRAIN_SIGN_QUOTA_SHORTFALL",
             "source_SHA256":source_sha,"source_bytes":len(raw),"old_identity_manifest_SHA256":OLD_PIN,
             "historical_event_identity_matches_new_primary_alias_or_preferred_code":True,
             "historical_reverse_alias_table_available_in_manifest":False,
             "old_manifest_reads":1,"old_train_dev_blind_body_reads":0,"new_HTTP":0,
             "training_ready":False,"actual_optimizer_updates":0,"counts":counts,
             "actual_candidate_training_rows":0,"actual_candidate_images":0}
    if build and complete:
        summary.update(materialize(chosen,summary))
        summary["state"]="TRAIN_CANDIDATE_RENDERED_VISUAL_REVIEW_PENDING"
    return summary


def self_test():
    features=[]
    def fixture(eid,slot,depth,ids=None):
        return {"type":"Feature","id":eid,"geometry":{"type":"Point","coordinates":[0,0,depth]},
            "properties":{"mag":5,"time":START+slot,"type":"earthquake","net":"us","code":eid,
                          "ids":","+(",".join(ids or [eid]))+",","magType":"mb","sources":",us,"}}
    depths=[(1.004,1.015),(4.005,3.994),(1.004,1.003)]
    features.extend([fixture("conflicting",0,1,["old_id","bridge"]),fixture("bridged",1,1,["bridge","bridge2"])])
    for i,(first,last) in enumerate(depths):
        for j,depth in enumerate([first,2,3,4,5,last]):
            n=2+i*6+j
            features.append(fixture(f"event{n:02d}",n,depth))
    features.extend([fixture("duplicate",20,1,["event02"]),fixture("tail",21,1)])
    raw=document({"type":"FeatureCollection","metadata":{"limit":LIMIT,"offset":1,"status":200},"features":features})
    selected,counts=project(raw,{"old_id":"blind"})
    need(counts["old_overlap_excluded"]==2 and counts["new_duplicate_excluded"]==1
         and counts["tail_events_not_windowed"]==1 and counts["windows_by_sign"]=={"positive":1,"negative":1,"zero":1}
         and [x["expected_delta"] for x in selected]==["0.02","-0.02"]
         and counts["quota_shortfall_by_sign"]=={"positive":5,"negative":5},"FIXTURE_SELECTION")
    bad=features[:]
    bad[3],bad[4]=bad[4],bad[3]
    bad_doc=document({"type":"FeatureCollection","metadata":{"limit":LIMIT,"offset":1,"status":200},"features":bad})
    try:
        project(bad_doc,{})
    except Rejected as error:
        need(str(error)=="CATALOG_TIME_ASCENDING","FIXTURE_OUT_OF_ORDER_ERROR")
    else:
        raise Rejected("FIXTURE_OUT_OF_ORDER_ACCEPTED")
    questions,_,derived=modules()
    for candidate in selected:
        [answer]=[r["answer"] for r in questions(candidate) if r["kind"]=="depth_change"]
        need(answer==candidate["expected_delta"],"FIXTURE_NATIVE_GENERATOR_ALIGNMENT")
    return {"state":"SELF_TEST_PASS","fixture_windows":3,"fixture_alias_overlap":2,"fixture_duplicate_removed":1,
            "fixture_out_of_order_rejected":True,"date_only_derived_renderer_SHA256":derived,
            "actual_source_reads":0,"actual_renders":0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    try:
        result=self_test() if args.self_test else run(args.source_sha256,args.build)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,ImportError,ArithmeticError) as error:
        result={"state":"BLOCKED","reason":str(error) if isinstance(error,Rejected) else type(error).__name__,
                "actual_optimizer_updates":0,"training_ready":False}
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return int(result["state"]=="BLOCKED" or (args.build and result["state"]=="TRAIN_SIGN_QUOTA_SHORTFALL"))


if __name__=="__main__":
    sys.exit(main())
