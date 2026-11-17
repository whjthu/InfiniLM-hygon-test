"""Freeze 120 cached official Treasury dates into independent table/chart QA data."""
import argparse
from collections import Counter
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW

SNAPSHOT_SHA256 = "fba621dcbd93771067c0b8c8448ce6032fef5f5b44dc8ca4bc4198033d72c2fe"
API = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny"
PAGE = "https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/debt-to-the-penny"
TERMS = "https://fiscaldata.treasury.gov/about-us/#licensing"
FIELDS = ("record_date","debt_held_public_amt","intragov_hold_amt","tot_pub_debt_out_amt")
NAMES = ("Debt held by public","Intragov. holdings","Total public debt")
SPLITS = {"train":("2025-01-01","2025-03-31"),"dev":("2025-04-01","2025-06-30"),
          "blind":("2025-10-01","2025-12-31")}
TOPICS = ("lookup","signed_difference","maximum_date")
PER_SPLIT_ROWS = 40
ROWS_PER_WINDOW = 5
MONEY = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", re.ASCII)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def queries():
    return {name:API+"?"+urlencode({"fields":",".join(FIELDS),
        "filter":f"record_date:gte:{start},record_date:lte:{end}","sort":"record_date",
        "page[number]":"1","page[size]":"40","format":"json"}) for name,(start,end) in SPLITS.items()}


def select_rows(raw):
    if len(raw)>2*1024**2 or sha(raw)!=SNAPSHOT_SHA256:
        raise ValueError("Official cached snapshot differs")
    payload=json.loads(raw)
    meta=payload["meta"]
    if (meta["total-count"]!=len(payload["data"]) or meta["total-pages"]!=1
            or any(meta["dataTypes"][field]!="CURRENCY" or meta["dataFormats"][field]!="$10.20" for field in FIELDS[1:])):
        raise ValueError("Cached API completeness or currency metadata differs")
    selected={}
    used=set()
    previous=""
    candidates={split:[] for split in SPLITS}
    for row in payload["data"]:
        day=row["record_date"]
        date.fromisoformat(day)
        if day<=previous:
            raise ValueError("Source dates are repeated or unordered")
        previous=day
        for split,(start,end) in SPLITS.items():
            if start<=day<=end:
                candidates[split].append(row)
    for split,rows in candidates.items():
        if len(rows)<PER_SPLIT_ROWS:
            raise ValueError("Frozen source interval has insufficient dates")
        rows=rows[:PER_SPLIT_ROWS]
        for row in rows:
            if set(row)!=set(FIELDS) or row["record_date"] in used:
                raise ValueError("Source fields or cross-split date identity differs")
            used.add(row["record_date"])
            if any(not isinstance(row[field],str) or not MONEY.fullmatch(row[field]) for field in FIELDS[1:]):
                raise ValueError("Invalid nominal USD decimal amount")
            amounts=[Decimal(row[field]) for field in FIELDS[1:]]
            if any(value<=0 for value in amounts) or amounts[0]+amounts[1]!=amounts[2]:
                raise ValueError("Official source accounting identity differs")
        selected[split]=rows
    if len(used)!=120:
        raise ValueError("Expected exactly 120 disjoint official dates")
    return selected


def questions(rows):
    for offset,(field,label) in enumerate(zip(FIELDS[1:],NAMES)):
        values=[Decimal(row[field]) for row in rows]
        question=f"What was {label} on {rows[2]['record_date']}? Answer USD with exactly two decimals."
        yield TOPICS[0],field,question,f"{values[2]:.2f}"
        question=(f"What is the signed change in {label} from {rows[0]['record_date']} to {rows[-1]['record_date']} "
                  "(last minus first)? Answer USD with exactly two decimals.")
        yield TOPICS[1],field,question,f"{values[-1]-values[0]:.2f}"
        maximum=max(values)
        answer=next(row["record_date"] for row,value in zip(rows,values) if value==maximum)
        question=f"On which date was {label} highest? If tied, choose the earliest date. Answer YYYY-MM-DD."
        yield TOPICS[2],field,question,answer


def render_table(rows,font):
    image=Image.new("RGB",(1440,360),"white")
    draw=ImageDraw.Draw(image)
    draw.text((22,12),"U.S. Treasury Debt to the Penny | nominal USD | locally rendered",font=font,fill="black")
    x=(20,220,620,1020)
    for index,row in enumerate((dict(zip(FIELDS,("Record date",*NAMES))),*rows)):
        y=56+42*index
        draw.rectangle((15,y-4,1420,y+36),fill="#deebf2" if index==0 else "#f0f4f6" if index%2 else "white")
        for column,field in enumerate(FIELDS):
            label=row[field] if column==0 or index==0 else f"{Decimal(row[field]):,.2f}"
            if draw.textlength(label,font=font)>((200,400,400,400)[column]-16):
                raise ValueError("Table text clipping")
            draw.text((x[column],y),label,font=font,fill="black")
    draw.text((22,318),"Official numeric records, local illustration. No change of currency unit.",font=font,fill="black")
    return image


def render_chart(rows,font):
    image=Image.new("RGB",(1440,1020),"white")
    draw=ImageDraw.Draw(image)
    draw.text((22,10),"U.S. Treasury Debt to the Penny | nominal USD | locally rendered",font=font,fill="black")
    for column,(field,label) in enumerate(zip(FIELDS[1:],NAMES)):
        top=65+column*300
        values=[Decimal(row[field]) for row in rows]
        low,high=min(values),max(values)
        span=high-low
        draw.text((22,top),label+" (USD, values annotated)",font=font,fill="black")
        points=[]
        for i,(row,value) in enumerate(zip(rows,values)):
            x=155+i*265
            y=top+155 if span==0 else top+200-int((value-low)*Decimal(130)/span)
            points.append((x,y))
            draw.text((x-130,y-33),f"{value:,.2f}",font=font,fill="black")
            draw.text((x-65,top+230),row["record_date"],font=font,fill="black")
        draw.line(points,fill="#145c86",width=4)
        for x,y in points:
            draw.ellipse((x-5,y-5,x+5,y+5),fill="#145c86")
    draw.text((22,980),"Each panel uses its own vertical scale. Annotated point values retain all cents.",font=font,fill="black")
    return image


def png(image):
    import io
    stream=io.BytesIO()
    image.save(stream,format="PNG",optimize=False)
    return stream.getvalue()


def build(source,font_path,output):
    selected=select_rows(source.read_bytes())
    font_raw=font_path.read_bytes()
    font=ImageFont.truetype(str(font_path),22)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    manifest=dict(schema="treasury-timeseries-120-rows-v1",publisher="U.S. Treasury Bureau of the Fiscal Service",
        source_page=PAGE,source_api=API,source_terms=TERMS,source_snapshot_SHA256=SNAPSHOT_SHA256,
        source_snapshot_bytes=source.stat().st_size,new_data_HTTP_bytes=0,acquisition="reused existing complete official API snapshot",
        per_split_reproduction_queries=queries(),cutoff="2025-12-31",fields=list(FIELDS),unit="nominal USD",
        dates_per_window=5,topics=list(TOPICS),media_origin="local Pillow rendering of original official numerical records",
        font_SHA256=sha(font_raw),pillow_version=PILLOW,previous_160_train_108_dev_unchanged=True,
        splits={},files_SHA256={})
    source_rows=[row for rows in selected.values() for row in rows]
    def write(name,raw):
        path=output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        manifest["files_SHA256"][name]=sha(raw)
    write("selected-source-rows.json",canonical(dict(source_snapshot_SHA256=SNAPSHOT_SHA256,data=source_rows)))
    image_ids=set()
    date_ids=set()
    window_ids=set()
    for split,rows in selected.items():
        inputs,labels,windows=[],[],[]
        topic_counts=Counter()
        for offset in range(0,len(rows),ROWS_PER_WINDOW):
            window=rows[offset:offset+ROWS_PER_WINDOW]
            window_id=sha(canonical(window))
            if window_id in window_ids or any(row["record_date"] in date_ids for row in window):
                raise ValueError("Window or date crossed split")
            window_ids.add(window_id)
            date_ids.update(row["record_date"] for row in window)
            windows.append(dict(id=window_id,dates=[row["record_date"] for row in window],source_rows=window))
            for modality,render in (("table",render_table),("chart",render_chart)):
                media=f"{split}/{window_id}-{modality}.png"
                payload=png(render(window,font))
                if sha(payload) in image_ids:
                    raise ValueError("Repeated media bytes across windows or splits")
                image_ids.add(sha(payload))
                write(media,payload)
                for topic,field,question,answer in questions(window):
                    identity=f"{window_id}:{modality}:{field}:{topic}"
                    prompt="<image>\n"+question+" Return only the answer, without currency symbols, grouping separators or explanation."
                    messages=[dict(role="user",content=prompt)]
                    if split=="train":
                        messages.append(dict(role="assistant",content=answer))
                    inputs.append(dict(id=identity,window_id=window_id,modality=modality,topic=topic,
                                       images=[media],messages=messages))
                    labels.append(dict(id=identity,window_id=window_id,topic=topic,field=field,answer=answer))
                    topic_counts[topic]+=1
        write(f"{split}/inputs.jsonl",b"".join(canonical(x) for x in inputs))
        write(f"{split}/labels.jsonl",b"".join(canonical(x) for x in labels))
        write(f"{split}/windows.jsonl",b"".join(canonical(x) for x in windows))
        manifest["splits"][split]=dict(raw_dates=len(rows),windows=len(windows),media=2*len(windows),
                                      examples=len(inputs),topic_counts=dict(topic_counts),
                                      input_messages_include_assistant=(split=="train"),
                                      time_interval=list(SPLITS[split]))
    with (output/"manifest.json").open("xb") as stream:
        stream.write(canonical(manifest))
    return dict(manifest_SHA256=sha((output/"manifest.json").read_bytes()),raw_rows=120,windows=24,
                images=48,examples=432,source_new_download_bytes=0,dataset_only=True,training_ready=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--font",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.font,args.output),sort_keys=True))
