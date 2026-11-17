"""Project all324 planned outcomes and reuse the original strict Treasury score."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scope import PACKAGE, SRC, CPU, ARMS, require, read, pin
from treasury_holdout import _score, normalized


def score(source, output):
    binding=read(SRC/"INPUT_BINDING.json")
    require(pin(CPU/"dev/visual.jsonl",100000)["SHA256"]==binding["metadata_files"]["visual.jsonl"]["SHA256"],"frozen dev108 changed")
    requests=[json.loads(x) for x in (CPU/"dev/visual.jsonl").read_bytes().splitlines()]
    ids=[x["id"] for x in requests]
    topics={identity:next(x for x in ("lookup","signed_difference","maximum_date") if identity.endswith("_"+x)) for identity in ids}
    require(len(ids)==len(set(ids))==108 and Counter(topics.values())=={x:36 for x in set(topics.values())},"108/topic36 denominator differs")
    output.mkdir()
    files,raw_reports={},{}
    for arm in ARMS:
        raw_path=PACKAGE/"arms"/arm/"outputs"/(arm+"-raw.jsonl")
        rows={}; integrity_error=None; raw_pin=None
        if raw_path.is_file():
            raw_pin=pin(raw_path,4*1024**2)
            try:
                for line in raw_path.read_bytes().splitlines():
                    row=json.loads(line); identity=row["id"]
                    require(identity in ids and identity not in rows and row["finish_reason"] in ("stop","length")
                            and (row["answer"] is None or isinstance(row["answer"],str))
                            and type(row["completion_tokens"]) is int and 0<=row["completion_tokens"]<=64,
                            "invalid original prediction record")
                    rows[identity]=row
                require(list(rows)==ids[:len(rows)],"raw result order differs")
            except (ValueError,KeyError,TypeError) as error:
                integrity_error=type(error).__name__;rows={}
        projection=[];counts={x:Counter() for x in (*set(topics.values()),"all")}
        for identity in ids:
            row=rows.get(identity);answer=None
            if integrity_error:
                category="invalid_response"
            elif row is None:
                category="missing"
            elif row["finish_reason"]=="length":
                category="cap"
            elif not row["answer"]:
                category="empty"
            else:
                category="stop_nonempty";answer=row["answer"]
            projection.append(dict(id=identity,answer=answer))
            for key in (topics[identity],"all"):counts[key][category]+=1
        path=output/(arm+".jsonl")
        with path.open("x",encoding="utf8") as stream:
            for row in projection:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        files[arm]=path
        raw_reports[arm]=dict(planned_rows=108,collected_raw_rows=len(rows),raw_pin=raw_pin,
                              integrity_error=integrity_error,raw_outcome_counts={k:dict(v) for k,v in counts.items()})
    report=_score(source,"dev",files,("base","old_domain","domain"))
    for arm in ARMS:
        for topic in ("lookup","signed_difference","maximum_date","all"):
            strict=report["arms"][arm]["scores"][topic];raw=raw_reports[arm]["raw_outcome_counts"][topic]
            expected=108 if topic=="all" else 36
            require(strict["denominator"]==expected and sum(raw.values())==expected,"full scoring denominator differs")
            require(strict["missing"]==sum(raw.get(x,0) for x in ("missing","cap","empty","invalid_response")),
                    "strict missing projection and raw failures disagree")
            raw_reports[arm].setdefault("disjoint_scored_outcomes",{})[topic]=dict(
                correct=strict["correct"],wrong=strict["wrong"],invalid_format=strict["invalid_format"],
                **{x:raw.get(x,0) for x in ("missing","cap","empty","invalid_response")})
    report.update(scope="development_three_fresh_single_device_arms",arm_legend={"domain":"new328checkpoint5","old_domain":"old285checkpoint5","base":"original_no_adapter"},
                  all_arm_denominator=324,raw_outcomes=raw_reports,equal_budget_causal_victory=False,
                  formal_quality=False,blind_effect=False,formal_performance=False)
    with (output/"score.json").open("x",encoding="utf8") as stream:json.dump(report,stream,indent=2,allow_nan=False)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=score(args.source,args.output)
    print(json.dumps(dict(all_arm_denominator=324,score_path=str(args.output/"score.json"),formal_quality=False)))
