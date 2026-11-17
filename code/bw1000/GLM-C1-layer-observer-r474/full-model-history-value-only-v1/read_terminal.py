"""Read the original FFN run after Slurm and the launcher have reached terminal state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from host_launch import ARM, COLUMNS, account, evidence, expand, verify_owners, verify_released, actual_resources
from shared import ARMS, D, HOST, NODES, PARENT, H, load_pinned, namespace, require, verify_manifest
from slurm_accounting import is_terminal
from budget import PAIR_SECONDS


def attention_verdict(observed):
    count = observed['observed_frames']
    require(observed['planned_ranks'] == 16 and observed['planned_frames'] == 48
            and type(count) is int and 0 <= count <= 48
            and type(observed['full_coverage']) is bool
            and observed['full_coverage'] == (count == 48), 'attention coverage denominator differs')
    full = observed['full_coverage']
    return dict(attention_schema_valid=True, attention_observed_rank_frames=count,
                attention_planned_rank_frames=48, full_attention_coverage=full,
                valid_attention_detail=full,
                attention_measurement_state='MEASURED_FULL48' if full else 'UNKNOWN_INSUFFICIENT_C1_FRAMES')


def read_terminal(run, claim_hash, request_id, commit):
    require(re.fullmatch(r"[0-9a-f]{32}", request_id) is not None, "invalid request identity")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid published commit")
    label, _ = namespace(run, ARM)
    claim = load_pinned(D / "GPU_RESOURCE_CLAIM.json", claim_hash)
    require(claim["schema"] == "glm474-history-value-model-claim-v1" and claim["run_id"] == run
            and claim["GPU_GO"] is True and claim["arm_order"] == [ARM]
            and claim["parent"] == PARENT and claim["nodes"] == NODES
            and claim["native_arms"] == {ARM: ARMS[ARM][1]}, "wrong published FFN claim")
    root = HOST / "results"
    dispatch = root / f"GLM474-history-value-model-{run}-dispatch"
    out = root / label
    files = {};total_read=0

    def read(path):
        nonlocal total_read
        stat=path.stat();require(stat.st_size<=128*1024**2 and total_read+stat.st_size<=2*1024**3,"reader exact-file budget")
        with path.open("rb") as stream:
            before=__import__("os").fstat(stream.fileno());raw=stream.read(stat.st_size);after=__import__("os").fstat(stream.fileno())
        identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        require(len(raw)==stat.st_size and identity(stat)==identity(before)==identity(after)==identity(path.stat()),"reader stable original file")
        total_read+=len(raw)
        files[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    request = read(dispatch / "request.json")
    owner = read(dispatch / "launcher-owner.json")
    terminal = read(dispatch / "launcher-terminal.json")
    require(all(v["run_id"] == run for v in (request, owner, terminal))
            and request["claim_SHA256"] == claim_hash
            and all(v["request_id"] == request_id and v["claim_commit"] == commit for v in (request, owner))
            and owner["stage"] == "LAUNCHER_STARTED"
            and all(type(owner[k]) is int and owner[k] > 0 for k in ("PID", "startticks"))
            and math.isfinite(request["epoch"]) and request["epoch"] > 0
            and owner["epoch"] == request["epoch"]
            and request["deadline"] == request["epoch"] + PAIR_SECONDS
            and math.isfinite(terminal["finished"]) and terminal["finished"] >= request["epoch"]
            and type(terminal["rc"]) is int, "wrong original launcher accounting")

    rows = {phase: account(f"{label}-{phase}", time.monotonic() + 12)
            for phase in ("pre", "gate", "post")}
    require(is_terminal(rows["gate"]), "original GPU accounting has no terminal row")
    require(all(row is None or is_terminal(row) for row in rows.values()),
            "a related Slurm step has not reached terminal state")
    report = dict(run_id=run, original_request_id=request_id, claim_commit=commit,
                  launcher_rc=terminal["rc"], planned_requests_per_rank=66,
                  planned_owner_ranks=16, accounting_columns=COLUMNS.split(","),
                  actual_accounting=rows, independent_post_clean=False, valid_ffn_evidence=False, valid_attention_detail=False,
                  attention_schema_valid=False, attention_observed_rank_frames=None, attention_planned_rank_frames=48,
                  full_attention_coverage=False, attention_measurement_state='UNKNOWN',
                  full_ffn_coverage=False, raw_artifact_SHA256=files, errors={})

    def verify_step(phase):
        row = rows[phase]
        require(row and row[2:4] == ["COMPLETED", "0:0"] and expand(row[5]) == set(NODES),
                f"original {phase} Slurm step did not complete cleanly")
        require(actual_resources(row, phase), 'original actual Slurm CPU/DCU resources differ')
        receipt = read(out / f"{phase}-receipt.json")
        require(receipt["accounting_columns"] == report["accounting_columns"]
                and receipt["terminal"] is True and receipt["passed"] is True
                and receipt["row"] == row, f"original {phase} receipt disagrees")
        for node in NODES:
            read(out / phase / (node + ".json"))
        return evidence(out, phase, row[0])

    try:
        before = verify_step("pre")
        after = verify_step("post")
        verify_released(before, after)
        report["independent_post_clean"] = True
    except (ValueError, KeyError, OSError, TypeError) as exc:
        report["errors"]["post"] = type(exc).__name__ + ": " + str(exc)

    owners={};workers={}
    report['planned_per_math_arm']=33;report['planned_screen_per_math_arm']=32
    report['owner_missing']=[];report['owner_original_errors']={}
    for node in range(2):
        try:workers[node]=read(out/f'worker-{node}.json')
        except (OSError,ValueError,KeyError) as exc:report['errors']['worker'+str(node)]=type(exc).__name__
    for rank in range(16):
        try:
            owners[rank]=read(out/f'owner-{rank:02d}.json')
            if owners[rank].get('error'):report['owner_original_errors'][str(rank)]=dict(error_type='ORIGINAL_OWNER_FAILURE',message_SHA256=hashlib.sha256(str(owners[rank]['error']).encode()).hexdigest())
        except (OSError,ValueError,KeyError) as exc:report['owner_missing'].append(rank);report['errors']['owner'+str(rank)]=type(exc).__name__
    report['saved_completed_per_rank']={str(r):sum(x.get('completed') is True for x in o.get('requests',[])) for r,o in owners.items()}
    report['quality']={arm:dict(planned=32,completed=0,missing=32,correct=0,wrong=0,cap=0,raw_answer_matches_gold=0) for arm in ('OFF','ON')}
    try:
        from scoring import grade
        from model_evidence import sampled_metrics
        report["screen_metrics"]={"OFF":[],"ON":[]}
        inputs=json.loads((D/'INPUT_BINDING.json').read_bytes());private=inputs['private_plan'];plan=load_pinned(HOST/Path(private['path']).relative_to(HOST),private['SHA256'])
        score_seen=set()
        for row in owners.get(0,{}).get('requests',[]):
            if row.get('screen_index') is None or not row.get('completed'):continue
            key=(row['arm'],row['screen_index']);require(row['arm'] in ('OFF','ON') and type(row['screen_index']) is int and 0<=row['screen_index']<32 and key not in score_seen,'quality unique planned slot');score_seen.add(key)
            pin=private['requests'][row['screen_index']];require(all(row[k]==pin[k] for k in ('ordinal','case_id','prompt_IDs_SHA256','body_SHA256')),'quality input pin')
            ids=row['generated_token_ids'];require(all(type(t)is int and 0<=t<154880 for t in ids) and not any(t in inputs['eos_token_ids'] for t in ids[:-1]),'quality token domain');natural=bool(ids and ids[-1] in inputs['eos_token_ids']);require(0<len(ids)<=4096 and row['generated_token_ids_SHA256']==hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest() and row['natural_EOS']==natural and row['cap_hit']==(not natural) and (natural or len(ids)==4096),'quality completion pin')
            metric=sampled_metrics(row,set(inputs['eos_token_ids']),set(inputs['tokenizer']['special_ids']));report['screen_metrics'][row['arm']].append(dict(screen_index=row['screen_index'],**metric))
            score=grade(plan['requests'][row['screen_index']]['case'],row['output_text'],natural_EOS=natural,cap_hit=row['cap_hit'],unexpected_tool=False);require(score==row['score'],'independent score differs')
            dest=report['quality'][row['arm']];dest['completed']+=1;dest['missing']-=1;dest['correct']+=int(score['correct']);dest['wrong']+=int(not score['correct']);dest['cap']+=int(score['cap_hit']);dest['raw_answer_matches_gold']+=int(score['answer_matches_gold'])
    except (OSError,ValueError,KeyError,TypeError) as exc:report['errors']['quality']=type(exc).__name__+': '+str(exc)
    try:
        require(terminal['rc']==0 and report['independent_post_clean'],'launcher or post failed; saved owners retained above')
        gate=verify_step('gate');manifest_path=out/'RUN_MANIFEST.json';manifest=read(manifest_path);verify_manifest(manifest,run,ARM,manifest['epoch'])
        require(manifest['claim_SHA256']==claim_hash and manifest['source_manifest_SHA256']==claim['source_manifest_SHA256'] and manifest['epoch']==request['epoch'],'original manifest binding')
        observed=verify_owners(out,ARM,H(manifest_path),rows['gate'][0],gate,owners,workers)['model_value_observation']
        report.update(valid_ffn_evidence=True,model_value_observation=observed,**attention_verdict(observed))
        report['source_derived_ON_exposure']=observed['all_ranks_source_exposure'];report['no_exposure_is_UNKNOWN']=not observed['all_ranks_source_exposure']
        report['numerical_comparison']=observed['numerical'];report['quality_complete']=all(v['completed']==32 for v in report['quality'].values()) and 'quality' not in report['errors']
        report['paired_candidate_evidence_complete']=bool(observed['all_ranks_source_exposure'] and observed['numerical'] and observed['numerical']['final_prefill_present'] and observed['numerical']['all_compared_equal'] and report['quality_complete'])
    except (ValueError,KeyError,OSError,TypeError) as exc:report['errors']['model_value']=type(exc).__name__+': '+str(exc)
    report.update(Model_qualified=False,performance_qualified=False,total_read_bytes=total_read)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("claim_sha256")
    parser.add_argument("request_id")
    parser.add_argument("commit")
    parser.add_argument("private_report", type=Path)
    args = parser.parse_args()
    report = read_terminal(args.run, args.claim_sha256, args.request_id, args.commit)
    with args.private_report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report.get("paired_candidate_evidence_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
