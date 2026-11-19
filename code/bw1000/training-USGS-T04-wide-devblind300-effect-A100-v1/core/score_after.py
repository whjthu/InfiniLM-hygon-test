"""First scoring only after all300 sealed outputs + independent terminal. No model."""
import argparse,importlib.util,json,random,sys
from collections import Counter,defaultdict
from pathlib import Path
from effect_contract import ARMS,dump,eos_observation,need,pinned,sha
from response_protocol import content_prediction
D=Path(__file__).resolve().parent

def aggregate(rows,preds):
    return dict(denominator=len(rows),correct=sum(r['correct'] for r in rows),failures=sum(not r['correct'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,
        content_parse_failures=sum(not p['content_parse_valid'] for p in preds),
        strict_FINAL_syntax=sum(p['strict_FINAL_format'] for p in preds),strict_FINAL=sum(p['strict_FINAL_format'] and p['natural_EOS'] and not p['cap_hit'] for p in preds),
        strict_format_correct=sum(r['correct'] and p['strict_FINAL_format'] for r,p in zip(rows,preds)),
        natural_EOS=sum(p['natural_EOS'] for p in preds),cap_hits=sum(p['cap_hit'] for p in preds))

def main(a):
    plan=json.loads((D/'PLAN.json').read_bytes())
    audit=json.loads(pinned(a.independent_audit,dict(bytes=a.audit_bytes,SHA256=a.audit_sha),8*1024**2))
    need(audit['engineering_metadata_verified'] is True and audit['independent_EOS_metadata_verified'] is True
         and audit['verified_predictions']==300 and audit['gold_read'] is False and audit['planned_generations']==300
         and audit['context']['source_manifest_SHA256']==sha((D.parent/'SOURCE_FILES.json').read_bytes()),'INDEPENDENT_COMPLETE300_TERMINAL_BEFORE_GOLD')
    final=json.loads(pinned(a.run/'generation-finalized.json',dict(bytes=a.finalized_bytes,SHA256=a.finalized_sha)))
    original_final=audit['original_small_files']['run/native/generation-finalized.json']
    need(original_final['bytes']==a.finalized_bytes and original_final['SHA256']==a.finalized_sha and final['epoch']==audit['context']['epoch'],'ORIGINAL_FINAL_SEAL_PIN')
    need(final['passed'] is True and final['all_predictions_finalized_before_gold'] is True and final['all_prediction_files_closed_before_gold'] is True
         and final['arms']==list(ARMS) and final['per_arm']==100 and final['planned_generations']==300 and final['split_denominators']=={'dev':50,'blind':50}
         and final['input_SHA256']=={k:v['SHA256'] for k,v in plan['inputs'].items()}
         and final['dev_gold_read'] is False and final['blind_gold_read'] is False,'FIXED300_SEAL_PRIVACY')
    raw_predictions={}
    for arm in ARMS:
        rec=final['arm_results'][arm];need(rec['planned']==rec['attempted']==rec['completed']==100 and rec['missing']==0,'FULL100_ARM')
        name=arm+'-predictions-private.jsonl';need(rec['file']==name,'EXACT_PREDICTION_NAME')
        saved=audit['original_small_files']['run/native/'+name];need(saved['bytes']==rec['bytes'] and saved['SHA256']==rec['SHA256'],'INDEPENDENT_ARM_PIN')
        rows=[json.loads(x) for x in pinned(a.run/name,rec).splitlines()];need(len(rows)==100 and [r['ordinal'] for r in rows]==list(range(100)),'ORDERED100_PREDICTIONS')
        for i,r in enumerate(rows):
            need(r['arm']==arm and r['split']==('dev' if i<50 else 'blind') and r['split_ordinal']==i%50 and r['native_field_identity_verified'] is True,'PREDICTION_SPLIT_BINDING')
            observed=eos_observation(r['raw_generated_token_ids'],r['raw_EOS_ids']);need(all(type(r[k]) is type(v) and r[k]==v for k,v in observed.items()),'ACTUAL_EOS_CAP')
        raw_predictions[arm]=rows
    # All300 complete raw predictions are pinned before any dataset/gold access.
    need(a.output.is_absolute() and a.output.parent.is_dir() and not a.output.exists(),'NEW_PRIVATE_SCORE_OUTPUT')
    attempt=a.run/'fixed300-first-gold-attempt.json'
    dump(attempt,dict(epoch=final['epoch'],original_context=audit['context'],all300_sealed=True,gold_read=False,max_attempts=1,replay_allowed=False))
    protocol=json.loads(pinned(D/'EVALUATION_PROTOCOL.json',plan['protocol_pin'],16384));pinned(D/'response_protocol.py',plan['content_parser'])
    manifest=json.loads(pinned(a.dataset/'manifest-private.json',plan['dataset_manifest'],32768))
    need(manifest['actual_inputs']==150 and manifest['actual_charts']==30,'ORIGINAL_NEW_DATASET')
    labels={};schemas={};predictions={};results={};private_rows={};difficulty={};summary={}
    for name,pin in plan['frozen_scorer'].items():pinned(D/'frozen_scorer'/name,pin)
    sys.path.insert(0,str(D/'frozen_scorer'));spec=importlib.util.spec_from_file_location('frozen_score',D/'frozen_scorer/score_once.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    for split in ('dev','blind'):
        # Schema is gold-free; content parser never receives an answer.
        schema_name=split+'/response-schemas-private.jsonl';schema_rows=[json.loads(x) for x in pinned(a.dataset/schema_name,manifest['outputs'][schema_name]).splitlines()]
        schemas[split]={x['id']:x for x in schema_rows};need(len(schema_rows)==len(schemas[split])==50,'FULL50_INPUT_SCHEMAS')
        label_name=split+'/labels-private.jsonl';need(manifest['outputs'][label_name]==plan['labels'][split],'EXACT_NEW_LABEL_PIN')
        labels[split]=[json.loads(x) for x in pinned(a.dataset/label_name,plan['labels'][split]).splitlines()]
        need(len(labels[split])==len({x['id'] for x in labels[split]})==50 and {x['id'] for x in labels[split]}==set(schemas[split]),'FULL50_NEW_GOLD')
        groups={r['group'] for r in labels[split]};charts={r['chart_id'] for r in labels[split]};need(len(groups)==len(charts)==10,'TEN_NEW_CLOSURE_GROUPS')
        chart_rows=defaultdict(dict)
        for q in labels[split]:chart_rows[q['chart_id']][q['kind']]=q
        degeneracy={}
        for chart,kinds in chart_rows.items():
            q=next(iter(kinds.values()));ids=schemas[split][q['id']]['event_ids'];full=' '.join(ids)
            # Deterministic input-constancy strata from independently validated displayed-value labels,
            # only after sealing; not a new image/number extraction or a selection rule.
            const_depth=kinds['maximum_depth']['answer']==full;const_magnitude=kinds['maximum_reported_number']['answer']==full
            for kind,r in kinds.items():degeneracy[r['id']]=(kind in ('depth_change','maximum_depth') and const_depth) or (kind=='maximum_reported_number' and const_magnitude)
        expected=plan['input_difficulty_reference'][split]
        need(sum(degeneracy.values())==expected['input_degenerate_tasks_total'],'PREDECLARED_INPUT_DEGENERACY_TOTAL')
        difficulty[split]=dict(input_degenerate_tasks=sum(degeneracy.values()),single_event_QA=sum(q['single_event_window'] for q in labels[split]),charts=10,closure_groups=10,origin='Independently validated input metadata plus post-seal label all-ties structure; no answer printed or selection changed')
        results[split]={};predictions[split]={};private_rows[split]={};summary[split]={}
        for arm in ARMS:
            raw=[r for r in raw_predictions[arm] if r['split']==split]
            need([x['id'] for x in raw]==[x['id'] for x in labels[split]],'SAME50_ORDERED_GOLD_IDS')
            preds=[content_prediction(r,schemas[split][r['id']]) for r in raw];predictions[split][arm]=preds
            score=scorer.score(labels[split],preds);need(score['denominator']==50,'FULL_FAILURE_DENOMINATOR');results[split][arm]=score
            summary[split][arm]=aggregate(score['rows'],preds)
            strata={}
            for dimension in ('kind','group','single_multi','degenerate'):
                buckets=defaultdict(list)
                for i,q in enumerate(labels[split]):
                    key=q[dimension] if dimension in ('kind','group') else ('single' if q['single_event_window'] else 'multi') if dimension=='single_multi' else ('input_degenerate' if degeneracy[q['id']] else 'nondegenerate')
                    buckets[key].append(i)
                strata[dimension]={key:aggregate([score['rows'][i] for i in ix],[preds[i] for i in ix]) for key,ix in buckets.items()}
            summary[split][arm]['strata']=strata;private_rows[split][arm]=score['rows']
        paired={}
        for baseline in ('initial','generic25'):
            differences={q['id']:int(d['correct'])-int(b['correct']) for q,d,b in zip(labels[split],results[split]['domain25']['rows'],results[split][baseline]['rows'])}
            group_values=[]
            for group in sorted(groups):
                values=[differences[q['id']] for q in labels[split] if q['group']==group];need(len(values)==5,'FIVE_CORRELATED_QA_PER_CLUSTER');group_values.append(sum(values)/len(values))
            rng=random.Random(42);boot=sorted(sum(group_values[rng.randrange(10)] for _ in range(10))/10 for _ in range(10000))
            vals=list(differences.values());dp=predictions[split]['domain25'];bp=predictions[split][baseline];comparable=[i for i in range(50) if dp[i]['content_parse_valid'] and bp[i]['content_parse_valid']]
            parse_comparison=dict(full_denominator=50,both_content_parsed=len(comparable),domain_only_parsed=sum(d['content_parse_valid'] and not b['content_parse_valid'] for d,b in zip(dp,bp)),baseline_only_parsed=sum(b['content_parse_valid'] and not d['content_parse_valid'] for d,b in zip(dp,bp)),both_parsed_domain_wins=sum(vals[i]==1 for i in comparable),both_parsed_domain_losses=sum(vals[i]==-1 for i in comparable),interpretation='Ancillary format-comparability subset; primary denominator remains50; no semantic-capability inference')
            paired[baseline]=dict(parse_comparability=parse_comparison,domain_minus_baseline=sum(vals)/50,wins=vals.count(1),losses=vals.count(-1),ties=vals.count(0),group_cluster_bootstrap95=[boot[249],boot[9749]],clusters=10,correlated_QA=50,seed=42,replicates=10000,independence_not_established=True,interpretation='Descriptive small-cluster interval; not formal significance, general domain capability or OOD qualification')
        summary[split]['paired_domain_differences']=paired
    output=dict(state='FIRST_FIXED300_NEW_DEV_BLIND_SCORE',summary=summary,input_difficulty=difficulty,
        original_all150_degenerate_tasks=plan['input_difficulty_reference']['all']['input_degenerate_tasks_total'],gold_read_after300_seal=True,dev_gold_read=True,blind_gold_read=True,blind_now_exposed=True,
        raw_row_scores=private_rows,protocol_SHA256=plan['protocol_pin']['SHA256'],parser_SHA256=plan['content_parser']['SHA256'],
        checkpoint_selection_or_tuning=False,old_scores_recomputed=False,domain_capability_qualified=False,Model_GO=False,formal17=False)
    dump(a.output,output)
    # Only aggregate metrics; no raw IDs, questions, target values or predicted text.
    public={k:v for k,v in output.items() if k!='raw_row_scores'}
    for split in ('dev','blind'):
        for arm in ARMS:public['summary'][split][arm]['strata']['group']={sha(k.encode()):v for k,v in public['summary'][split][arm]['strata']['group'].items()}
    print(json.dumps(public,allow_nan=False))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','dataset','output','independent-audit'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('audit-bytes','finalized-bytes'):p.add_argument('--'+key,type=int,required=True)
    for key in ('audit-sha','finalized-sha'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
