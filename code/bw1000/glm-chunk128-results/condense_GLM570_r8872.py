from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=r/'GLM570-chunk128-numerical-negative-compact-r8872.json';q=json.loads(p.read_bytes())
q['full_compact_source']={'file':p.name,'SHA256':sha(p)}
rows=q.pop('owner_reports');q['owner_report_name_pattern']='glm-transient-Model-r8872-{run}-stage{stage}-rank{rank}.json'
q['owner_columns']=['stage','rank','SHA256','normal','A128goldEOS','beforeQA','Ctoken0_reject','destroy_empty','load_s'];q['owners']=[[x[k]for k in q['owner_columns']]for x in rows]
rows=q.pop('phase_source_files');q['phase_source_name_pattern']='GLM-Model-phase-r8872-{run}-stage{stage}-rank{rank}.jsonl';q['phase_source_SHA256_stage_rank_order']=[x['SHA256']for x in rows]
q['phase_free_not_peak']=[{'stage':x['stage'],'minimum_free_by_saved_phase':{k:v['min']for k,v in x['saved_free_not_peak'].items()if v}}for x in q['phase_free_not_peak']]
for x in q['C128']['failure_raw_CPU_checks']:x['mismatch_rows_only']=[y for y in x.pop('rows')if y['mismatch']]
q['original_watch']['audit_stderr_tail']=q['original_watch']['audit_stderr_tail'].strip()
p=r/'GLM570-chunk128-numerical-negative-compact-r8872-b.json';p.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());assert p.stat().st_size<=15360,p.stat().st_size
ready={'CPU_only':True,'GPU':False,'Git_mutation':False,'result':p.name,'result_SHA256':sha(p),'result_bytes':p.stat().st_size,'sourceclaim':'892566ca4cc82d65aefe5e98c666c99522f75aaf','producer_sources':[{'path':n,'SHA256':sha(r/n)}for n in ['compact_GLM570_negative_r8872.py','condense_GLM570_r8872.py']],'raw_full_compact':q['full_compact_source'],'stage_receipt_mechanical_evidence':{'path':'GLM-chunk128-stage-receipt-mechanical-r8872.json','SHA256':sha(r/'GLM-chunk128-stage-receipt-mechanical-r8872.json')},'original_stage_parser_fix_source':{'path':'stage_GLM_chunk128_Model_r8872.py','SHA256':sha(r/'stage_GLM_chunk128_Model_r8872.py')},'no_shared_repository_files_written':True}
p2=r/'GLM570-publication-ready-r8872.json';p2.write_bytes((json.dumps(ready,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(ready))
