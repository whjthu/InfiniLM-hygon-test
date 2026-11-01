"""Mechanical private349 clone. Never changes parent/live files."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;G=D.parents[2]
P=D.parent/'GLM-W4-cross-workload-r349/owner';O=D/'owner';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'SOURCE_MANIFEST.json')=='4e5bbea816c8569fd7fbf382f2efc44787f8937dcf096fbacb889b1c8b917366'
mf=json.loads((P/'SOURCE_MANIFEST.json').read_bytes())
for n,h in mf['files_SHA256'].items():
 b=(P/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def change(n,pairs):
 p=O/n;s=p.read_text(encoding='utf8')
 for old,new in pairs:
  assert old in s,(n,old);s=s.replace(old,new)
 p.write_text(s,encoding='utf8',newline='\n')
change('deadline_r290.py',[('SERVICE_SECONDS=9120','SERVICE_SECONDS=6120'),('HARD_NATIVE_SECONDS=10200','HARD_NATIVE_SECONDS=6600')])
change('service_budget_r204.py',[('10200','6600'),('10800','7200'),('12000','7800'),('NORMAL_CLEANUP_RESERVE=1080','NORMAL_CLEANUP_RESERVE=480')])
change('prepared_family_entry_r268.py',[('real_cache_rows=1,','real_cache_rows=2,')])
change('service_queue_bridge_r268.py',[("assert not any(q['started']and not q['finished']for q in self._journal.values())","assert sum(q['started']and not q['finished']for q in self._journal.values())<2"),('original Native serial FIFO; HTTP entry concurrent, not native batching','Private current038d C2M1 finite cohort; two stable Native rows; buffered HTTP')])
change('preengine_r133.py',[("os.environ['GLM_CACHE_ROWS_R3627']=='1'","os.environ['GLM_CACHE_ROWS_R3627']=='2'")])
change('model_entry_r290.py',[
 ('assert(ROWS,CAPACITY)==(1,32768)','assert(ROWS,CAPACITY)==(2,32768)'),
 ("report_path = r / 'results' / f'GLM-latest-full-owner-r290-{tag}-stage{stage}-rank{rank}.json'","report_path = r / 'results' / f'GLM-r355-{tag}-stage{stage}-rank{rank}.json'"),
 ("'GLM-latest-full-owner-error-r290-'","'GLM-r355-error-'"),
 ("f'GLM-latest-full-owner-r290-control-{tag}-ids.json'","f'GLM-r355-control-{tag}-ids.json'"),
 ("benchmark={'generation_config':generation,'prefill_pair_r240':{'fixture_card':str(pathlib.Path(__file__).parent/'FIXTURE_CARD.json'),'output_root':'/refactor/results/GLM-latest-full-owner-r290-pair-'+tag}}","benchmark={'generation_config':generation,'fixture_card':str(pathlib.Path(__file__).parent/'FIXTURE_CARD.json')}"),
 ("print('GLM_REGULAR_SERVICE_R204_NORMAL',stage,flush=True)","print('GLM_R355_NORMAL',stage,flush=True)"),
 ("if time.monotonic()-begin>1200:raise TimeoutError('GLM benchmark identities missing')","_original_deadline_check('waiting original WORLD identities',120)\n        if time.monotonic()-begin>300:raise TimeoutError('GLM benchmark identities missing')"),
 ("report['hardware_gate_before_create']=hardware","report['hardware_gate_before_create']=hardware\n    report['C2_constructor_rows']=ROWS\n    lib.modern_glm_constructor_census_r4158.argtypes=[ctypes.c_void_p];lib.modern_glm_constructor_census_r4158.restype=ctypes.c_char_p\n    census=json.loads(lib.modern_glm_constructor_census_r4158(ctx));assert census['rows']==2 and census['capacity']==32768\n    report['constructor_logical_census']=census\n    report['actual_step']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']"),
 ("_preloaded_probe_r3091(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,lambda x:phase_marker(x,rank),_protocol_preweight[\"prepared\"])","try:\n        _preloaded_probe_r3091(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,lambda x:phase_marker(x,rank),_protocol_preweight[\"prepared\"])\n    except BaseException:\n        report['failed_owner_cleanup_attempted']=True;report['service_traceback']=traceback.format_exc()\n        try:check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown());report['failed_owner_local_cleanup_completed']=True\n        except BaseException:report['failed_owner_cleanup_traceback']=traceback.format_exc()\n        report_path.write_text(json.dumps(report,indent=2));raise"),
 ])
change('GPU_inside_r290.sh',[
 ('GLM_CACHE_ROWS_R3627=1','GLM_CACHE_ROWS_R3627=2'),
 ('export GLM_STAGE_R5040="$SLURM_PROCID"','test "${GLM_STAGE_R5040:?}" -ge 0\ntest "$GLM_STAGE_R5040" -le 3\ncase "$(hostname -s)" in\n a02r3n03) test "$GLM_STAGE_R5040" = 0; test "$SLURM_JOB_ID" = 910371;;\n a02r3n07) test "$GLM_STAGE_R5040" = 1; test "$SLURM_JOB_ID" = 910371;;\n a02r4n04) test "$GLM_STAGE_R5040" = 2; test "$SLURM_JOB_ID" = 910374;;\n a02r4n05) test "$GLM_STAGE_R5040" = 3; test "$SLURM_JOB_ID" = 910374;;\n *) exit 2;;\nesac'),
 ('GLM-W4-cross-workload-r349/runtime_v4/owner','GLM-two-cohort-Model-r355/owner'),('10200','6600')])
card=json.loads((P/'FIXTURE_CARD.json').read_bytes());card['fixtures']=card['fixtures'][:2]
card['scope']='r355 two unchanged complete fixtures;6 pairs OFF/ON/OFF/ON/ON/OFF;12 logical requests; no onlineSLO'
(O/'FIXTURE_CARD.json').write_text(json.dumps(card,indent=2)+'\n',encoding='utf8')
for n in ['consumer.py','native_binding.py','decode_wave_r9225.py']:
 (O/n).write_bytes((D.parent/'GLM-two-cohort-CPU-r354'/n).read_bytes())
print(json.dumps({'private_owner_prepared':True,'GPU':0,'parent_files':len(mf['files_SHA256'])}))
