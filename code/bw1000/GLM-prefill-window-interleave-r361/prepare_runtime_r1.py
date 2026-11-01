"""Package the published361 consumer onto exact355, with new private identity."""
from pathlib import Path
import hashlib,json,ast
D=Path(__file__).resolve().parent;BASE=D.parent/'GLM-two-cohort-Model-r355';OUT=D/'runtime-r1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SNAP='11cf0416e344e0fe8795c809c2cf54a87588f9dc6c8093a8d203749068deed8e';PARENT='32f24ae36f4148c9eac9f86294a4bcccb3c1f269d6b1123cb9750d88a0693366'
assert sha(D/'SOURCE_MANIFEST.json')==SNAP and sha(BASE/'owner/SOURCE_MANIFEST.json')==PARENT
snapshot=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
for n,h in snapshot['files_SHA256'].items():assert sha(D/n)==h,n
for sub,count in [('owner',43),('controller',7)]:
 mf=json.loads((BASE/sub/'SOURCE_MANIFEST.json').read_bytes());assert len(mf['files_SHA256'])==count
 for n,h in mf['files_SHA256'].items():
  assert sha(BASE/sub/n)==h,n;p=OUT/sub/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes((BASE/sub/n).read_bytes())
O=OUT/'owner';C=OUT/'controller'
(O/'owner_r290.py').write_bytes((D/'owner_r290.py').read_bytes());(O/'prefill_interleave.py').write_bytes((D/'prefill_interleave.py').read_bytes())
def namespace(s):return s.replace('GLM-two-cohort-Model-r355','GLM-prefill-window-interleave-r361/runtime-r1').replace('GLM-r355','GLM-r361').replace('GLM_R355','GLM_R361').replace('GLM355','GLM361')
for n in ['owner_r290.py','model_entry_r290.py','GPU_inside_r290.sh','client_r355.py','workload_r355.py']:
 p=O/n;p.write_bytes(namespace(p.read_bytes().decode()).encode())
edit_failure_old="        report['failed_owner_cleanup_attempted']=True;report['service_traceback']=traceback.format_exc()\r\n        try:check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown());report['failed_owner_local_cleanup_completed']=True\r\n        except BaseException:report['failed_owner_cleanup_traceback']=traceback.format_exc()"
entry=(O/'model_entry_r290.py').read_bytes().decode()
newline='\r\n'if '\r\n'in entry else '\n'
edit_failure_old=edit_failure_old.replace('\r\n',newline)
assert edit_failure_old in entry
entry=entry.replace(edit_failure_old,("        report['failed_owner_failstop']=True;report['service_traceback']=traceback.format_exc()\n        # Native may already have advanced. Original guarded() exits2; controller\n        # stops all owned stage handles. Never reset/replay a partial frontend ledger.").replace('\n',newline))
(O/'model_entry_r290.py').write_bytes(entry.encode())
for n in ['backend.py','controller.py','contract.py','invoke.py']:
 p=C/n;p.write_bytes(namespace(p.read_bytes().decode()).encode())
def edit(path,replacements):
 s=path.read_bytes().decode()
 for old,new in replacements:
  assert old in s,(path.name,old);s=s.replace(old,new)
 path.write_bytes(s.encode())
edit(O/'workload_r355.py',[("PLAN=(False,True,False,True,True,False)","PLAN=(False,True,False,True,True,False)\nCANDIDATE='r361-prefill-boundary-interleave-decode-defer-fixed-ON'"),("requests=2,ordinary_budget=4096,rows=2,context=32768)","requests=2,ordinary_budget=4096,rows=2,context=32768,candidate=CANDIDATE,decode_defer_fixed=True)")])
edit(O/'owner_r290.py',[
 ("from workload_r355 import PLAN,load,marker_identity","from workload_r355 import PLAN,load,marker_identity,CANDIDATE"),
 ("emitter_step=report['actual_step'],static_cohort=True,incremental_HTTP=False)","emitter_step=report['actual_step'],static_cohort=True,incremental_HTTP=False,candidate=CANDIDATE,decode_defer_fixed=True,prefill_interleave_plan=list(PLAN))"),
 ("generated[row].append(token)","generated[row].append(token)\n                pair.setdefault('feedback_events',[]).append(dict(row=row,produced=len(generated[row]),phase=active_phase[0],after_Native_calls=len(pair['calls'])))"),
 ("            pair['decode_defer_fixed_both_arms']=True", "            pair['decode_defer_fixed_both_arms']=True\n            boundaries=[x for x in interleave_events if x['kind']=='complete_long_prefill_group']\n            advances=[x for x in interleave_events if x['kind']=='short_decode_at_complete_long_W4_boundary']\n            expected_groups=(len(fixtures[1]['ids'])+2047)//2048\n            assert (len(boundaries)==expected_groups and len(advances)>0) if enabled else not interleave_events,'Prefill policy path not actually consumed; stop first warm without retry'\n            assert states[0].past==len(fixtures[0]['ids'])+len(advances) and states[0].produced==1+len(advances)\n            assert states[1].past==len(fixtures[1]['ids']) and states[1].produced==1\n            assert [x['short_produced']for x in advances]==list(range(2,2+len(advances)))\n            pair['prefill_consumption']=dict(candidate=CANDIDATE,enabled=enabled,complete_long_W4_groups=len(boundaries),short_advances=len(advances),states=[dict(row=x.row,past=x.past,produced=x.produced,done=x.done)for x in states],qualified=True)"),
 ("            decode_start=time.monotonic()", "            active_phase[0]='decode'\n            pair['decode_schedule_scope']='Remaining decode after both prefills; ON includes earlier short advances. Use full timeline and complete schedule, never trimmed decode as whole gain.'\n            decode_start=time.monotonic()"),
 ("r355_pair=pair_index,r355_row=row,r355_enabled=enabled", "r361_pair=pair_index,r361_row=row,r361_enabled=enabled"),
 ])
edit(O/'client_r355.py',[
 ("from workload_r355 import PLAN,load,marker_identity", "from workload_r355 import PLAN,load,marker_identity,CANDIDATE"),
 ("    R=pathlib.Path('/refactor');out=", "    assert ready['candidate']==CANDIDATE and ready['decode_defer_fixed']is True and ready['prefill_interleave_plan']==list(PLAN)\n    R=pathlib.Path('/refactor');out="),
 ("import argparse,concurrent.futures,json,pathlib,time,urllib.request,hashlib,os", "import argparse,concurrent.futures,json,pathlib,time,urllib.request,hashlib,os,threading"),
 ("result=dict(completed=False,passed=False,HTTP_requests=0,logical_requests=12,pairs=[],buffered_HTTP=True,online_SLO_qualified=False)", "result=dict(completed=False,passed=False,planned_HTTP=12,planned_logical_requests=12,attempted_HTTP=0,completed_HTTP=0,quality_passed_HTTP=0,HTTP_requests=0,logical_requests=12,pairs=[],buffered_HTTP=True,online_SLO_qualified=False,candidate=CANDIDATE,decode_defer_fixed=True)\n    result_lock=threading.Lock()"),
 ("        started=time.monotonic()", "        with result_lock:result['attempted_HTTP']+=1;save()\n        started=time.monotonic()"),
 ("        ended=time.monotonic();assert status==200", "        ended=time.monotonic()\n        with result_lock:result['completed_HTTP']+=1;save()\n        assert status==200"),
 ("        assert usage['prompt_tokens']==len(f['ids'])", "        assert usage['prompt_tokens']==len(f['ids'])\n        with result_lock:result['quality_passed_HTTP']+=1;save()"),
 ("    try:\n        for index", "    save()\n    try:\n        for index"),
 ("assert result['HTTP_requests']==12", "assert result['HTTP_requests']==result['attempted_HTTP']==result['completed_HTTP']==result['quality_passed_HTTP']==12"),
 ])
preparation=dict(scope='361 interleave only at complete longW4 boundaries; decode defer fixedON both arms; six static pairs, not openarrival/SLO',base_355_owner_manifest_SHA256=PARENT,interleave_snapshot_manifest_SHA256=SNAP,Native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',GPU_GO=False,rows=2,TP=8,PP=4,cards=32,context=32768,ordinary_budget=4096,pairs=6,HTTP=12,plan=[False,True,False,True,True,False],decode_defer_fixed_both_arms=True,no_CXX_recompile=True,no_SDK_weights_quantization_or_memory_cache_change=True)
(O/'PREPARATION.json').write_bytes((json.dumps(preparation,indent=2)+'\n').encode())
edit(C/'contract.py',[("GPU_SECONDS,NATIVE_SECONDS,CPU_SECONDS=7200,6600,7800", "GPU_SECONDS,NATIVE_SECONDS,CPU_SECONDS=7200,6600,7800\nCANDIDATE='r361-prefill-boundary-interleave-decode-defer-fixed-ON'"),("    assert g['no_retry_reload']is True", "    assert g['no_retry_reload']is True\n    assert g['candidate']==CANDIDATE and g['decode_defer_fixed']is True and g['arm_variable']=='prefill_interleave'")])
edit(C/'controller.py',[
 ("from contract import validate,bundle,sha,STAGES,TERMINAL,NATIVE,GPU_SECONDS", "from contract import validate,bundle,sha,STAGES,TERMINAL,NATIVE,GPU_SECONDS,CANDIDATE"),
 ("    from namespace_binding import verify_namespace", "    base=bundle(Path(root)/'incoming/GLM-two-cohort-Model-r355/owner','"+PARENT+"')\n    current=json.loads((Path(root)/'incoming/GLM-prefill-window-interleave-r361/runtime-r1/owner/SOURCE_MANIFEST.json').read_bytes())\n    assert current['base_355_owner_manifest_SHA256']=='"+PARENT+"' and current['interleave_snapshot_manifest_SHA256']=='"+SNAP+"'\n    assert current['files_SHA256']['prefill_interleave.py']=='54861403adad4a6143eb260a5735fb0fdfbaf9e5202457202dc4a4a7054bb932'\n    assert set(current['files_SHA256'])==set(base['files_SHA256'])|{'prefill_interleave.py'}\n    for name,h in base['files_SHA256'].items():\n        if name not in current['changed_from_355']:assert current['files_SHA256'][name]==h\n    from namespace_binding import verify_namespace"),
 ("                assert len(pair['requests'])==2", "                assert len(pair['requests'])==2 and pair['decode_defer_fixed_both_arms']is True\n                hit=pair['prefill_consumption'];assert hit['candidate']==CANDIDATE and hit['qualified'] and hit['enabled']==pair['enabled']\n                assert (hit['complete_long_W4_groups']==5 and hit['short_advances']>0)if pair['enabled']else hit['short_advances']==hit['complete_long_W4_groups']==0"),
 ("    assert q['passed']and q['HTTP_requests']==q['logical_requests']==12", "    assert q['candidate']==CANDIDATE and q['decode_defer_fixed']is True\n    assert q['passed']and q['HTTP_requests']==q['logical_requests']==q['attempted_HTTP']==q['completed_HTTP']==q['quality_passed_HTTP']==12"),
 ("C2M1_PAIRED_MODEL_PASS_PENDING_COST_AUDIT", "PREFILL_INTERLEAVE_MODEL_PASS_PENDING_FULL_TIMELINE_AUDIT"),
 ])
# Exact original algorithm dependencies and loader/prepared paths stay untouched.
for n in ['consumer.py','decode_wave_r9225.py','glm_prefill_execute_r236.py','native_binding.py','prepared_family_entry_r268.py','preweight_r290.py','cleanup_owned_r290.py','loader_guard_r290.py']:
 assert (O/n).read_bytes()==(BASE/'owner'/n).read_bytes(),n
assert sha(O/'prefill_interleave.py')==snapshot['files_SHA256']['prefill_interleave.py']
for p in OUT.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
for p in OUT.rglob('*.sh'):assert b'\r'not in p.read_bytes(),str(p)
g=json.loads((BASE/'execution-GO.template.json').read_bytes());g.update(run='GLM-C2M1-r361-a',candidate='r361-prefill-boundary-interleave-decode-defer-fixed-ON',decode_defer_fixed=True,arm_variable='prefill_interleave',owner_manifest_SHA256='0'*64,controller_manifest_SHA256='0'*64,CPU_receipt_SHA256='0'*64,execute_once=False,scope=preparation['scope'])
(OUT/'execution-GO.template.json').write_bytes((json.dumps(g,indent=2)+'\n').encode())
print(json.dumps({'new_runtime':str(OUT),'snapshot_unchanged':sha(D/'SOURCE_MANIFEST.json')==SNAP,'algorithm_helper_exact':True,'GPU_GO':False}))
