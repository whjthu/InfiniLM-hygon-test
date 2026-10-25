from pathlib import Path
import json,hashlib,ast
root=Path(__file__).resolve().parent;r=root/'GLM-chunk128-Model-private-r8872'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rep(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)
def write(p,s):ast.parse(s);p.write_bytes(s.encode())
(r/'glm_chunk128_schedule_r8872.py').write_bytes((root/'glm_chunk128_schedule_r8872.py').read_bytes())
p=r/'audit_GLM_prefill_Model_r8872.py';s=p.read_text(encoding='utf-8')
s=s.replace('from glm_continuous_feedback_r8872 import tail_ready_metrics','from glm_continuous_feedback_r8872 import tail_ready_metrics\nfrom glm_chunk128_schedule_r8872 import prompt_calls,validate_events')
s=rep(s,"assert row['batch']==32 and row['micro']in(8,16)","assert row['batch']==16 and row['micro']==8")
s=rep(s,"    assert not post['observed_is_peak'];visited=[[]for _ in gen];native_sum=[0]*5", "    schedule_check=validate_events(events,gen,row['prompt_chunk_tokens'])\n    assert row['early_decode_outputs']==schedule_check['early_decode_outputs_before_last_prefill']\n    assert not post['observed_is_peak'];visited=[[]for _ in gen];native_sum=[0]*5")
s=rep(s,"     flat=sum(row['input_lengths'][i]for i in e['rows'])if e['round']==0 else len(e['rows']);assert flat==e['flat_tokens']", "     flat=sum(x[2]for x in e['requests']);assert flat==e['flat_tokens']")
s=rep(s,"     for i in e['rows']:assert e['round']<len(gen[i]);visited[i].append(e['round'])", "     for i in e['rows']:\n      assert e['round']<len(gen[i])\n      if e['round']>0 or e['prompt_offset']+e['prompt_count']==512:visited[i].append(e['round'])")
s=rep(s,"    pre=[e for e in events if e['round']==0];assert [e['rows']for e in pre]==[list(range(i,min(i+pm,row['batch'])))for i in range(0,row['batch'],pm)]", "    pre=[e for e in events if e['round']==0];assert [(e['rows'],e['requests'][0][1],e['requests'][0][2])for e in pre]==prompt_calls(16,row['prompt_chunk_tokens'])")
s=rep(s,"sum(row['input_lengths'][i]+e['round']for e in events for i in e['rows'])", "sum(q[1]+q[2]for e in events for q in e['requests'])")
s=s.replace('Same-load exactresident constructor32/fixedmask B32M8 thenB32M16 I512O128','Same-load exactresident constructor32/fixedmask B16M8 ordinaryA thenchunk128C I512O128')
s=s.replace("len(tailcases)in(2,4)","len(tailcases)==2")
write(p,s)
p=r/'GLM_transient_preflight_r8872.py';s=p.read_text(encoding='utf-8').replace('==(8,2,640,32,32,[8,16])','==(8,2,640,16,32,[8])');write(p,s)
p=r/'modern_glm_prefill_micro_r8872.py';s=p.read_text(encoding='utf-8').replace('B32M8thenM16','B16M8ordinaryAthenchunk128C');write(p,s)
p=r/'launch_GLM_prefill_Model_r8872.py';s=p.read_text(encoding='utf-8').replace("run='M16-8862-'","run='CHUNK128-8872-'").replace('B32M8 thenB32M16','B16M8 ordinaryA thenchunk128C');write(p,s)

observer=r/'GLM-transient-Model-observer-manifest-r8872.json';o=json.loads(observer.read_bytes())
for n in list(o['source_SHA256']):o['source_SHA256'][n]=sha(r/n)
o['source_SHA256']['glm_chunk128_schedule_r8872.py']=sha(r/'glm_chunk128_schedule_r8872.py')
observer.write_bytes((json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode())
p=r/'GLM-CROSSREQUEST-Model-source-manifest-r8872.json';m=json.loads(p.read_bytes())
m.update(active=16,performance_micro=[8],decode_micro_candidates=[8],prefill_token_chunks=[512,128],
 scope='CPUready pendingrootGO: single16DCU TP8PP2/B16M8 ordinaryA thenchunk128C, I512/O128 distinct477first16. One load, exactlytwo full128 arms. Same currentCPU210/native68bdbd/SDK/math, exactbanks constructor32/cap640/fixedadmittedmask; prefix/vectorON/PVtile2quantreuseOFF, original6208MiB/512MiB guards; QA/16normal/audit/post.',
 root_GO='CPU_READY_WAIT_FINAL_ROOT_GPU_GO',perf_labels=['perf-I512-B16-M8-A-N-r0','perf-I512-B16-M8-C128-N-r0'],
 diagnostic_budget='PROPOSAL45minhard2600inner, one load/exactlytwo full128 arms includingbefore/restoreQA/cleanup. Historical56735:10 includesB32M8/M16; thisB16chunkcostunknown. No extraarm/retry/extension.',
 math='No native changes; current per-query attention, fullDSA causalindexer, FP8 per-row Kblock quantizer. Chunked GEMM/route provider shape may vary; only final477 IDs/EOS+QA qualifies.',
 scope_hit='A2x8row full512prefill; C8x8row128prefill and3decode turns betweenremainingchunks, then originalM8 continuation. Both128actual outputs, firsttokenonlyafter512, no partial-logit publication.',
 mechanical_revision={'parent':'qualified567single16owner consumer/feedback;507sourceanalysisonly','change':'privatePythonchunk128+resumableoriginalfeedbackcursor; no sharedhelper overwrite or nativebuild','current_scope':'Aordinary512thenCchunk128, single16DCU/B16M8'},
 requested_resource={'allocation':910376,'nodes':['a02r4n12','a02r4n13'],'cards':16,'cpu_per_node':64,'memory_MiB_per_node':240000,'hard_seconds':2700,'inside_seconds':2600})
m['input_provenance']={**m['input_provenance'],'distinct':16,'global_rows':list(range(16)),'source_fixture_rows':32,'tokens_each':512,'padding':False,'candidate_source':'first16 original477distinct; sameexactinputsA/C'}
for n in list(m['source_SHA256']):m['source_SHA256'][n]=sha(r/n)
for n in ['glm_chunk128_feedback_r8872.py','glm_chunk128_schedule_r8872.py']:m['source_SHA256'][n]=sha(r/n)
m['observer_manifest_SHA256']=sha(observer)
p.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode())
entry=r/'dispatch_GLM_resident_exact_Model_r8872.py';s=entry.read_text(encoding='utf-8');s=s.replace('57f0e164d8daa6dac15c36afc85ddf620c52faccdb55da7fcb31e5c8e72b5432',sha(p));write(entry,s)
out={'CPU_only':True,'GPU_GO':False,'native_build':False,'private':str(r),'manifest_SHA256':sha(p),'entry_SHA256':sha(entry),'DSO_SHA256':m['model_SHA256'],'CPU_native_receipt':m['CPU_native_receipt'],'resource_proposal':m['requested_resource'],'shared_helper_preserved':sha(r/'kimi_cohort_feedback_idle_r4915.py')==sha(root/'GLM-M16-Model-private-r8866/kimi_cohort_feedback_idle_r4915.py'),'scope':'single16owner/B16M8 A-C128 full128/QA, pendingrootread; no upload/Git/GPU'}
assert out['shared_helper_preserved']
for n,h in m['source_SHA256'].items():assert sha(r/n)==h
for p2 in r.glob('*.py'):ast.parse(p2.read_text(encoding='utf-8'))
rp=root/'GLM-chunk128-review-package-r8872.json';rp.write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(out))
