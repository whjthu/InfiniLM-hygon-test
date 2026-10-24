"""Create one O-D-O Model runtime from frozen5783, no launch here."""
import pathlib,json,hashlib,ast,re
r=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();oldhash='fb059e2a772afa32fa9617b252fa4d67ee749404a6dd80e3e1ddc03ccf919cf5';newhash='7de2990fe7028fe8514ce813a4a57f9d25315c4b2df4b84db0e8770af24dd910'
names=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r5783.json').read_text())['source_SHA256'];created=[]
def write(n,s):
 p=r/n
 if p.exists():assert p.read_text()==s,('existing draft differs',n);created.append(n);return
 if n.endswith('.py'):ast.parse(s)
 p.write_text(s,encoding='utf8',newline='\n');created.append(n)
def replace(s):return s.replace('r5783','r5799').replace(oldhash,newhash).replace('glm-layer77-model-r5775/libglm_layer77_model_r5775.so','glm-window-model-r5797/libglm_window_model_r5797.so')
for n in names:
 if 'r5783'not in n:continue
 s=replace((r/n).read_text());n=n.replace('r5783','r5799')
 if n=='glm_layer77_probe_r5799.py':
  s=s.replace("shadow=-1):","shadow=-1,model_window=False):",1)
  point='        cross_before=int(lib.modern_glm_crossrequest_prefill_calls_r7328(rank))';assert s.count(point)==1
  s=s.replace(point,'        if model_window:window.prepare()\n'+point)
  point='                    begin = time.perf_counter(); cpu_begin=time.thread_time();native_start_realtime_ns=time.time_ns()';assert s.count(point)==1
  s=s.replace(point,'                    if model_window:window.before(step,group)\n'+point)
  point='\n                begin = time.perf_counter(); cpu_begin=time.thread_time();native_start_realtime_ns=time.time_ns()';assert s.count(point)==1
  s=s.replace(point,'\n                if model_window:window.before(step,group)'+point)
  point='                    native = native_end-begin; native_cpu=native_cpu_end-cpu_begin';assert s.count(point)==1
  s=s.replace(point,point+'\n                    if model_window:window.after(label,step,group)')
  point='\n                native = native_end-begin; native_cpu=native_cpu_end-cpu_begin';assert s.count(point)==1
  s=s.replace(point,point+'\n                if model_window:window.after(label,step,group)')
  at=s.index("    report['layer77_scope']=")
  s=s[:at]+'''    from glm_window_control_r5799 import Window
    window=Window(lib,ctx,stage,rank,control,report,save)
    report['layer77_scope']='Observation only: currentqualified B16M8 I512 O-D-O, VON/TOFF/resident38/40/crossrequestOFF. No changed math or new Modelperformance claim.'
    labels=['perf-I512-O-before','prof-I512-D-window','perf-I512-O-restored']
    report['resident_plan']=[{'label':x}for x in labels];rejected=False
    try:
        for label in labels:
            generate(label,i512_inputs,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,label=='prof-I512-D-window')
    except CandidateRejected as exc:
        failed=report.get('last_completed_generation')
        if failed:report.setdefault('rejected_generation_records',{})[failed['phase']]=failed
        rejected=True;report.update(resident_candidate_rejected=True,normal_numeric_rejection=True,resident_rejection_detail=str(exc));save()
    finally:
        public(1);resident.apply('final-O-before-B1',False,False);policy.switch('final-O-before-B1',0,False)
        assert chat_gate('after_select'),'finalOFF frozenB1'
        public(0);resident.apply('final-O-empty',False,False);resident.check(resident.vector_OFF())
        final=resident.snapshot();report['resident_final_OFF_empty']=not final['any_enabled']and not final['vector_mode']and not final['transient']['enabled']and not final['transient']['busy']and not final['transient']['poison']
        report['resident_candidate_quality_passed']=not rejected;report['cohort_phases_completed']=not rejected
        report['skipped_rejected_candidate_phases']=[x for x in labels if x not in[c['phase']for c in report['multi_batch_cases']]]
        report['planned_phase_accounting_completed']=len(report['multi_batch_cases'])+len(report['skipped_rejected_candidate_phases'])==3
        report['continuous_transport_closed']={'all_cases_ended':True,'pending_drained_on_emitter':hub is None or(not hub.pending and hub.case is None)}
        feedback.close()
        if hub is not None:hub.close()
        save()
'''
 if n=='GLM_transient_preengine_r5799.py':
  s=s.replace("'glm_layer77_policy_r5799.py']:","'glm_layer77_policy_r5799.py','glm_window_control_r5799.py']:")
  # Exact actual DSO resolve+SHA logic already present; do not add basename filters.
 if n=='glm_transient_Model_r5799_inside.sh':s=s.replace('3450 python','2250 python')
 if n=='audit_GLM_crossrequest_Model_r5799.py':
  a=s.index("   if q.get('resident_candidate_quality_passed'):");b=s.index("   cases=q['multi_batch_cases']",a);s=s[:a]+s[b:]
  s=s.replace('len(plan)==10','len(plan)==3')
  a=s.index(" d['ABBA']={};perf=");b=s.index(" d['candidate_rejected']",a)
  s=s[:a]+''' d['ABBA']={};perf=[x for x in d['owners'][8]['phases']if x['phase'].startswith('perf-')]
 d['window_samples']={}
 for stage in [0,1]:
  for rank in range(8):
   q=json.loads((r/'results'/f'glm-transient-Model-r5799-{run}-stage{stage}-rank{rank}.json').read_text());samples=q.get('selected_decode_window_r5799',[])
   if not rejected:
    assert len(samples)==(6 if stage==1 else 0)
    if stage==1:assert sorted((x['round'],x['cohort'])for x in samples)==[(s,c)for s in [63,64,65]for c in [0,1]]
   for x in samples:
    assert x['layer']==38 and x['stage']==1 and x['rank']==rank and x['past']==512+x['round']-1 and x['phase']=='prof-I512-D-window'
    assert 2<=x['marker_count']<=512 and len(x['intervals'])==x['marker_count']-1
    for k in ['whole_stream_timeline_ms','whole_host_wall_ms']:assert math.isfinite(x[k])and x[k]>=0
    assert math.isclose(sum(z['stream_timeline_ms']for z in x['intervals']),x['whole_stream_timeline_ms'],rel_tol=.005,abs_tol=.01)
   d['window_samples'][f'{stage}:{rank}']=samples
'''+s[b:]
  s=s.replace('len(perf)==4','len(perf)==2').replace(" d['scope']='Same image5707", " d['scope']='Observation O-D-O; Dwall is instrumented, not optimized throughput. Same image5707")
 write(n,s)
source={n.replace('r5783','r5799'):sha(r/n.replace('r5783','r5799'))for n in names};source['glm_window_control_r5799.py']=sha(r/'glm_window_control_r5799.py')
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r5783.json').read_text());m['model_SHA256']=newhash;m['source_SHA256']=source;m['DSO_SHA256']={k.replace('glm-layer77-model-r5775/libglm_layer77_model_r5775.so','glm-window-model-r5797/libglm_window_model_r5797.so'):newhash if v==oldhash else v for k,v in m['DSO_SHA256'].items()};m['scope']='Single required O-D-O attribution load, fixedoriginalmath/currentruntime, layer38 only rounds63..65/cohort0,1.'
write('GLM-CROSSREQUEST-Model-source-manifest-r5799.json',json.dumps(m,indent=2))
side=json.loads((r/'GLM-transient-Model-observer-manifest-r5783.json').read_text());side['source_SHA256']={n.replace('r5783','r5799'):sha(r/n.replace('r5783','r5799'))for n in side['source_SHA256']};write('GLM-transient-Model-observer-manifest-r5799.json',json.dumps(side,indent=2))
s=replace((r/'launch_GLM_crossrequest_Model_r5783.py').read_text()).replace('01:00:00','00:40:00').replace('layer77-5783-','window-5799-').replace('f8ec85c37bd3fcec6e0be6fa4b6596660ebb11d8dca5377f1e70512cdbadbe8b',sha(r/'GLM-CROSSREQUEST-Model-source-manifest-r5799.json')).replace('15f0bfefa1eb62739fc0a63d2e33ba632b8d8b8a0e0a23d7b09175b26a333fcc',sha(r/'GLM-transient-Model-observer-manifest-r5799.json'))
write('launch_GLM_crossrequest_Model_r5799.py',s)
print('DRAFT_RUNTIME_AST',len(created))
