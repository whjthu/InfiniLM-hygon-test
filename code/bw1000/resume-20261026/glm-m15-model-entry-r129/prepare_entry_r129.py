import pathlib,json,hashlib,shutil,ast,difflib
o=pathlib.Path(__file__).resolve().parent;r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群');base=r/'GLM-down-row2-Model-private-r9284b';p125=o.parent/'glm-m15-oproj-model-r125';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['arrival_runner_r9258.py','arrival-trace.json','corpus-reference.json','quality_metrics_r9132b.py','glm_crossrequest_control_r9059.py','glm_crossrequest_resident_base_r9059.py','GLM-CROSSREQUEST-Model-source-manifest-r9059.json']:shutil.copyfile(base/n,o/n)
for n in ['case_plan_r125.py','case_plan_r125.json']:shutil.copyfile(p125/n,o/n)
(o/'CPU-result.json').write_text(json.dumps({'program_SHA256':'d4392103384e141481956ea8d8f5e46f9d0a38a3a4d82e213a04f33a14647210','original755_CPU_closure_reused':True,'new_native_CPU':'910407.445','new_native_CPU_SHA256':'83bd99026decb9f7b38acd4301506854288bfe2887c3265e32b3051f3813c009'},indent=2),encoding='utf8')
probe=(p125/'glm_arrival_probe_r125.py').read_text();probe=probe.replace('GLM-arrival-progress-r125-','GLM-arrival-progress-r129-').replace('R125 M15-o_proj','R129 M15-o_proj')
needle='before=stats();rag_before,hist_before=raggedstats();torch.cuda.synchronize();tick='
rep="t16_control=len(flat)==16 and not prefill;m15_t16_before=m15stats()if t16_control else None;before=stats();rag_before,hist_before=raggedstats();torch.cuda.synchronize();tick="
assert probe.count(needle)==1;probe=probe.replace(needle,rep)
needle="  pipeline_profile=[float(lib.modern_glm_pipeline_last_profile()[j])for j in range(3)];native_events.append("
rep="""  if t16_control:
   m15_t16_after=m15stats();dd=[b-a for a,b in zip(m15_t16_before,m15_t16_after)];assert dd[1]==dd[2]==dd[3]==dd[5]==dd[15]==dd[16]==0,('T16 no M15 path',stage,rank,dd)
   report.setdefault('M15_T16_same_load_controls',[]).append({'label':label,'rows':len(rows),'input_tokens':len(flat),'native_seconds':ended-tick,'M15_delta17':dd,'default_nonselection_verified':True})
  pipeline_profile=[float(lib.modern_glm_pipeline_last_profile()[j])for j in range(3)];native_events.append("""
assert probe.count(needle)==1;probe=probe.replace(needle,rep);ast.parse(probe);(o/'glm_arrival_probe_r129.py').write_text(probe,encoding='utf8')
entry=(base/'model_entry_r9284b.py').read_text().replace('r9284b','r129').replace('GLM_ARRIVAL_NODE_R9284B_NORMAL','GLM_ARRIVAL_NODE_R129_NORMAL')
needle="    initial=batch_stats3075();assert initial==[0]*5,('not defaultOFF',initial)"
addition="""    lib.modern_glm_model_m15_oproj_policy_r125.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.modern_glm_m15_oproj_stats_r125.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def m15_stats129():
        a=(ctypes.c_uint64*17)();check(lib.modern_glm_m15_oproj_stats_r125(rank,a,17));return list(a)
    initial15=m15_stats129();assert initial15==[0]*17,'M15 defaultOFF/newcacheempty'
    assert lib.modern_glm_model_m15_oproj_policy_r125(ctx,2)!=0 and m15_stats129()==initial15
    sentinel15=(ctypes.c_uint64*17)(*[77]*17);assert lib.modern_glm_m15_oproj_stats_r125(rank,sentinel15,16)!=0 and list(sentinel15)==[77]*17
    import threading
    foreign15=[];thread15=threading.Thread(target=lambda:foreign15.append(lib.modern_glm_model_m15_oproj_policy_r125(ctx,1)));thread15.start();thread15.join(20);assert not thread15.is_alive()and foreign15==[1]and m15_stats129()==initial15
    report.update(M15_default_OFF_verified=True,M15_invalid_wrongthread_untouched=True)
"""
assert entry.count(needle)==1;entry=entry.replace(needle,addition+needle)
needle="    report['host_bank_after_destroy_r4504']=host_stats()"
addition="""    report['M15_after_destroy']=m15_stats129();released15=report['M15_after_destroy'];assert released15[0]==0 and released15[7]==released15[8]and released15[12]==released15[13]==0 and released15[5]==released15[16]==0
"""
assert entry.count(needle)==1;entry=entry.replace(needle,addition+needle);ast.parse(entry);(o/'model_entry_r129.py').write_text(entry,encoding='utf8')
pre=(base/'preengine_r9284b.py').read_text().replace('preflight_r9284b','preflight_r129');needle=" for name in sorted(symbols):assert getattr(lib,name),('missing CABI',name)";rep=" symbols.update(['modern_glm_model_m15_oproj_policy_r125','modern_glm_m15_oproj_stats_r125'])\n"+needle;assert pre.count(needle)==1;pre=pre.replace(needle,rep);pre=pre.replace("('glm_ragged_policy_r9136',private)","('modern_glm_model_m15_oproj_policy_r125',private),('modern_glm_m15_oproj_stats_r125',private),('glm_ragged_policy_r9136',private)",1);ast.parse(pre);(o/'preengine_r129.py').write_text(pre,encoding='utf8')
shell=(base/'GPU_inside.sh').read_text().replace('GLM-down-row2-Model-r9284b','GLM-M15-Model-r129').replace('model_entry_r9284b.py','model_entry_r129.py');(o/'GPU_inside_r129.sh').write_text(shell,encoding='utf8')
audit=(base/'audit.py').read_text().replace('GLM-down-row2-Model-r9284b','GLM-M15-Model-r129').replace('glm-arrival-r9284b-','glm-arrival-r129-');audit=audit.replace("[(e['label'],e['chunked'],e['down_row2'])for e in arms]==[('A0-whole-row2OFF',False,False),('B0-whole-row2ON',False,True),('B1-whole-row2ON',False,True),('A1-whole-row2OFF',False,False)]","[(e['label'],e['chunked'],e['down_row2'],e['m15_oproj'])for e in arms]==[('A0-M15OFF',False,True,False),('B0-M15ON',False,True,True),('B1-M15ON',False,True,True),('A1-M15OFF',False,True,False)]")
needle="   assert len(q['warmups'])==2"
extra="""   released=q['M15_after_destroy'];assert released[0]==0 and released[7]==released[8]and released[12]==released[13]==0 and released[5]==released[16]==0
   assert q['M15_default_OFF_verified']and q['M15_invalid_wrongthread_untouched'];assert q['M15_T16_same_load_controls']and all(x['default_nonselection_verified']and all(x['M15_delta17'][i]==0 for i in [1,2,3,5,15,16])for x in q['M15_T16_same_load_controls'])
"""
assert audit.count(needle)==1;audit=audit.replace(needle,extra+needle)
needle="    vc=a['vector_counters'];"
rep="""    md=a['m15_provider'];expected=sum(q['native_events'][c['event_index']]['input_tokens']==15 for c in a['native_calls'])*[38,40][stage];dd=md['delta'];on=a['m15_oproj'];assert md['actual_owner_verified']and md['expected_eligible_projection_calls']==expected and dd[1]==expected and dd[2]==(expected if on else 0)and dd[3]==(0 if on else expected)and dd[5]==dd[16]==0 and dd[15]==dd[2]
    vc=a['vector_counters'];"""
assert audit.count(needle)==1;audit=audit.replace(needle,rep);ast.parse(audit);(o/'audit_r129.py').write_text(audit,encoding='utf8')
print('actual755 entry/preflight/preengine/probe/ABI/postdestroy/audit andGPUentry generated')
