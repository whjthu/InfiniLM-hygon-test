import pathlib,json,hashlib,shutil,ast,difflib
o=pathlib.Path(__file__).resolve().parent;r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群');base=r/'GLM-down-row2-Model-private-r9284b'
for n in ['arrival_runner_r9258.py','arrival-trace.json','corpus-reference.json','quality_metrics_r9132b.py','glm_crossrequest_control_r9059.py','glm_crossrequest_resident_base_r9059.py']:shutil.copyfile(base/n,o/n)
plan=dict(schema=1,episodes=[dict(label=l,chunked=False,down_row2=True,m15_oproj=b)for l,b in [('A0-M15OFF',False),('B0-M15ON',True),('B1-M15ON',True),('A1-M15OFF',False)]],max_live=16,chunkcap=32,warm_prompt_tokens=40,warm_output_cap=2)
(o/'case_plan_r125.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
validator='''import pathlib,json
def load():
 q=json.loads(pathlib.Path(__file__).with_name("case_plan_r125.json").read_bytes());expected=[("A0-M15OFF",False),("B0-M15ON",True),("B1-M15ON",True),("A1-M15OFF",False)]
 assert q["max_live"]==16 and q["warm_prompt_tokens"]==40 and q["warm_output_cap"]==2 and len(q["episodes"])==4
 for a,(label,on)in zip(q["episodes"],expected):assert a["label"]==label and type(a["m15_oproj"])is bool and a["m15_oproj"]is on and a["chunked"]is False and a["down_row2"]is True
 return q
'''
(o/'case_plan_r125.py').write_text(validator,encoding='utf8')
s=(base/'glm_arrival_probe_r9284b.py').read_text();s=s.replace('from case_plan_r9284b import load as load_plan','from case_plan_r125 import load as load_plan').replace('r9284b','r125')
needle=' def vectorstats():'
add=''' lib.modern_glm_model_m15_oproj_policy_r125.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.modern_glm_m15_oproj_stats_r125.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def m15stats():
  a=(ctypes.c_uint64*17)();check(lib.modern_glm_m15_oproj_stats_r125(rank,a,17));return list(a)
 assert m15stats()[0]==0,'M15 defaultOFF'
 report['M15_stats_order']=['enabled','eligible','selected','off_fallback','other_M15_rejected','errors','switches','created','destroyed','hits','misses','evictions','active','active_bytes','highwater','runs','invalid']
'''
assert s.count(needle)==1;s=s.replace(needle,add+needle)
s=s.replace('def episode(label,mode,case_requests,case_arrivals,warm=False,down_row2=False):','def episode(label,mode,case_requests,case_arrivals,warm=False,down_row2=True,m15_oproj=False):')
needle='barrier();reset(1);vector_before=vectorstats();'
rep="barrier();reset(1);check(lib.modern_glm_model_m15_oproj_policy_r125(ctx,int(m15_oproj)));barrier();m15_before=m15stats();assert m15_before[0]==int(m15_oproj);vector_before=vectorstats();"
assert s.count(needle)==1;s=s.replace(needle,rep)
needle="  if is_emitter:\n   assert data is not None;receipt['serving']=data"
rep="""  m15_after=m15stats();m15_delta=[b-a for a,b in zip(m15_before,m15_after)];eligible=sum(e['input_tokens']==15 for e in native_events[start_index:])*[38,40][stage]
  assert m15_delta[1]==eligible and m15_delta[2]==(eligible if m15_oproj else 0) and m15_delta[3]==(0 if m15_oproj else eligible) and m15_delta[5]==m15_delta[16]==0 and m15_delta[15]==m15_delta[2],('M15 actualownerbranch',stage,rank,m15_delta,eligible)
  receipt['m15_oproj']=m15_oproj;receipt['m15_provider']={'before':m15_before,'after':m15_after,'delta':m15_delta,'expected_eligible_projection_calls':eligible,'actual_owner_verified':True};receipt['private_cache_added_logical_bytes']=m15_after[13]
  if is_emitter:
   assert data is not None;receipt['serving']=data"""
assert s.count(needle)==1;s=s.replace(needle,rep)
s=s.replace("episode('warm-whole-row2OFF',False,warm_requests,[0.]*32,True,False);episode('warm-whole-row2ON',False,warm_requests,[0.]*32,True,True)","episode('warm-M15OFF',False,warm_requests,[0.]*32,True,True,False);episode('warm-M15ON',False,warm_requests,[0.]*32,True,True,True)")
s=s.replace("episode(arm['label'],arm['chunked'],requests,arrivals,False,arm['down_row2'])","episode(arm['label'],arm['chunked'],requests,arrivals,False,True,arm['m15_oproj'])")
needle="check(lib.modern_glm_model_reset(ctx));resident.apply('R9284B-final-OFF'"
rep="check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_m15_oproj_policy_r125(ctx,0));report['M15_before_destroy']=m15stats();resident.apply('R9284B-final-OFF'"
assert s.count(needle)==1;s=s.replace(needle,rep)
s=s.replace("report.update(scope=", "report.update(M15_row2_fixed_ON=True,scope=")
ast.parse(s);(o/'glm_arrival_probe_r125.py').write_text(s,encoding='utf8')
(o/'probe-delta-r125.patch').write_text(''.join(difflib.unified_diff((base/'glm_arrival_probe_r9284b.py').read_text().splitlines(True),s.splitlines(True),fromfile='qualified755probe',tofile='r125-M15-oproj-onlyprobe')),encoding='utf8')
# Integration fragment for the unchanged loader: add after original Model destroy.
(o/'model_entry_postdestroy_r125.inc.py').write_text("""# Called on each of16owner processes after originalmodern_glm_model_destroy(ctx).
released=(ctypes.c_uint64*17)();check(lib.modern_glm_m15_oproj_stats_r125(rank,released,17));report['M15_after_destroy']=list(released)
assert released[0]==0 and released[7]==released[8] and released[12]==released[13]==0 and released[5]==released[16]==0
""",encoding='utf8')
print('paired protocol/probe prepared; loader/nativeSHA binding waits for actual CPU closure')
