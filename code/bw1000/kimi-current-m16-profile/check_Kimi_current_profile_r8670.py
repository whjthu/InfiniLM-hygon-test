from pathlib import Path
import ast,hashlib,json,copy,textwrap
r=Path('.')
p=r/'kimi_current_profile_protocol_r8670.py';s=p.read_text().replace("'start_pasts':list(pasts)","'start_pasts':[pasts[i]for i in rows]");p.write_text(s,encoding='utf-8',newline='\n')
p=r/'audit_Kimi_current_profile_r8670.py';s=p.read_text(encoding='utf-8-sig').replace("t['start_pasts']==[544]*64","t['start_pasts']==[544]*16");p.write_text(s,encoding='utf-8',newline='\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['modern_Kimi_current_profile_r8670.py','kimi_current_profile_protocol_r8670.py','Kimi_current_profile_r8670_inside.sh','launch_Kimi_current_profile_r8670.py','audit_Kimi_current_profile_r8670.py','observe_Kimi_current_profile_r8670.py']
for n in files:
 p=r/n
 if n.endswith('.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
 assert b'\r'not in p.read_bytes(),n
checks=[]
def ok(name,value):assert value,name;checks.append(name)
p=r/'kimi_current_profile_protocol_r8670.py';src=p.read_text();tree=ast.parse(src)
# Execute actual marker callback unchanged inside a small CPU owner-state closure.
callback=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='owner_forward')
class Profiler:
 def __init__(self):self.calls=[]
 def hipProfilerStart(self):self.calls.append('start');return 0
 def hipProfilerStop(self):self.calls.append('stop');return 0
factory='''def make(tracing):
 trace_active=False;trace_window=None
 report={'profiling_windows':[]};profiler=Profiler();label='B64-M16-hiptrace36';B=64;M=16;pasts=[544]*64;ids=[[7]for _ in range(64)];token_only=True
 def forward(*a,**k):return None,0.001,None
'''+textwrap.indent(ast.unparse(callback),' ')+'\n return owner_forward,report,profiler\n'
ns={'Profiler':Profiler,'time':__import__('time')};exec(factory,ns)
f,q,pr=ns['make'](True)
for step in [31,32,33,34]:
 for cohort in range(4):f(list(range(cohort*16,(cohort+1)*16)),step,cohort)
ok('actual_callback_exactly_one_start_one_stop',pr.calls==['start','stop'])
w=q['profiling_windows'][0];ok('actual_callback_window_32first_to33last',(w['start_decode_step0'],w['start_cohort'],w['stop_decode_step0'],w['stop_cohort'])==(32,0,33,3));ok('actual_callback_16_active_pasts544',w['start_pasts']==[544]*16)
f,q,pr=ns['make'](False)
for step in range(35):
 for cohort in range(4):f(list(range(cohort*16,(cohort+1)*16)),step,cohort)
ok('ordinary_callback_never_StartStop',not pr.calls and not q['profiling_windows'])
# Actual source counter branch with N16 policy1 routes to grouped4 field0, not fallback.
forward=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward')
branch=next(n for n in forward.body if isinstance(n,ast.If)and ast.unparse(n.test)=='policy and 8 <= N <= 64')
for policy,N,want in [(1,16,[23,368,0,0]),(1,64,[0,0,23,1472]),(0,16,[0]*4),(1,1,[0]*4)]:
 v=dict(policy=policy,N=N,moe=23,expected=[0]*4);exec(compile(ast.Module(body=[branch],type_ignores=[]),'actual_counter','exec'),v);ok('actual_counter_policy%s_N%s'%(policy,N),v['expected']==want)
# Actual user-clock block preserves prefill first-token stamps and EOS filtering.
a=src.index("row['instrumented_baseline']=True");b=src.index("report['serving_cases'].append(row);save()",a)
clockblock=textwrap.dedent('  '+src[a:b]);v=dict(row={'per_request_decode_tok_s':[]},canonical={'ttft':[3.,5.],'decode_anchor_from_start_seconds':10.},times=[[0.,1.,2.],[0.,3.,4.]],B=2,ids=[[7,8,99],[9,10,11]],eos={99});exec(clockblock,v)
ok('actual_user_clock_first_not_common_anchor',v['row']['token_ready_times_from_phase_start']==[[3.,11.,12.],[5.,13.,14.]])
ok('actual_user_clock_EOS_removed',v['row']['per_request_first_last_tok_s']==[1/8,2/9])
ok('actual_user_clock_ITL_retains_prefill_gap',v['row']['per_request_max_ITL_seconds']==[8.,8.])
main=(r/files[0]).read_text();ok('constructor_capacity64',"head.encode(),port,64)"in main)
ok('exact_sixfield_geometry',"{'max_rows':64,'pages':260,'page_size':256,'KDA_pool_rows':65,'pages_per_request':4,'max_total_tokens':1024}"in main)
ok('existing_DSO_binding_reused',"from Kimi_fixed512_PP_binding_r8626 import load"in main)
ok('historical_gold_before_ordinary',src.index("'B64-M16-full-reference128'")<src.index("'B64-M16-ordinary'"))
ok('policy1_before_B64_reference',"select(1,'current-serving-policy1')"in src)
ok('QA_and_restore_policy0',"qa(0,'before')"in src and"qa(0,'after');select(0,'done')"in src)
ok('trace_only36tokens',"'B64-M16-hiptrace36',36"in src)
ok('reserve4GiB_and_runtime512MiB',"minimum>4*1024**3"in src and"512*1024**2"in src)
ok('fullgold_64x128',any(v['B']==64 and v['M']==16 and len(v['fixed_complete_IDs'])==64 and all(len(z)==128 for z in v['fixed_complete_IDs'])for v in json.loads((r/'r4917-Kimi-two-geometry-frozen-gold.json').read_text())['geometries']))
launcher=(r/files[3]).read_text();ok('watchready_before_numeric',launcher.index("ready=json.loads(wp.read_text())")<launcher.index("p=subprocess.Popen(a,"))
ok('unique_intent_exclusive',"record.open('x')"in launcher and"wp.open('x')"in launcher)
ok('transient_accounting_errors_retry_same_handle',"except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)"in launcher)
ok('terminal_post_before_audit',launcher.index("w['post_KFD']=idle")<launcher.index("audit_Kimi_current_profile_r8670.py"))
receipt=dict(revision=8670,status='CPU_SOURCE_CONTRACT_PASSED_GPU_NOT_RUN',checks=checks,count=len(checks),source_SHA256={n:sha(r/n)for n in files},native='UNCHANGED f6f10c2b; no compile/no dlopen/no GPU',limitations=['CPU fixtures do not prove current Model finalIDs or kernel traces','hipprof process interposer retained during ordinary active-trace-OFF','B32 deferred per root priority; no AUX/route histogram hook'])
(r/'Kimi-current-profile-CPU-r8670.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8',newline='\n')
old=json.loads((r/'Kimi-fixed512-PP-manifest-r8626.json').read_text())
deps=['stage_control_r2386.py','model_affinity_check_r2628.py','r5034-Kimi-packed-preparation.json','resource_wave_local_r3097_outer.sh','Kimi_memory_domains_r8152.py','Kimi-head-Model-manifest-r8235.json','Kimi_Q0_idle_r8066.py','Kimi_load_cgroup_probe_r8470.py','r4917-Kimi-two-geometry-frozen-gold.json','kimi_cohort_feedback_idle_r4915.py','Kimi_fixed512_PP_binding_r8626.py']
for n in deps:assert (r/n).is_file(),n
manifest=dict(revision=8670,status='CPU_READY_AWAIT_ROOT_GPU_GO',nodes=old['nodes'],runtime_files=files,source_sha256={n:sha(r/n)for n in files},existing_dependency_sha256={n:sha(r/n)for n in deps},remote_evidence_sha256=old['remote_evidence_sha256'],native_DSO_SHA256=old['native_DSO_SHA256'],CPU_receipt_SHA256=old['CPU_receipt_SHA256'],CPU_source_contract_SHA256=sha(r/'Kimi-current-profile-CPU-r8670.json'),scope='Current183f6f TP8PP4/B64M16; cap64/pool65/pages260/workspace64; prefillM2; SpecMath0/CCL0/proj0, servingpolicy1 grouped4; QA0, full540128gold, ordinary2 active-traceOFF, separate trace36 decode32..33, QA0/OFF restore. B32 deferred.',ordinary_clock='Instrumented baseline only: process hipprof interposer present; no unwrapped636/540 gain. Emitter first-prefilldecision to lastusefultoken is user first-last; separate common-anchor puredecode. Not client clock.',trace_scope='One HIPtrace window per node before firstcohort step32 through lastcohort step33, no additional sync/barrier; GPU interval union required, never sum ranks as criticalpath.',time_budget_minutes=dict(loading=[12,18],first_reference=[18,28],ordinary_and_trace=[6,12],QA_and_cleanup=[4,10],hard=60),GPU='NOT_AUTHORIZED_NOT_RUN',B32='DEFERRED per root prioritization')
(r/'Kimi-current-profile-manifest-r8670.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(dict(checks=len(checks),manifest_SHA256=sha(r/'Kimi-current-profile-manifest-r8670.json'),source_files=files)))
