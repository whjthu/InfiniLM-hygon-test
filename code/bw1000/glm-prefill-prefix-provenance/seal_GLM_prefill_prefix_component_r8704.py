from pathlib import Path
p=Path('GLM_prefill_prefix_component_r8704_inside.sh');s=p.read_text().replace('1100 python -u','540 python -u');p.write_text(s,encoding='utf-8',newline='\n')
p=Path('controller_GLM_prefill_prefix_component_r8704.py');s=p.read_text(encoding='utf-8-sig').replace("'state':'watchready','numeric_attempts'","'state':'watchready','watch_ready_before_numeric':True,'numeric_attempts'").replace("assert all(x['profile']==0 for x in []);",'').replace("'-w','a02r4n14'","'-w',v['selected_node']").replace("d['node']=='a02r4n14'","d['node']==v['selected_node']")
needle="assert len(d['trials'])==10 and len(d['decode_fallback'])==2;"
repl=needle+"assert [a['mode']for a in d['trials']]==[0,1,0,1,1,0,0,1,1,0];assert all(a['proof_success_rows_tokens_skipped_D2H_bytes']==([8,4096,16809984]if a['mode']else[0,0,0])for a in d['trials']);"
assert needle in s;s=s.replace(needle,repl);p.write_text(s,encoding='utf-8',newline='\n')
p=Path('launch_GLM_prefill_prefix_component_r8704.py');s=p.read_text(encoding='utf-8-sig')
s=s.replace("i/'GLM-prefill-prefix-component-root-GO-r8704.json'","i/'GLM-prefill-prefix-root-GO-r8704.json'").replace("assert go['manifest_SHA256']==sha(mp)and go['one_finite_component_GPU_GO'];","assert go['status']=='CONTINUOUS_FINITE_COMPONENT_GO'and go['DSO_SHA256']==m['candidate_DSO_SHA256'];")
a=s.index("for line in snapshot.splitlines():");b=s.index("args=['srun'",a)
s=s[:a]+'''busy=set()
for line in snapshot.splitlines():
 f=line.split('|')
 if len(f)>5 and '.'in f[0]and f[0].split('.')[-1].isdigit()and f[2].split()[0].split('+')[0]in['RUNNING','PENDING','COMPLETING','CONFIGURING']:
  busy.update(subprocess.check_output(['scontrol','show','hostnames',f[5]],text=True,timeout=30).splitlines())
selected=None;idle_records=[]
for node in ['a02r4n14','a02r4n15']:
 if node in busy:continue
 idle=subprocess.run(['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-prefill-prefix-idle-r8704','python3',str(i/'verify_gpu_idle_r5819.py')],text=True,capture_output=True,timeout=150)
 rec={'node':node,'returncode':idle.returncode,'stdout':idle.stdout,'stderr':idle.stderr};idle_records.append(rec)
 if idle.returncode==0:
  kd=json.loads(idle.stdout)
  if kd['node']==node and kd['kfd_pids']==[]:selected=node;break
assert selected is not None,('no_idle_node_no_numeric',sorted(busy),idle_records)
'''+s[b:]
s=s.replace("'-w','a02r4n14','--gres=dcu:8','--mem=240000M','-t','00:20:00'","'-w',selected,'--gres=dcu:8','--mem=128000M','-t','00:10:00'")
s=s.replace("'state':'unique_intent_before_controller','args':args","'state':'unique_intent_before_controller','selected_node':selected,'args':args").replace("'fresh_idle':kd","'fresh_idle':kd,'idle_probe_records':idle_records")
p.write_text(s,encoding='utf-8',newline='\n')
import ast,json,hashlib
for n in ['GLM_prefill_prefix_component_r8704.py','controller_GLM_prefill_prefix_component_r8704.py','launch_GLM_prefill_prefix_component_r8704.py']:ast.parse(Path(n).read_text())
mech={'reviewed_prior_runner_SHA256':'120528aa49c00cf89363be1d71796824b6a5f3f81b40a2bdcd3db30b39ac95ef','actual_runner_SHA256':hashlib.sha256(Path('GLM_prefill_prefix_component_r8704.py').read_bytes()).hexdigest(),'root_GO_SHA256':hashlib.sha256(Path('GLM-prefill-prefix-root-GO-r8704.json').read_bytes()).hexdigest(),'authority':'root continuous finite GO mechanical_repairs; unchanged numerical algorithm/2warm+2ABBA/count1fallback','mechanical_delta':['Before load runtime DSO SHA and zero-failure blas-policy metadata checks','Persist active_call counter/timing evidence before unanimous control','On output mismatch preserve actual/reference complete BF16 output files withSHA','Controller/launcher bind sole selected fresh-idle n14 or n15;128000M/10min/inside540;post same node'], 'GPU':False}
Path('GLM-prefill-prefix-mechanical-r8704.json').write_text(json.dumps(mech,indent=2),encoding='utf-8')
names=['GLM_prefill_prefix_component_r8704.py','GLM_prefill_prefix_component_r8704_inside.sh','controller_GLM_prefill_prefix_component_r8704.py','launch_GLM_prefill_prefix_component_r8704.py','resource_wave_local_r3097_outer.sh','verify_gpu_idle_r5819.py','GLM-prefill-prefix-root-GO-r8704.json','GLM-prefill-prefix-mechanical-r8704.json']
d=json.loads(Path('GLM-CROSSREQUEST-Model-source-manifest-r8701.json').read_text());dsos={k:v for k,v in d['DSO_SHA256'].items()if 'glm-prefill-window-r8701'not in k};dsos['candidates/glm-prefill-prefix-r8703/libglm_prefill_prefix_r8703.so']='4572a7ad82a7977da19dc53d5962690c82e378834fa08411f9cd2c075fee1bd3'
for n in names:
 p=Path(n);s=p.read_text(encoding='utf-8-sig');p.write_text(s,encoding='utf-8',newline='\n')
m={'scope':'Finite component only/noModelTPS/adoption; existing checkpoint layer38 TP8 wholelayer; synthetic replicated hidden; actual prefixplan proof against same resident O reference','candidate_DSO_SHA256':dsos['candidates/glm-prefill-prefix-r8703/libglm_prefill_prefix_r8703.so'],'CPU_handle':'910407.166','CPU_receipt_SHA256':hashlib.sha256(Path('GLM-prefill-prefix-CPU-observed-r8703-a.json').read_bytes()).hexdigest(),'source_CPU_cases':4006,'GPU_scope_hit':'NOT_RUN','maximum_weight_loads_per_owner':1,'resource':{'parent':910376,'one_of_nodes':['a02r4n14','a02r4n15'],'DCU':8,'CPU':64,'mem_MiB':128000,'hard_minutes':10,'inside_seconds':540},'matrix':{'layer':38,'rows':8,'tokens_per_request':512,'constructor_rows':64,'capacity':640,'dense_mode':2,'profile':0,'warmups':[0,1],'temporal_cycles':[[0,1,1,0],[0,1,1,0]],'count1_fallback':'after each warmup/past512/sameactualhistory;irregularpartialCPUonly'},'metrics':['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms','owner_threadCPU_ms','proof_success_rows_tokens_skipped_D2H_bytes'],'fullwall_includes_new_CPU_proof_cost':True,'observer_preload':'blas-policy-r271 retained same forO/N; noeventprofile','DSO_SHA256':dsos,'sources_SHA256':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest()for n in names},'failure':'First badoutput/nonfinite/proofcounter/runtimeerror stops; saveactual/reference onmismatch. No retry/resetGPU/rebaseline. Existingdestroy once whenpossible; controllerfinallypost always evenparseerror.','watch':'single external parent/controller writesready before sole numericPopen; actualhandle boundedterminal bookkeeping; all8normal+strictaudit+same-nodeuniqueCPUonlyKFDpost'}
p=Path('GLM-prefill-prefix-component-manifest-r8704.json');p.write_text(json.dumps(m,indent=2),encoding='utf-8');print('MANIFEST',hashlib.sha256(p.read_bytes()).hexdigest());print('RUNNER',mech['actual_runner_SHA256'])
