import pathlib,json,hashlib,ast
r=pathlib.Path('.');ver='8708';old='fe179cf20aba957b2b6dbd53349dd10dbf650ef9c2ef7a0e70fb843cfc36ef6f';new='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
files=['glm_prefill_micro_probe','modern_glm_prefill_micro','GLM_transient_preengine','GLM_transient_preflight','launch_GLM_prefill_Model','watch_GLM_prefill_Model_terminal','audit_GLM_prefill_Model']
for prefix in files:
 s=(r/(prefix+'_r8701.py')).read_text().replace('8701',ver).replace(old,new).replace('candidates/glm-prefill-window-r8708/libglm_prefill_window_r8708.so','candidates/glm-prefix-bridge-r8705/libglm_prefix_bridge_r8705.so')
 if prefix=='glm_prefill_micro_probe':
  a=s.index('    lib.modern_glm_window_prepare');b=s.index('    gold_path',a);s=s[:a]+'''    lib.modern_glm_prefill_prefix_plan_counts_r8703.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.modern_glm_prefill_prefix_plan_counts_r8703.restype=ctypes.c_uint64
    def prefix_stats():return [int(lib.modern_glm_prefill_prefix_plan_counts_r8703(rank,k))for k in range(3)]
'''+s[b:]
  s=s.replace('                arm_window(label,step,group)\n','').replace('                read_window(label,step,group)\n','')
  needle='        native_before=native_stats();';s=s.replace(needle,"        proof_on=int('-N-'in label);os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']=str(proof_on)\n        prefix_before=prefix_stats()\n"+needle)
  s=s.replace("        row = {'phase': label,", "        row = {'prefix_plan_enabled':proof_on,'prefix_plan_counter_delta':[b-a for a,b in zip(prefix_before,prefix_stats())],'phase': label,")
  needle="                if stage == PP-1: assert values == [saved_ids[i] for i in active]\n"
  s=s.replace(needle,needle+'''                if reference_kind=='I512'and step==0:
                    expected=[historical_i512['generated_ids'][i][0]for i in active]
                    if values!=expected:
                        report['first_token_failure']={'phase':label,'rows':active,'actual':values,'expected':expected};save();raise CandidateRejected('first_token_gold_mismatch')
''')
  a=s.index('    labels=');b=s.index('    except CandidateRejected',a)
  s=s[:a]+'''    labels=['perf-I512-proof-O-r0','perf-I512-proof-N-r0','anchor-short-N-K4'];report['resident_plan']=[{'label':x}for x in labels];rejected=False
    report['performance_adoption']=False;report['ordinary_profileOFF']=True
    try:
        epoch=time.monotonic();walls=[]
        for label in labels[:2]:
            tick=time.monotonic();generate(label,i512_inputs,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8);walls.append(time.monotonic()-tick)
        remaining=report['remaining_process_seconds_before_cohorts']-(time.monotonic()-epoch);need=2*max(walls)+240
        vote=torch.tensor([float(remaining>need)],device='cuda');control(vote)
        repeat=vote.item()==16;report['optional_cycle_budget']={'actual_remaining_seconds':remaining,'required_seconds':need,'two_more_full128':repeat}
        if repeat:
            extra=['perf-I512-proof-N-r1','perf-I512-proof-O-r1'];labels[2:2]=extra;report['resident_plan']=[{'label':x}for x in labels];save()
            for label in extra:generate(label,i512_inputs,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)
        os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']='0'
        generate(labels[-1],unique_inputs,8,True,4,set(payload['chat_eos_token_ids']),False,1,True,False,True,'short',0,False,-1,True,8)
'''+s[b:]
  s=s.replace('    finally:\n        public(1);',"    finally:\n        os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']='0'\n        public(1);")
  s=s.replace("        assert chat_gate('after_select'),'candidate shared proof frozenB1'","        if not rejected:assert chat_gate('after_select'),'candidate shared proof frozenB1'")
 if prefix=='modern_glm_prefill_micro':
  s=s.replace("ns = {}", "LIBRARY=pathlib.Path(json.loads((r/'candidates/glm-prefix-bridge-r8705/CPU-build.json').read_text())['library'])\nns = {}")
  a=s.index("exec(baseline[:");b=s.index('\nlib, ptr',a);s=s[:a]+'''exec(baseline[:baseline.index('def worker(rank,payload):')].replace("r/'build/libmodern_glm_full_model_probe.so'",repr(str(LIBRARY))),ns)'''+s[b:]
  s=s.replace("(r/'candidates/glm-prefix-bridge-r8705/libglm_prefix_bridge_r8705.so')",'LIBRARY').replace("(r / 'build/../candidates/glm-prefix-bridge-r8705/libglm_prefix_bridge_r8705.so')",'LIBRARY')
 if prefix=='GLM_transient_preflight':
  a=s.index(" receipt=r/");b=s.index(" assert (m['TP']",a);s=s[:a]+''' receipt=r/'candidates/glm-prefix-bridge-r8705/CPU-build.json';proof=json.loads(receipt.read_text());assert proof['completed']and proof['library_SHA256']==m['model_SHA256']and len(proof['reused_objects_SHA256'])==27
 assert hashlib.sha256(pathlib.Path(proof['library']).read_bytes()).hexdigest()==m['model_SHA256']
'''+s[b:]
 if prefix=='GLM_transient_preengine':
  a=s.index(" private=r/");b=s.index(';symbols=set()',a);s=s[:a]+" private=pathlib.Path(json.loads((r/'candidates/glm-prefix-bridge-r8705/CPU-build.json').read_text())['library']);assert sha(private)=='"+new+"';lib=ctypes.CDLL(str(private))"+s[b:]
 if prefix=='launch_GLM_prefill_Model':
  s=s.replace("run='prefill-5845-'","run='prefix-8708-'").replace("'--gres=dcu:8','--mem=1000M'","'--gres=none','--mem=1000M'")
 if prefix=='audit_GLM_prefill_Model':
  s=s.replace("len(cases)+len(skipped)==len(plan)==3","len(cases)+len(skipped)==len(plan)and len(plan)in(3,5)")
  a=s.index(" d['prefill_windows']=[]");b=s.index(' return d',a)
  s=s[:a]+''' d['ABBA']={};d['candidate_rejected']=rejected;d['selected_counts']=chosen
 tailcases=[c for c in tail['multi_batch_cases']if c['phase'].startswith('perf-I512-proof-')]
 d['tail_ordinary_cases']=tailcases
 for stage in (0,1):
  for rank in range(8):
   owner=json.loads((r/'results'/f'glm-transient-Model-r8708-{run}-stage{stage}-rank{rank}.json').read_text());assert not owner.get('native_pipeline_profile_samples_r8219')
   for c in owner['multi_batch_cases']:
    on=int('-N-'in c['phase']);assert c['prefix_plan_enabled']==on
    delta=c['prefix_plan_counter_delta'];assert (delta[0]>0 and delta[1]>0)if on else delta==[0,0,0]
 d['qualified']=jobs[0][1:3]==['COMPLETED','0:0']and not rejected and len(tailcases)in(2,4)
 d['performance_adopted']=False;d['scope']='Same-load fullModel ordinary proofOFF/ON B16M8 I512O128; optionalON/OFF repeat byactualbudget; originalfullgold/QA/CPUargmax/normalclosure. No eventwindow/profile.'
'''+s[b:]
 ast.parse(s);(r/(prefix+'_r'+ver+'.py')).write_text(s,encoding='utf-8',newline='\n')
s=(r/'glm_prefill_micro_Model_r8701_inside.sh').read_text().replace('8701','8708');(r/'glm_prefill_micro_Model_r8708_inside.sh').write_text(s,encoding='utf-8',newline='\n')
print('8708 AST ready')
