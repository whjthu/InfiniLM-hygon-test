import ast,hashlib,json,pathlib
r=pathlib.Path('.');d=r/'GLM-PP2-B32-Model-private-r8844';d.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cpu=json.loads((r/'GLM-PP2-B32-CPU-observed-r8843-b.json').read_text(encoding='utf-8-sig'))['receipt'];assert cpu['completed']
gold=json.loads((r/'GLM477-frozen-gold-r8844.json').read_text(encoding='utf-8-sig'))
H=cpu['library_SHA256'];G=gold['source_report_SHA256'];F=gold['input_fixture_SHA256']
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r8708.json').read_text())
mapping={n:n.replace('r8708','r8844').replace('r5742','r8844').replace('r5466','r8844').replace('r5819','r8844')for n in m['source_SHA256']}
def tr(s):
 for a,b in mapping.items():s=s.replace(a,b).replace(a[:-3],b[:-3])
 return s.replace('r8708','r8844')
def put(n,s):
 s=s.replace('\r\n','\n');ast.parse(s,filename=n)if n.endswith('.py')else None;(d/n).write_bytes(s.encode())
for old,new in mapping.items():
 if new.startswith('dispatch_'):continue
 if old=='modern_glm_prefill_micro_r8708.py':
  s=(r/'GLM-PP3-Model-private-r8800/modern_glm_prefill_micro_r8800.py').read_text().replace('r8800','r8844').replace('candidates/glm-PP3-Model-r8801','candidates/glm-PP2-B32-r8843').replace('(64,640)','(32,640)').replace('assert PP==3','assert PP==2').replace('BOUNDS=[0,26,54,78]','BOUNDS=[0,38,78]').replace("part['PP']==3","part['PP']==2")
  s=s.replace('glm-I512-real-inputs-r5713.json','glm-I512x32-real-inputs-r5732.json').replace('2f1076c4933394579097fc58fb399b53667f11cbce81a06d263aa5b4e2d83bbc',F).replace("fixture['cases']]*4","fixture['cases']]")
  s=s.replace('TP8PP3, actual26/28/24layers, ctx64rows','TP8PP2, actual38/40layers, ctx32rows; same fixed admittedmask B16thenB32')
 elif old=='glm_crossrequest_resident_base_r5742.py':s=(r/'glm_rows32_resident_base_r5738.py').read_text()
 elif old=='glm_crossrequest_control_r5742.py':s=(r/'glm_rows32_control_r5738.py').read_text().replace('glm_rows32_resident_base_r5738','glm_crossrequest_resident_base_r8844').replace('Rows32Control','CrossrequestControl')
 elif old=='glm_continuous_feedback_r5466.py':s=(r/'GLM-PP3-Model-private-r8800/glm_continuous_feedback_r8800.py').read_text().replace('cards==24','cards==16')
 else:s=tr((r/old).read_text())
 if old=='glm_crossrequest_contract_r5742.py':
  s=s.replace('len(stopped)==16','len(stopped) and len(prompts)in(16,32)').replace('len(prompts)==len(generated)==len(stopped)==16','len(prompts)==len(generated)==len(stopped) and len(prompts)in(16,32)').replace('else i512_inputs','else i512_inputs[:len(prompts)]').replace('len(q[\'cases\'])==16','len(q[\'cases\'])==32').replace("==8192","==16384").replace('))==16', '))==32').replace('list(range(16))','list(range(32))').replace('2f1076c4933394579097fc58fb399b53667f11cbce81a06d263aa5b4e2d83bbc',F)
  a=s.index(' if historical_i512 is not None:');b=s.index(" key=('I512'",a)
  s=s[:a]+''' assert historical_i512 is not None,'Frozen477fullgoldrequired_no_rebaseline'
 want=[g[:limit]for g in historical_i512['generated_ids'][:len(prompts)]]
 eos=historical_i512['stopped_at_EOS'][:len(prompts)]if limit==128 else[False]*len(prompts)
 ok=generated==want and stopped==eos
 return ok,[]if ok else[{'477_original_full_ID_or_EOS_mismatch':True}]
'''
 if old=='glm_prefill_micro_probe_r8708.py':
  s=s.replace('len(i512_inputs)==16','len(i512_inputs)==32')
  s=s.replace('report, report_path, chat_gate):','report, report_path, chat_gate,phase_marker):')
  s=s.replace("        assert prefill_micro in(4,8)and micro==8","        _first_io={'prefill':False,'decode':False}\n        assert prefill_micro in(4,8)and micro==8")
  lines=[]
  for line in s.splitlines():
   if 'check(lib.modern_glm_model_forward(ctx,'in line:
    ind=line[:len(line)-len(line.lstrip())];lines.extend([ind+"_io='prefill'if step==0 else'decode';_mark=not _first_io[_io]",ind+"if _mark:phase_marker(label+'.first_'+_io+'.enter')",line,ind+"if _mark:phase_marker(label+'.first_'+_io+'.return');_first_io[_io]=True"])
   else:lines.append(line)
  s='\n'.join(lines)+'\n'
  a=s.index("    oldpath=root/");b=s.index('    lib.modern_glm_crossrequest_prefill_calls_r7328',a)
  s=s[:a]+f"    frozen477=root/'incoming/GLM477-frozen-gold-r8844.json'\n    assert hashlib.sha256(frozen477.read_bytes()).hexdigest()=={sha(r/'GLM477-frozen-gold-r8844.json')!r}\n    historical_i512=json.loads(frozen477.read_text());assert historical_i512['input_ids']==i512_inputs\n"+s[b:]
  a=s.index("    labels=['perf-I512-proof-O-r0'");b=s.index('    except CandidateRejected',a)
  s=s[:a]+'''    labels=['perf-I512-B16-N-r0','perf-I512-B32-N-r0'];report['resident_plan']=[{'label':x}for x in labels];rejected=False
    report['performance_adoption']=False;report['ordinary_profileOFF']=True
    try:
        for label in labels:
            prompts=i512_inputs[:16] if '-B16-'in label else i512_inputs
            generate(label,prompts,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)
'''+s[b:]
  s=s.replace('resident38/40','fixedactualadmittedmask32rowbanks').replace('Same-load proofOFF/ON','Same-load B16/B32 schedulingonly')
 if old in ['GLM_transient_preflight_r8708.py','GLM_transient_preengine_r8708.py']:
  s=s.replace('candidates/glm-prefix-bridge-r8705','candidates/glm-PP2-B32-r8843').replace('5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52',H).replace("len(proof['reused_objects_SHA256'])==27","len(proof['reused_objects_SHA256'])==22").replace('(8,2,640,16,16,8)','(8,2,640,32,32,8)').replace('(128*1024**2,6*1024**3','(256*1024**2,6*1024**3').replace("GLM_CACHE_ROWS_R3627']=='16'","GLM_CACHE_ROWS_R3627']=='32'").replace('glm-I512-real-inputs-r5713.json','glm-I512x32-real-inputs-r5732.json')
 if old=='audit_GLM_prefill_Model_r8708.py':
  s=s.replace('glm-I512-real-inputs-r5713.json','glm-I512x32-real-inputs-r5732.json').replace("q['cache_rows']==16","q['cache_rows']==32").replace('2*16*640','2*32*640').replace('len(plan)in(3,5)','len(plan)==2').replace("row['batch']==16","row['batch']in(16,32)").replace('selected*128*1024**2','selected*256*1024**2').replace('selected*80*1024**2','selected*160*1024**2').replace('and chosen==[38,40]','').replace('min(i+pm,16)',"min(i+pm,row['batch'])").replace('range(0,16,pm)',"range(0,row['batch'],pm)").replace("prompts=i512 if kind=='I512'","prompts=i512[:row['batch']] if kind=='I512'").replace("startswith('perf-I512-proof-')","startswith('perf-I512-')")
  a=s.index(' oldhist=');b=s.index(" d={'run':",a)
  s=s[:a]+f" historical_i512=json.loads((r/'incoming/GLM477-frozen-gold-r8844.json').read_text());assert sha(r/'incoming/GLM477-frozen-gold-r8844.json')=={sha(r/'GLM477-frozen-gold-r8844.json')!r}\n assert sha(hostpath(historical_i512['source_report']))==historical_i512['source_report_SHA256']\n"+s[b:]
 if old=='launch_GLM_prefill_Model_r8708.py':
  s=s.replace("'GLM_CACHE_ROWS_R3627=16'","'GLM_CACHE_ROWS_R3627=32'").replace("'00:35:00'","'00:35:00'").replace("'01:00:00'","'00:35:00'")
 if old.endswith('_inside.sh'):s=s.replace('3500 python','2000 python')
 s=s.replace('a02r4n14','a02r4n12').replace('a02r4n15','a02r4n13')
 if old=='modern_glm_prefill_micro_r8708.py':s=s.replace('glm_crossrequest_environment_r5742','glm_crossrequest_environment_r8844').replace('3500-(time.monotonic()','2000-(time.monotonic()')
 if old=='glm_prefill_micro_probe_r8708.py':s=s.replace('ROWS16 TP8PP2/B16M8','ROWS32 TP8PP2/B16thenB32M8').replace('independent I512 OFF regression reference','original477 frozen I512 gold')
 if old=='launch_GLM_prefill_Model_r8708.py':
  s=s.replace("run='prefix-8708-'","run='b32-8844-'")
  a=s.index("'scope':",s.index("d={'utc':"));s=s[:a]+"'scope':'One load sameconstructor32/fixedadmittedmask B16M8 thenB32M8 full128 ordinary; currentprefix/vector; original477gold/EOS/QA/16normal/post;35minhard2000inside. No kernelarms or assumedfit/gain.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d))\n"
 if old==new:(d/new).write_bytes((r/old).read_bytes())
 else:put(new,s)
# Phase-save implementation and exactbytes dependencies copied from qualified sources.
(d/'GLM477-frozen-gold-r8844.json').write_bytes((r/'GLM477-frozen-gold-r8844.json').read_bytes())
for p in d.glob('*.py'):ast.parse(p.read_text(),filename=p.name)
observer={'source_SHA256':{n:sha(d/n)for n in ['watch_GLM_prefill_Model_terminal_r8844.py','audit_GLM_prefill_Model_r8844.py','glm_crossrequest_environment_r8844.py'if(d/'glm_crossrequest_environment_r8844.py').exists()else'glm_crossrequest_environment_r5742.py','glm_continuous_feedback_r8844.py','glm_crossrequest_contract_r8844.py','verify_gpu_idle_r8844.py'if(d/'verify_gpu_idle_r8844.py').exists()else'verify_gpu_idle_r5819.py']}}
op=d/'GLM-transient-Model-observer-manifest-r8844.json';op.write_bytes((json.dumps(observer,indent=2)+'\n').encode())
m.update(TP=8,PP=2,active=32,constructor_rows=32,resident_physical_bytes_per_layer=256*1024**2,resident_logical_bytes_per_layer=160*1024**2,runtime_floor_bytes=512*1024**2,model_SHA256=H,CPU_native_receipt={'path':'GLM-PP2-B32-CPU-observed-r8843-b.json','SHA256':sha(r/'GLM-PP2-B32-CPU-observed-r8843-b.json')},CPU_contracts={'actualCPU':'910407.198 COMPLETED0/51s 6hostTU22reuse','currentMD_all6_private_policy_consumers':True,'defaultnewkernels':'tile2/quantreuse absent','ordinary_profileOFF':True},fixed_ordinary_arm_count=2,maximum_loads=1,maximum_full128_arms=2,perf_labels=['perf-I512-B16-N-r0','perf-I512-B32-N-r0'],observer_manifest_SHA256=sha(op),all_policies_fixed='PVOFF/VON/TOFF/crossOFF/sharedON/prefixON; oneadmittedfixedmask',math='Current522prefix/vector math; onlyconstructor32/active32 metadata fromqualified477, M8unchanged',scope_hit='B16 first16 subset of original47732distinct512inputs; full128IDs/EOS; no rebaseline',diagnostic_budget='35minhard2000inside one loadtwoarms+QA. Historical27.2min estimate only; unknownpeak/fit no automaticretry.',input_provenance={'fixture':gold['input_fixture'],'SHA256':F,'distinct':32,'tokens_each':512,'padding':False,'B16':'first16subset','gold477_source_SHA256':G})
m['DSO_SHA256']={k:v for k,v in m['DSO_SHA256'].items()if'glm-prefix-bridge-r8705'not in k};m['DSO_SHA256'][cpu['library'].replace('/refactor/','',1)]=H
m['source_SHA256']={p.name:sha(p)for p in d.iterdir()if p.suffix in('.py','.sh','.json')and not p.name.startswith(('dispatch_','GLM-CROSSREQUEST-','review-package'))}
mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8844.json';mp.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode());mh=sha(mp)
put('dispatch_GLM_PP2_B32_Model_r8844.py',"import pathlib,sys\nsys.argv=['launch_GLM_prefill_Model_r8844.py',%r]\nexec(compile((pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/launch_GLM_prefill_Model_r8844.py').read_text(),'launch_GLM_prefill_Model_r8844.py','exec'))\n"%mh)
review={'status':'CPU_READY_PENDING_ROOT_MODEL_GO','GPU_submitted':False,'CPU':'910407.198 COMPLETED0/51s','DSO_SHA256':H,'manifest_SHA256':mh,'entry_SHA256':sha(d/'dispatch_GLM_PP2_B32_Model_r8844.py'),'resources':'910376 n12+n13 16DCU c64/240000M each 35minhard2000inner','arms':m['perf_labels'],'fixed_bank':'same constructor32/same actualfirstadmissionfixedmask, notbest52216bankbaseline','guard':'6GiB+64MiB admission/512MiB runtimefloor; noeviction/drain/owner/READY EMPTY','gold':'original477full1284armsidentical; B16subset32uniqueI512; max128EOSsame, failureSTOP no rebaseline','metrics':'prefill/decode aggregate/card/E2E distinct; realemitter perrequestfirstlast/TTFT/ITLP95/P99/maxgap','memory':'weights/currentbankallocation actualadmission required; scratch/allocatorpeakUNKNOWN; noOOMretry','budget':'27.2min historicalestimate notguarantee; one loadtwoarmsQA only','remaining':'CPUcontracttests/staticconstantclosure/sourceclaimpush/rootREADYGO thenfreshadmission uniquewatchentry'}
(d/'review-package-r8844.json').write_bytes((json.dumps(review,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(review,ensure_ascii=False))
