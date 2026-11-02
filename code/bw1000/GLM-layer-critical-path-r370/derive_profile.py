"""Minimal immutable consumer derivation; original367 hardware/loader/cleanup retained."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;B=D.parent/'GLM-smallM-FFN-feasibility-r367';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=(B/'GPU_run.py').read_text();assert sha(B/'GPU_run.py')=='1f68a3e97d074e0ea3d1b02ddfc434236884467c4608aecd7cc8b9ff9953990b'
original_loader=s[s.index('  keys=json.loads'):s.index('  vote=torch.ones')]
s=s.replace('"""Whole real-checkpoint layer38 TP8 M1 component, synthetic state; no Model claim."""','"""Existing038d/5670 profile bool only; syntheticprefix512/allvisible, not Model criticalpath."""')
s=s.replace('GLM-smallM-FFN-feasibility-r367','GLM-layer-critical-path-r370').replace('GLM367','GLM370')
begin=s.index('def verify():');end=s.index('def api(path):')
s=s[:begin]+'''def verify():
 m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes());assert sha(D/'GPU_SOURCE_MANIFEST.json')==os.environ['GLM370_GPU_SOURCE_SHA256']
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 p=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(p)==m['Native_CPU_receipt_SHA256'];a=json.loads(p.read_bytes());assert a['completed']and a['CPU_only']and not a['dlopen']and not a['GPU_initialized']
 assert a['library_SHA256']==m['Native_library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and sha(a['library'])==a['library_SHA256']
 for p,h in m['dependency_SHA256'].items():assert sha(p)==h,p
 return dict(library=a['library'],library_SHA256=a['library_SHA256']),m
'''+s[end:]
s=s.replace("from plan import FIXTURES,records","from plan import FIXTURES,records\nfrom profile_contract import validate_profile")
s=s.replace(",'modern_glm_M1_gate_policy_r367'",'')
s='\n'.join(line for line in s.split('\n')if'lib.modern_glm_M1_gate_get_r367.argtypes'not in line)
s=s.replace(";assert lib.modern_glm_M1_gate_get_r367(rank)==0",'')
s=s.replace("'modern_glm_M1_gate_policy_r367',", "'glm_layer_mark_r5670',")
s=s.replace("candidate_library_SHA256=a['library_SHA256']","Native_library_SHA256=a['library_SHA256']")
s=s.replace("baseline='Same current038d objects and SDK, defaultOFF original9284 dispatch; only M1 gate/up provider changes'","baseline='Direct original038d; vector and down-row2 OFF; only5670 profile bool changes'")
s=s.replace("profile=0,measured_wall=", "profile_mode='Mixed0/1 originalhook',measured_wall=")
s=s.replace("performance_gate=m['performance_gate']", "profile_contract=m['profile_contract']")
s=s.replace("check(lib.modern_glm_M1_gate_policy_r367(rank,0));",'')
s=s.replace("check(lib.modern_glm_M1_gate_policy_r367(rank,mode));before=stats('modern_glm_M1_gate_counts_r367',4);",'')
s=s.replace("ptr(decode_pos),ptr(out),0));torch.cuda.synchronize(rank)","ptr(decode_pos),ptr(out),mode));torch.cuda.synchronize(rank)")
s=s.replace("after=stats('modern_glm_M1_gate_counts_r367',4);",'')
s=s.replace("delta=[v-u for u,v in zip(before,after)];",'')
s=s.replace("assert report['N']==report['B']==1 and report['profile']==0 and not report['stages']and report.get('layers',[report.get('layer')])==[38]","validate_profile(report,mode)")
s=s.replace("assert delta==[mode,1-mode,0,0]and vd==[mode,1-mode,0,1,0,mode,0,0],(delta,vd)","assert vd==[0,1,0,1,0,0,0,0],vd")
s=s.replace("profile=0,route_U=", "profile=mode,stages=report['stages'],lease_peak_logical_bytes=report['lease_peak_logical_bytes'],route_U=")
s=s.replace("M1_counts_delta=delta,",'')
s=s.replace("measurement_calls']==48 and r['warm_calls']==4", "measurement_calls']==12 and r['warm_calls']==2")
s=s.replace("assert q['input_fixtures'][0]['decode_SHA256']!=q['input_fixtures'][1]['decode_SHA256'];", "assert len(q['input_fixtures'])==1;")
assert'modern_glm_M1_gate'not in s
assert original_loader==s[s.index('  keys=json.loads'):s.index('  vote=torch.ones')]
compile(s,'GPU_run.py','exec');(D/'GPU_run.py').write_bytes(s.encode())
for name in ['GPU_inside.sh','resource_probe.py','controller.py']:
 text=(B/name).read_text().replace('GLM-smallM-FFN-feasibility-r367','GLM-layer-critical-path-r370').replace('GLM-r367-M1-gate-layer','GLM-r370-layer-profile').replace('GLM367','GLM370')
 if name=='controller.py':
  text=text.replace("'candidates/GLM-smallM-FFN-r367/CPU-build.json'","'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json'").replace("m['CPU_receipt_SHA256']","m['Native_CPU_receipt_SHA256']")
  text=text.replace("a['measurement_calls']==48 and a['warm_calls']==4","a['measurement_calls']==12 and a['warm_calls']==2")
  text=text.replace("performance_audit_required=True","profile_overhead_audit_required=True,optimization_gain_claim=False")
  text=text.replace("profile=0,no_retry=True","profile='Mixed0/1 existinghook',no_retry=True")
 if name.endswith('.py'):compile(text,name,'exec')
 (D/name).write_bytes(text.encode())
q=dict(original_worker_SHA256=sha(B/'GPU_run.py'),original_controller_SHA256=sha(B/'controller.py'),original_GPU_inside_SHA256=sha(B/'GPU_inside.sh'),unchanged_actual_checkpoint_loader_block_SHA256=hashlib.sha256(original_loader.encode()).hexdigest(),checkpoint_loader_byteidentical=True,ownership_cleanup_PCImapping_budget_and_failstop_reused=True,no_new_CPP_or_GPU_TU=True,Native='Direct038d;367candidate/privateM1bitsnotused',changes=['Only existing profilebool0/1 on M1decode;prefix0','Full original stage event/host/CPU triples retained','One fixed existingfixture and2warm+ABBA3rep,12measure','Manifest NativeCPU184b/SDK/object pins;3class ownership unchanged','Source/run labels private370'])
(D/'DERIVATION.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print('Derived370 from immutable367; actual checkpointloader exact, direct038d, no newNative')
