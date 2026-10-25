import pathlib,sys,ast,json,hashlib,types
r=pathlib.Path(__file__).resolve().parent;d=r/'GLM-DP-Mmatrix-private-r8868';sys.path.insert(0,str(d));m=json.loads((d/'GLM-CROSSREQUEST-Model-source-manifest-r8868.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in m['source_SHA256'].items():
 assert sha(d/n)==h,n
 if n.endswith('.py'):ast.parse((d/n).read_bytes(),filename=n)
assert m['model_SHA256']=='68bdbddc918063dee48b4da21bb3ca24a55abdc4a4a3c0d7c415e3e167de100e'and m['maximum_loads_per_replica']==1 and m['maximum_full128_arms_per_replica']==2
assert m['DP_global_rows']=={'A':list(range(16)),'B':list(range(16,32))}and m['DP_total_service_cards']==32
# Execute actual current CrossrequestControl.finish including BaseControl.finish.
actual=(r/'check_GLM564_consumer_repair_r8866.py').read_text();start=actual.index('checks=[]');end=actual.index('tree=ast.parse')
import importlib
mod=importlib.import_module('glm_crossrequest_control_r8868');base=importlib.import_module('glm_crossrequest_resident_base_r8868');env={'mod':mod,'base':base,'types':types,'copy':__import__('copy')};exec(actual[start:end],env)
tree=ast.parse((d/'glm_prefill_micro_probe_r8868.py').read_bytes());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run');plan=next(x for x in run.body if isinstance(x,ast.Try)and any(isinstance(y,ast.For)and ast.unparse(y.target)=='dp_phase'for y in x.body));code=compile(ast.fix_missing_locations(ast.Module(body=plan.body,type_ignores=[])),'actual-DP-phase-generation','exec')
for a in('A','B'):
 calls=[]
 class Feedback:
  def request(self,op,case,**kw):return {'run':True}if op=='dp_gate'else{'all32_done':True}
 rows=m['DP_global_rows'][a];prompts=[[i]*512 for i in rows];ns={'feedback':Feedback(),'replica':a,'i512_inputs':prompts,'payload':{'chat_eos_token_ids':[]},'generate':lambda *args:calls.append(args),'protocol_success':False};exec(code,ns);assert ns['protocol_success']and[(c[0],len(c[1]),c[2],c[4])for c in calls]==[(f'perf-I512-DP-{a}-jointM8-N-r0',16,8,128),(f'perf-I512-DP-{a}-jointM16-N-r0',16,16,128)]
# Same mapped frozen gold applies to both cohort geometries; no gold regeneration.
g=json.loads((d/'GLM477-frozen-gold-r8844.json').read_bytes());contract=importlib.import_module('glm_crossrequest_contract_r8868')
for a,rows in m['DP_global_rows'].items():
 mapped={**g,**{k:[g[k][i]for i in rows]for k in('input_ids','generated_ids','stopped_at_EOS')}}
 assert contract.validate_reference('I512','O',mapped['input_ids'],mapped['generated_ids'],mapped['stopped_at_EOS'],128,{},mapped['input_ids'],{},mapped)[0]
 assert len({tuple(x)for x in mapped['input_ids']})==16
out={'CPU_only':True,'GPU':False,'nativebuild':False,'passed':True,'total_requests_fixed':32,'replicas':2,'requests_per_replica':16,'phases':['jointM8','jointM16'],'actual_phase_generate_micro_and_full128':True,'actual_current_finish_super_and_override_checks':env['checks'],'mapped477_fullgold_positive':True,'B16M16_native_support':'Same CPU210/qualified567 DSO68bdbd; exact5555 backend1B16 and vectorroutes128, unchangedperquerymath. B16 cohort16 supported; actualdualB16M16 Modelquality/perf stillpending','DSO_SHA256':m['model_SHA256'],'manifest_SHA256':sha(d/'GLM-CROSSREQUEST-Model-source-manifest-r8868.json'),'entry_SHA256':sha(d/'dispatch_GLM_resident_exact_Model_r8868.py'),'budget':m['diagnostic_budget'],'GPU_GO':'NOT_REQUESTED_OR_GRANTED'}
(r/'GLM-DP-Mmatrix-CPUchecks-r8868.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(out))
