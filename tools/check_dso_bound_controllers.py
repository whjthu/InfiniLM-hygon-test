"""CPU-only source/scope/evidence binding checks; never call a launcher or main."""
from pathlib import Path
import ast,copy,datetime,hashlib,importlib.util,json,shutil,sys,tempfile
G=Path(__file__).resolve().parents[1];B=G/'code/bw1000/public-native-bridge';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(cpu_receipt):
 cpu=json.loads(Path(cpu_receipt).read_bytes())['CPU-inside.json'];out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'Native_dlopen':0,'execution_GO_created':False,'models':{}}
 for model,new,old,owner,prior in [('GLM',302,296,299,290),('Deep',303,297,300,287)]:
  d=B/(model+'-latest-controller-r'+str(new));o=B/(model+'-latest-controller-r'+str(old));sm=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(d/n)==h for n,h in sm['files_SHA256'].items());assert sha(o/'SOURCE_MANIFEST.json')==sm['previous_controller_manifest_SHA256'];sys.path.insert(0,str(d))
  for n in sm['files_SHA256']:
   if n.endswith('.py'):compile((d/n).read_bytes(),str(d/n),'exec')
  m=module('controller_'+model,d/('controller_r'+str(new)+'.py'));h=module('DSO_'+model,d/'dso_binding.py');g=json.loads((d/'execution-GO.template.json').read_bytes())
  try:m.validate_scope(g)
  except PermissionError:pass
  else:raise AssertionError('Unapproved template accepted')
  # True only in this in-memory scope fixture, never saved as execution GO.
  fixture={**g,'GPU_authorized':True,'publication_commit':'a'*40,'controller_manifest_SHA256':sha(d/'SOURCE_MANIFEST.json')};m.validate_scope(fixture)
  bad={'GPU_authorized':False,'publication_branch':'main','owner_manifest_SHA256':'0'*64,'GPU_parent':1,'CPU_keeper_parent':1,'cards':1,'TP':1,'PP':99,'context':1024,'ordinary_budget':127,'thinking_budget':127,'max_HTTP':999,'max_Native':999,'GPU_max_s':999999,'Native_max_s':999999,'CPU_keeper_max_s':999999,'one_attempt':False,'no_retry_reload':False,'wrapper_SHA256':'0'*64,'namespace_CPU_receipt_SHA256':'0'*64,'DSO_CPU621_receipt_SHA256':'0'*64}
  for k,v in bad.items():
   try:m.validate_scope({**fixture,k:v})
   except(AssertionError,PermissionError):pass
   else:raise AssertionError('Scope accepted bad '+k)
  for row,result in [(['910376.1','x','FAILED','1:0'],True),(['910376.1','x','CANCELLED+','0:15'],True),(['910376.1','x','TIMEOUT','0:0'],True),(['910376.1','x','COMPLETED','1:0'],True),(['910376.1','x','COMPLETED','0:0'],False)]:assert m.terminal_failure(row)==result
  # Preserve the whole runtime body, including resource checks, budgets, normal markers and failure cleanup.
  def main(path):return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
  oldmain=main(o/('controller_r'+str(old)+'.py'));newmain=main(d/('controller_r'+str(new)+'.py'))
  for n in ast.walk(newmain):
   if isinstance(n,ast.Dict):
    pairs=[(k,v)for k,v in zip(n.keys,n.values)if not(isinstance(k,ast.Constant)and k.value=='DSO_CPU621_inside_SHA256')];n.keys=[k for k,v in pairs];n.values=[v for k,v in pairs]
   if isinstance(n,ast.Constant)and isinstance(n.value,str):n.value=n.value.replace('WAVE_INSIDE_R2323='+model+'-latest-full-owner-r'+str(owner)+'/','WAVE_INSIDE_R2323='+model+'-latest-full-owner-r'+str(prior)+'/')
  assert ast.dump(oldmain)==ast.dump(newmain),'Runtime logic changed beyond source selection and evidence field'
  owner_sha=sm['owner_manifest_SHA256'];h.validate_receipt(cpu,model,owner_sha)
  mutated=[]
  for key,value in [('passed',False),('completed',False),('CPU_only',False),('actual_CPU','910407.600'),('Model_qualified',True),('LD_PRELOAD_cleared_for_CPU',False),('GPU',1),('Model',1),('HTTP',1),('weights',1),('Native_dlopen',1),('runtime_imports',1)]:
   q=copy.deepcopy(cpu);q[key]=value
   try:h.validate_receipt(q,model,owner_sha)
   except AssertionError:mutated.append(key)
   else:raise AssertionError('Bad receipt accepted: '+key)
  q=copy.deepcopy(cpu);q['phases']=q['phases'][:-1]
  try:h.validate_receipt(q,model,owner_sha)
  except AssertionError:mutated.append('missing_phase')
  else:raise AssertionError('Incomplete phases accepted')
  try:h.validate_receipt(cpu,model,'0'*64)
  except AssertionError:mutated.append('wrong_owner_manifest')
  else:raise AssertionError('Wrong source accepted')
  with tempfile.TemporaryDirectory(prefix='common-dso-binding-')as tmp:
   root=Path(tmp);cp=root/'candidates/common-dso-audit-r301/CPU-inside.json';cp.parent.mkdir(parents=True);raw=(json.dumps(cpu,indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==h.CPU621_SHA;cp.write_bytes(raw)
   for n in [owner,prior]:
    src=B/(model+'-latest-full-owner-r'+str(n));dst=root/'incoming'/src.name;dst.mkdir(parents=True);q=json.loads((src/'SOURCE_MANIFEST.json').read_bytes())
    for name in ['SOURCE_MANIFEST.json',*q['files_SHA256']]:p=dst/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((src/name).read_bytes())
   proof=h.verify_dso(root,root/'incoming'/(model+'-latest-full-owner-r'+str(owner)),model)
   cp.write_bytes(raw+b' ')
   try:h.verify_dso(root,root/'incoming'/(model+'-latest-full-owner-r'+str(owner)),model)
   except AssertionError:mutated.append('receipt_bytes_tampered')
   else:raise AssertionError('Receipt SHA mismatch accepted')
  out['models'][model]={'controller_manifest_SHA256':sha(d/'SOURCE_MANIFEST.json'),'owner_manifest_SHA256':owner_sha,'source_members':len(sm['files_SHA256']),'unapproved_template_rejected':True,'scope_negative_cases':list(bad),'terminal_cases':5,'whole_runtime_main_AST_preserved_except_owner_path_and_evidence_field':True,'CPU621_binding_negative_cases':mutated,'actual_CPU621_receipt_and_original_source_compatibility_passed':True,'binding_proof':proof}
  sys.path.pop(0)
 out.update(completed=True,passed=True,qualification='Controller/source/CPU621 binding and unchanged runtime body only; no execution GO, keeper, GPU, live ABI or Model proof');return out
if __name__=='__main__':
 q=run(sys.argv[1]);p=Path(sys.argv[2]);assert not p.exists();p.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(q,ensure_ascii=False,indent=2))
