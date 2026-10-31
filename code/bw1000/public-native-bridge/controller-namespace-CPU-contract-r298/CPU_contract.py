"""CPU-only scope, immutable source and terminal-branch contracts; no subprocess."""
from pathlib import Path
import sys,importlib.util,json,hashlib,ast,copy,datetime
B=Path(sys.argv[1]);remote='--remote'in sys.argv;out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'srun':0,'execution_GO_file_created':False,'synthetic_scope_flag_not_human_authorization':True,'models':{}}
def load(p):
 sys.path.insert(0,str(p.parent))
 spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sys.path.pop(0);return m
for model,old,new in [('GLM','292','296'),('Deep','283','297')]:
 d=B/(model+'-latest-controller-r'+new);manifest=json.loads((d/'SOURCE_MANIFEST.json').read_bytes())
 assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in manifest['files_SHA256'].items())
 m=load(d/('controller_r'+new+'.py'));go=json.loads((d/'execution-GO.template.json').read_bytes())
 try:m.validate_scope(go)
 except PermissionError:pass
 else:raise AssertionError('Unapproved template accepted')
 go.update(GPU_authorized=True,publication_commit='942115403d6d7d5c64c38705146313f36b25bae2',controller_manifest_SHA256=hashlib.sha256((d/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
 assert m.validate_scope(go) is go
 mutations={'cards':1,'context':1024,'GPU_max_s':29401,'CPU_keeper_max_s':32401,'max_HTTP':1000,'max_Native':1000,'ordinary_budget':1024,'thinking_budget':1024,'GPU_parent':910384,'CPU_keeper_parent':910384,'publication_branch':'main','wrapper_SHA256':'0'*64,'namespace_CPU_receipt_SHA256':'0'*64,'no_retry_reload':False,'one_attempt':False,'TP':True,'PP':True}
 rejected=[]
 for k,v in mutations.items():
  g=copy.deepcopy(go);g[k]=v
  try:m.validate_scope(g)
  except(AssertionError,PermissionError):rejected.append(k)
  else:raise AssertionError('Scope mutation accepted '+k)
 cases=[(['j','label','COMPLETED','0:0'],False),(['j','label','FAILED','1:0'],True),(['j','label','CANCELLED+','0:15'],True),(['j','label','TIMEOUT','0:0'],True),(['j','label','COMPLETED','1:0'],True)]
 assert all(m.terminal_failure(row)==expected for row,expected in cases)
 tree=ast.parse((d/('controller_r'+new+'.py')).read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main')
 failed=[x for x in ast.walk(main)if isinstance(x,ast.If)and isinstance(x.test,ast.Call)and isinstance(x.test.func,ast.Name)and x.test.func.id=='terminal_failure']
 assert len(failed)==1 and isinstance(failed[0].body[-1],ast.Return)
 assert 'TERMINAL_GPU_FAILED_POST_CLEAN_NO_RETRY'in ast.unparse(failed[0])
 assert 'resource_wave_local_r3097_outer.sh'not in (d/('controller_r'+new+'.py')).read_text()
 if remote:
  m.verify_sources(go)
  namespace=m.verify_namespace(m.R,m.D)
 else:
  olddir=B/(model+'-latest-controller-r'+old)
  oldtree=ast.parse((olddir/('controller_r'+old+'.py')).read_text())
  def gpu_args(tree):
   choices=[n.value for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='args'for v in n.targets)and isinstance(n.value,ast.List)and any(isinstance(e,ast.Constant)and e.value=='--gres=dcu:8'for e in n.value.elts)]
   assert len(choices)==1
   return choices[0]
  a=copy.deepcopy(gpu_args(oldtree));b=copy.deepcopy(gpu_args(tree))
  assert ast.dump(a.elts[-2])==ast.dump(ast.parse("str(R/'incoming/resource_wave_local_r3097_outer.sh')",mode='eval').body)
  assert ast.dump(b.elts[-2])==ast.dump(ast.parse("str(D/'resource_wave_host_alias_outer.sh')",mode='eval').body)
  a.elts[-2]=b.elts[-2];assert ast.dump(a)==ast.dump(b),'GPU argv changed beyond explicit wrapper'
  namespace=None
 out['models'][model]={'manifest_SHA256':hashlib.sha256((d/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'members':len(manifest['files_SHA256']),'unapproved_template_rejected':True,'scope_negatives_rejected':rejected,'terminal_cases':len(cases),'known_failure_returns_before_model_audit':True,'original_GPU_argv_preserved_except_wrapper':not remote,'actual_remote_verify_sources_passed':remote,'namespace_proof':namespace}
out.update(completed=True,passed=True,qualification='Controller scope/source/static binding only; no actual Model/ABI/startup qualification')
print(json.dumps(out,indent=2))
