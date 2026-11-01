"""CPU scope/provenance checks for the independently scoped Deep68 controller. No submit."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,shutil,sys,tempfile
G=Path(__file__).resolve().parents[1];B=G/'code/bw1000/public-native-bridge';D=B/'Deep-semantic68-controller-r322'
sys.path.insert(0,str(D))
spec=importlib.util.spec_from_file_location('semantic68_controller',D/'controller_r322.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from supplement_binding import validate_receipt,verify_supplement
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rejected(fn):
 try:fn()
 except (AssertionError,PermissionError):return
 raise AssertionError('Invalid scope or evidence accepted')
def run():
 manifest=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in manifest['files_SHA256'].items())
 g=json.loads((D/'execution-GO.template.json').read_bytes());rejected(lambda:m.validate_scope(g))
 g.update(GPU_authorized=True,publication_commit='42f2b42a2336561caf86a46cf4d3a52282ec904b',controller_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'));assert m.validate_scope(g) is g
 changes={'cards':1,'TP':True,'PP':2,'context':1024,'ordinary_budget':1024,'thinking_budget':4096,'max_HTTP':103,'max_Native':102,'GPU_max_s':29401,'Native_max_s':28801,'CPU_keeper_max_s':32401,'GPU_parent':910380,'GPU_node':'a02r4n12','CPU_keeper_parent':910374,'CPU_keeper_node':'a02r4n01','publication_branch':'main','owner_manifest_SHA256':'0'*64,'client_CPU635_receipt_SHA256':'0'*64,'DSO_CPU621_receipt_SHA256':'0'*64,'wrapper_SHA256':'0'*64,'namespace_CPU_receipt_SHA256':'0'*64,'strict_bare_JSON_format_qualified':True,'normal35_repeated':True,'one_attempt':False,'no_retry_reload':False}
 for k,v in changes.items():
  z=copy.deepcopy(g);z[k]=v;rejected(lambda:m.validate_scope(z))
 receipt=json.loads((G/'evidence/bw1000/Deep68-CPU635-whole-client-entry-r321.json').read_bytes())['receipt'];validate_receipt(receipt)
 for mutate in [lambda c:c.update(actual_step='910407.632'),lambda c:c.update(owner_source_manifest_SHA256='0'*64),lambda c:c.update(cases=c['cases'][:-1]),lambda c:c['cases'][0].update(synthetic_HTTP=67),lambda c:c['cases'][1].update(quality_returncode=1),lambda c:c['cases'][3].update(passed=False)]:
  z=copy.deepcopy(receipt);mutate(z);rejected(lambda:validate_receipt(z))
 with tempfile.TemporaryDirectory()as temp:
  root=Path(temp)
  for name in ['Deep-latest-full-owner-r300','Deep-semantic68-owner-r320']:shutil.copytree(B/name,root/'incoming'/name)
  p=root/'candidates/Deep-semantic68-CPU-r321/CPU-prepared.json';p.parent.mkdir(parents=True);p.write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
  # Preserve the exact original remote receipt bytes archived in the source readout.
  original=json.loads((G/'evidence/bw1000/Deep68-CPU635-whole-client-entry-r321.json').read_bytes())
  assert sha(p)==original['receipt_SHA256'],'Archived receipt cannot reconstruct exact byte identity'
  proof=verify_supplement(root,root/'incoming/Deep-semantic68-owner-r320')
  f=root/'incoming/Deep-semantic68-owner-r320/cleanup_owned_r287.py';f.write_bytes(f.read_bytes()+b'\n');rejected(lambda:verify_supplement(root,f.parent))
 tree=ast.parse((D/'controller_r322.py').read_text());old=ast.parse((B/'Deep-latest-controller-r303/controller_r303.py').read_text())
 def gpu_args(t):
  rows=[n.value for n in ast.walk(t)if isinstance(n,ast.Assign)and isinstance(n.value,ast.List)and any(isinstance(e,ast.Constant)and e.value=='--gres=dcu:8' for e in n.value.elts)];assert len(rows)==1;return ast.dump(rows[0])
 assert gpu_args(tree)==gpu_args(old),'Assigned8card GPU argv changed'
 failures=[n for n in ast.walk(tree) if isinstance(n,ast.If)and isinstance(n.test,ast.Call)and isinstance(n.test.func,ast.Name)and n.test.func.id=='terminal_failure'];assert len(failures)==1 and isinstance(failures[0].body[-1],ast.Return)
 assert 'actual_HTTP\')==68' in (D/'controller_r322.py').read_text() and 'actual_Native\')==68' in (D/'controller_r322.py').read_text()
 return {'completed':True,'passed':True,'GPU':0,'Model':0,'HTTP':0,'submit':0,'manifest_SHA256':sha(D/'SOURCE_MANIFEST.json'),'scope_negative_cases':len(changes),'receipt_negative_cases':6,'source_tamper_rejected':True,'original8card_GPU_argv_retained':True,'known_GPU_failure_returns_before_qualification':True,'source_compatibility':proof,'real_Model_or_performance_qualified':False}
if __name__=='__main__':print(json.dumps(run(),indent=2))
