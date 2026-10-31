"""CPU-only regression for exact library identities and the pinned legacy AST delta."""
from pathlib import Path
import ast,base64,copy,hashlib,importlib.util,json,sys,datetime
G=Path(__file__).resolve().parents[1]
B=G/'code/bw1000/public-native-bridge'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(private_receipt):
 out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU':0,'Native_dlopen':0,'weights':0,'Model_qualified':False,'models':{}}
 for model,new,old in [('GLM',299,290),('Deep',300,287)]:
  d=B/(model+'-latest-full-owner-r'+str(new));o=B/(model+'-latest-full-owner-r'+str(old));mf=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());omf=o/'SOURCE_MANIFEST.json'
  assert sha(omf)==mf['previous_owner_manifest_SHA256']
  assert set(mf['files_SHA256'])=={p.relative_to(d).as_posix() for p in d.rglob('*')if p.is_file()and p.name!='SOURCE_MANIFEST.json'}
  for n,h in mf['files_SHA256'].items():
   assert sha(d/n)==h
   if n.endswith('.py'):compile((d/n).read_bytes(),str(d/n),'exec')
  actual_changed={n for n in json.loads(omf.read_bytes())['files_SHA256'] if (d/n).read_bytes()!=(o/n).read_bytes()}
  assert actual_changed==set(mf['changed_members'])
  assert set(mf['unchanged_members'])|actual_changed==set(json.loads(omf.read_bytes())['files_SHA256'])
  m=load('mapping_'+model,d/'common_mapping_audit.py');pins=m.expected(model);paths=set(pins);assert len(paths)==2
  assert m.check_metadata(model,paths,pins)['passed']
  ext=next(p for p in paths if '_infinilm.cpython-'in p);rt=next(p for p in paths if 'libinfinilm_runtime.so'in p)
  oldext='/refactor/candidates/rebased-5f91ad7-r3668/Common/'+model+'/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so'
  cases={'old_extension':({oldext,rt},pins),'dual_extension':(paths|{oldext},pins),'missing_extension':({rt},pins),'missing_runtime':({ext},pins),'foreign_runtime':({ext,'/foreign/libinfinilm_runtime.so'},pins),'extra_family_library':(paths|{'/foreign/_infinilm.cpython-310-x86_64-linux-gnu.so'},pins),'wrong_SHA':(paths,{**pins,ext:'0'*64}),'extra_SHA':(paths,{**pins,'/extra':'0'*64})}
  for name,(p,h)in cases.items():
   try:m.check_metadata(model,p,h)
   except AssertionError:pass
   else:raise AssertionError('Accepted negative: '+name)
  source=load('source_pin_'+model,d/('source_pin_r290.py'if model=='GLM'else'source_pin_r281.py'));assert source.verify(sha(d/'SOURCE_MANIFEST.json'))==mf
  try:source.verify('0'*64)
  except ValueError:pass
  else:raise AssertionError('Invalid source manifest accepted')
  out['models'][model]={'manifest_SHA256':sha(d/'SOURCE_MANIFEST.json'),'files':len(mf['files_SHA256']),'changed_old_members':sorted(actual_changed),'negative_mapping_cases':list(cases),'fixed_production_pins':pins,'actual_source_verifier_passed':True,'all_python_syntax_passed':True}
 # Import only transform definitions, never private loader or GPU libraries.
 q=json.loads(Path(private_receipt).read_bytes());raw=base64.b64decode(q['source_base64']);assert hashlib.sha256(raw).hexdigest()=='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'
 patch=load('legacy_mapping_patch_test',B/'Deep-latest-full-owner-r300/legacy_mapping_patch.py')
 def transformed(owner,extra):
  t=ast.parse((B/owner/'bootstrap_r206b.py').read_bytes());fn=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='transform');ns={'ast':ast,'BASE':Path('/fixed-private-BASE'),**extra};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<public transform>','exec'),ns);return ns['transform'](raw)
 before,bp=transformed('Deep-latest-full-owner-r287',{})
 after,ap=transformed('Deep-latest-full-owner-r300',{'replace_mapping_audit':patch.replace_mapping_audit})
 def locate(tree,name):
  return [(p,i)for p in ast.walk(tree)if isinstance(getattr(p,'body',None),list)for i,n in enumerate(p.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
 hits=locate(before,'loaded_paths');assert len(hits)==1;p,i=hits[0];oldblock=copy.deepcopy(p.body[i:i+5]);del p.body[i:i+5]
 hits=locate(after,'__common_proof_r300__');assert len(hits)==1;p,i=hits[0];newblock=copy.deepcopy(p.body[i:i+3]);del p.body[i:i+3]
 assert ast.dump(before)==ast.dump(after),'Unexpected changes outside exact legacy audit block'
 assert {k:v for k,v in ap.items()if k!='mapping_audit_patch'}==bp
 pins=out['models']['Deep']['fixed_production_pins'];maps='\n'.join('address perms offset dev ino '+p for p in pins)
 class MapPath:
  def __init__(self,p):assert p=='/proc/self/maps'
  def read_text(self):return maps
 class Paths:Path=MapPath
 try:exec(compile(ast.fix_missing_locations(ast.Module(body=oldblock,type_ignores=[])),'<actual historical audit>','exec'),{'pathlib':Paths,'report':{},'hashlib':hashlib,'prebuild':{'actual_DTK_bindings':{p:{'SHA256':h}for p,h in pins.items()}}})
 except AssertionError:pass
 else:raise AssertionError('Actual old condition did not reject correct new mapping')
 for name,t in [('missing',ast.parse('x=1')),('changed',ast.parse(raw)),('duplicate',ast.parse(raw))]:
  if name=='changed':p,i=locate(t,'loaded_paths')[0];p.body[i+1]=ast.Pass()
  if name=='duplicate':p,i=locate(t,'loaded_paths')[0];p.body[i:i]=copy.deepcopy(p.body[i:i+5])
  try:patch.replace_mapping_audit(t)
  except AssertionError:pass
  else:raise AssertionError('Unexpected AST patch accepted: '+name)
 out['Deep_AST']={'private_BASE_SHA256':hashlib.sha256(raw).hexdigest(),'private_source_not_published':True,'actual_old_mapping_condition_reproduced_failure':True,'whole_transformed_AST_unchanged_outside_exact_audit_block':True,'patch_negatives_rejected':['missing','changed','duplicate'],'patch_proof':ap['mapping_audit_patch']}
 # Verify the early audit remains before model construction/weight loading.
 glm=(B/'GLM-latest-full-owner-r299/model_entry_r290.py').read_text();assert glm.index('_preengine = _preengine_validate()')<glm.index('_early_common_library_audit = _validate_common_loaded("GLM")')<glm.index('def worker(')
 deep=(B/'Deep-latest-full-owner-r300/preweight_r206b.py').read_text();assert deep.index("lib=ctypes.CDLL(c['library'])")<deep.index("common_proof=validate_deep(c['actual_DTK_bindings'])")<deep.index('return dict(')
 out.update(completed=True,passed=True,early_audit_placement_source_checked=True,qualification='Source closure, pure-data identity negatives, actual old condition and full transformed AST delta; no live GPU DSO, Model or performance qualification')
 return out
if __name__=='__main__':
 result=run(sys.argv[1]);p=Path(sys.argv[2]);assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
