"""Root-bound new guard Native receipt. No ac937 fallback and no library load."""
import pathlib,json,hashlib,re,copy
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');BASE=R/'incoming/Deep-C4-independent64-r9278/modern_Deep_independent64_r9278.py'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def binding():
 b=json.loads((D/'Native-binding.json').read_bytes())
 assert b['rows']==1 and b['capacity']==32768 and b['prefill_chunk']==1024
 for key in ['CPU_receipt_SHA256','library_SHA256','guard_source_manifest_SHA256']:
  assert re.fullmatch('[0-9a-f]{64}',b[key])and b[key]!='0'*64,'Root must fill the actual new Native pin'
 p=pathlib.Path(b['CPU_receipt']);assert p.is_relative_to(R/'candidates')and sha(p)==b['CPU_receipt_SHA256'];q=json.loads(p.read_bytes())
 assert q['completed']and q['CPU_only']and not q['GPU_initialized']and not q['Native_loaded']and not q['weights_opened']and q['GPU_TUs_compiled']==0 and q['all_old_guard_code_removed']
 assert q['source_manifest_SHA256']==b['guard_source_manifest_SHA256']and q['library_SHA256']==b['library_SHA256']and sha(q['library'])==b['library_SHA256']
 assert b['library_SHA256']!='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835','Old ac937 is not the new32K guard artifact'
 objects=dict(q['reused_objects_SHA256'])
 for u in q['compiled']:assert u['fresh_M_equal_MD'];objects[u['object']]=u['object_SHA256']
 assert len(objects)==48 and all(sha(p)==h for p,h in objects.items())
 for p,v in q['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 normalized={'completed':True,'CPU_only':True,'libraries':{'full':{'path':q['library'],'SHA256':q['library_SHA256']}},'link':{'full':{'objects':objects,'argv':q['link']}},'actual_DTK_bindings':q['actual_DTK_bindings']}
 return b,q,normalized
def prebuild():return binding()[2]
def manifest():
 b,q,_=binding();m=json.loads(BASE.with_name('manifest.json').read_bytes());m=copy.deepcopy(m);relative=str(pathlib.Path(b['CPU_receipt']).relative_to(R));m['cpu_receipt_path']=relative;m['DSO_SHA256']=q['library_SHA256'];m['existing_remote_SHA256'][relative]=b['CPU_receipt_SHA256'];return m
