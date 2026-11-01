"""Derived latest worker binding, retaining the original32K guard proof."""
import copy,pathlib,json,hashlib
from native_binding_guard_r206b import BASE,binding as guard_binding
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
BUILD_SHA='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5'
NATIVE_SHA='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
EXTENSION_SHA='6e1a7e7643f84a52ab147cfc3c55eee4816affc6898109dbef31d559feaa86d1'
def binding():
 old_binding,old_receipt,old_normal=guard_binding();p=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(p)==BUILD_SHA;build=json.loads(p.read_bytes());assert build['completed']and build['passed']and build['CPU_only']and not build['GPU']and not build['Native_dlopen']and not build['weights_or_Model_loaded']and build['new_GPU_TUs']==0
 f=build['families']['Deep'];n=f['Native'];w=f['Native_worker'];e=f['Common_extension'];cw=f['Common_worker'];assert n['original_SHA256']==old_receipt['library_SHA256']and n['new_SHA256']==NATIVE_SHA and e['new_SHA256']==EXTENSION_SHA
 old_objects=old_normal['link']['full']['objects'];replaced=R/'candidates/deep-c4-projection-model-r5920/objects/unit-8.o';assert str(replaced)in old_objects
 objects=dict(n['unchanged_objects_SHA256']);assert len(objects)==47 and objects=={p:h for p,h in old_objects.items()if p!=str(replaced)};objects[w['new_object']]=w['new_object_SHA256'];assert len(objects)==48
 for p,h in objects.items():assert sha(p)==h
 for record in [w,cw]:
  assert record['all_project_and_system_dependencies_equal_except_worker_source']and sha(record['new_source'])==record['new_source_SHA256']and sha(record['new_object'])==record['new_object_SHA256']
  for p,h in record['dependency_SHA256'].items():assert sha(p)==h
 assert cw['new_source_SHA256']=='734c65675547d9415778eb75877489024a10c278650c25c7da153afa56fea06a'
 assert sha(n['new'])==NATIVE_SHA and sha(e['new'])==EXTENSION_SHA
 bindings=copy.deepcopy(old_receipt['actual_DTK_bindings']);oldext=e['original'];assert oldext in bindings and bindings[oldext]['SHA256']==e['original_SHA256'];bindings[e['new']]=bindings.pop(oldext);bindings[e['new']]['SHA256']=EXTENSION_SHA;bindings[e['new']]['resolved']=str(pathlib.Path(e['new']).resolve())
 for p,v in bindings.items():assert sha(p)==v['SHA256']
 b=dict(CPU_receipt=str(R/'candidates/latest-family-native-r269/CPU-build.json'),CPU_receipt_SHA256=BUILD_SHA,library_SHA256=NATIVE_SHA,rows=1,capacity=32768,prefill_chunk=1024,original_guard_binding=old_binding,latest_worker_only=True)
 q=copy.deepcopy(old_receipt);q.update(library=n['new'],library_SHA256=NATIVE_SHA,actual_DTK_bindings=bindings,derived_binding=True,derivation_receipt_SHA256=BUILD_SHA,original_guard_receipt_SHA256=old_binding['CPU_receipt_SHA256'])
 normalized={'completed':True,'CPU_only':True,'libraries':{'full':{'path':n['new'],'SHA256':NATIVE_SHA}},'link':{'full':{'objects':objects,'argv':n['link_argv']}},'actual_DTK_bindings':bindings,'derived_binding':True,'original_guard_proof_preserved':True}
 return b,q,normalized

def prebuild():return binding()[2]
def manifest():
 b,q,_=binding();m=json.loads(BASE.with_name('manifest.json').read_bytes());relative=str(pathlib.Path(b['CPU_receipt']).relative_to(R));m['cpu_receipt_path']=relative;m['DSO_SHA256']=NATIVE_SHA;m['existing_remote_SHA256'][relative]=BUILD_SHA;return m
