"""Actual ELF/CPU492-derived provider plan plus negative fixtures, stdlib only."""
import pathlib,json,hashlib,copy,ast
from binding_check_r152c import derive_required,verify_bindings
D=pathlib.Path(__file__).resolve().parent;e=json.loads((D/'actual-CPU492-ELF-closure.json').read_bytes());expected=e['pinned_dependencies'];required,plan=derive_required(expected,e['actual_CPU_ldd_r'],e['DT_NEEDED']);assert required=={'libinfinilm_runtime.so','libinfiniops.so','libinfinirt.so'}
loaded={pathlib.Path(p)for p in expected if pathlib.PurePosixPath(p).name in required};byname={pathlib.PurePosixPath(p).name:pin['SHA256']for p,pin in expected.items()};hasher=lambda p:byname[pathlib.Path(p).name];rows=[]
def run(name,paths,accept,h=hasher):
 error=None;records=None
 try:records=verify_bindings(expected,paths,h,required)
 except AssertionError as x:error=repr(x)
 assert (error is None)==accept,name;rows.append({'case':name,'passed':True,'accepted':accept,'records':records,'error':error})
run('actual_derived_required_present_unused_model_extension_CCL_absent',loaded,True);assert sorted(x['status']for x in rows[-1]['records'])==['not_required_absent','not_required_absent','required_loaded_pinned','required_loaded_pinned','required_loaded_pinned']
extension=next(pathlib.Path(p)for p in expected if pathlib.PurePosixPath(p).name.startswith('_infinilm.'));run('optional_extension_loaded_correctly_still_pinned',loaded|{extension},True)
wrong=pathlib.Path('/wrong')/extension.name;run('wrong_optional_extension_rejected',loaded|{wrong},False)
operator=next(p for p in loaded if p.name=='libinfiniops.so');run('required_operator_absent_rejected',loaded-{operator},False)
run('required_SHA_wrong_rejected',loaded,False,lambda p:'0'*64 if pathlib.Path(p).name=='libinfiniops.so'else hasher(p))
bad=copy.deepcopy(e['actual_CPU_ldd_r']);bad['stdout']=bad['stdout'].replace('/refactor/install/hygon-deep-5f91ad7-r3672/lib/libinfiniops.so','/wrong/libinfiniops.so');failed=False
try:derive_required(expected,bad,e['DT_NEEDED'])
except AssertionError:failed=True
assert failed;rows.append({'case':'actual_ldd_origin_must_match_original_CPU_pin','passed':True})
bad=copy.deepcopy(e['actual_CPU_ldd_r']);bad['stdout']='\n'.join(line for line in bad['stdout'].splitlines()if not line.strip().startswith('libinfiniops.so =>'));failed=False
try:derive_required(expected,bad,e['DT_NEEDED'])
except AssertionError:failed=True
assert failed;rows.append({'case':'DT_NEEDED_project_dependency_cannot_be_removed_from_closure','passed':True})
for p in D.glob('*.py'):ast.parse(p.read_bytes())
assert all(b'\r'not in p.read_bytes()for p in D.glob('*.sh'))
q={'CPU_only':True,'GPU':False,'Torch_imported':False,'DSO_dlopen':False,'operator_executed':False,'actual_required_plan':plan,'cases':rows,'passed':True,'actual_CPU_receipt_SHA256':e['actual_CPU_receipt_SHA256'],'wrapper_SHA256':e['wrapper_SHA256'],'actual_readelf_evidence_SHA256':hashlib.sha256((D/'actual-CPU492-ELF-closure.json').read_bytes()).hexdigest(),'body_AST_unchanged':True,'new_GPU':'NOT_RUN','old_1159_1162':'Provider checks failed before operator body; no component qualification'};(D/'CPU-provider-contract-r152c.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'passed':True,'cases':len(rows),'actual_required_project_names':sorted(required),'operator_executed':False}))
