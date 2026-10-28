"""Pure stdlib provider classification tests; no DSO import or GPU execution."""
import pathlib,tempfile,json,hashlib,ast
from binding_check_r152b import verify_bindings,REQUIRED_NAMES
D=pathlib.Path(__file__).resolve().parent;rows=[]
with tempfile.TemporaryDirectory(prefix='deep-provider-')as temp:
 root=pathlib.Path(temp);names=sorted(REQUIRED_NAMES)+['libinfiniccl.so'];expected={}
 for n in names:
  p=root/n;p.write_bytes(n.encode());expected[str(p)]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
 sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();required={pathlib.Path(p)for p in expected if pathlib.Path(p).name in REQUIRED_NAMES};cc=root/'libinfiniccl.so'
 def run(label,loaded,accept,hasher=sha):
  got=None;error=None
  try:got=verify_bindings(expected,loaded,hasher)
  except AssertionError as exc:error=str(exc)
  assert (error is None)==accept,label;rows.append({'case':label,'passed':True,'accepted':accept,'provider_records':got,'error':error})
 run('unused_CCL_absent_is_not_required',required,True)
 assert next(x for x in rows[-1]['provider_records']if x['expected_path']==str(cc.resolve()))['status']=='not_required_absent'
 run('optional_CCL_loaded_must_match',required|{cc},True)
 other=root/'other';other.mkdir();wrong=other/'libinfiniccl.so';wrong.write_bytes(b'wrong');run('wrong_optional_CCL_rejected',required|{wrong},False)
 op=root/'libinfiniops.so';run('missing_required_operator_rejected',required-{op},False)
 wrongop=other/'libinfiniops.so';wrongop.write_bytes(b'wrong');run('wrong_required_origin_rejected',(required-{op})|{wrongop},False)
 run('wrong_required_SHA_rejected',required,False,lambda p:'0'*64 if pathlib.Path(p).name=='libinfiniops.so'else sha(p))
for p in D.glob('*.py'):ast.parse(p.read_bytes())
assert all(b'\r'not in p.read_bytes()for p in D.glob('*.sh'))
q={'CPU_only':True,'GPU':False,'Torch_imported':False,'DSO_dlopen':False,'actual_operator_execution':False,'source_AST_passed':True,'required_names':sorted(REQUIRED_NAMES),'cases':rows,'passed':True,'reused_actual_CPU_receipt_SHA256':'3cc5496a4c6e4a49de3bdad10e31b51f0ee3bb28d2ca634d2faa8e57e3dffa3a','reused_wrapper_SHA256':'71dbd8627e2ba9ec9e0dca3462c5efe69b5fd4525345bad34eac4923da54c09d','old_GPU1159':'FAILED before operator body; no component qualification','new_GPU_state':'NOT_RUN','no_forced_dlopen_CCL':True};(D/'CPU-provider-contract-r152b.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'passed':True,'cases':len(rows),'operator_executed':False}))
