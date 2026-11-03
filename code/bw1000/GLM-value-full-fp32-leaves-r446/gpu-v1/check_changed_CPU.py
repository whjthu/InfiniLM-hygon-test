"""Only new policy/highctx/actual numeric-counter consumer wiring, no old suite."""
from pathlib import Path
import ast,copy,ctypes,json,sys,tempfile
from types import SimpleNamespace
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import binding,consumer

def run():
 for p in D.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
 parent=D.parents[1]/'GLM-row-FP8-roundtrip-value-r433/gpu-v1'
 for name in ['checkpoint_loader.py','original_consumer_r400.py','diagnostic_io.py']:assert(D/name).read_bytes()==(parent/name).read_bytes(),name
 def query_operands(p):return[ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(p.read_bytes()))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='glm_attention_query_r398']
 assert query_operands(D/'original_consumer_r400.py')==query_operands(parent/'original_consumer_r400.py')
 seen=[];high=0x1234567887654321
 @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_int)
 def actual_thunk(ctx,mode):seen.append((ctx,mode));return 0
 lib=SimpleNamespace(glm_attention_value_fullfp32_r446=actual_thunk);fn=binding.bind_policy(lib)
 assert fn.argtypes==[ctypes.c_void_p,ctypes.c_int]and fn.restype is ctypes.c_int and fn(high,1)==0 and seen==[(high,1)]
 base=dict(past=32767,count=1,union_U=2048,seeded=32768,mode=True,persistent_V_bytes=0,value_roundtrip=True,r433_chosen=1,r433_rowquant_submits=1,r433_roundtrip_F32_identity_calls=1,r446_F32_aggregation_GEMV_leaves=8,r446_F32_V_GEMV_leaves=8,r446_final_software_BF16_cast_leaves=1,selectedV_HOST_completed_expand_calls=0,diagnostic_snapshot_detail=2)
 assert consumer.native_contract(base,1,True,2)
 negatives=[]
 for name in ['past','count','union_U','seeded','r446_F32_aggregation_GEMV_leaves','r446_F32_V_GEMV_leaves','r446_final_software_BF16_cast_leaves']:
  bad=dict(base);bad[name]=0;assert not consumer.native_contract(bad,1,True,2);negatives.append(name)
 record=dict(raw_outputs_complete=True,result=dict(identity_valid=True,canary=True,numerics=[dict(reference_RMS=.125,finite=True,NRMSE=.001)]))
 assert consumer.numeric_contract(record)
 for field,value in [('reference_RMS',0),('finite',False),('NRMSE',.020001)]:
  bad=copy.deepcopy(record);bad['result']['numerics'][0][field]=value;assert not consumer.numeric_contract(bad)
 # Actual new consume loop, real tempfiles, new counter failure is persisted
 # before failstop. Native CABI/weights remain explicitly recording leaves.
 with tempfile.TemporaryDirectory(prefix='r446-changed-')as temp:
  root=Path(temp);path=root/'report.json'
  class Leaf:
   query_CAPI_invoked=False;last_query_rc=None;latest_result=None
   def __init__(self):self.closed=False;self.query_count=0
   def check_deadline(self):pass
   def begin(self,mode):self.mode=mode
   def seed(self,*args):pass
   def query(self,case,ids,arm):
    self.query_count+=1;self.query_CAPI_invoked=True;self.last_query_rc=0
    return dict(identity_valid=self.mode==0,canary=True,numerics=[dict(reference_RMS=.1,finite=True,NRMSE=0.)])
   def persist(self,row):
    p=root/('sample'+str(self.query_count)+'.raw');p.write_bytes(b'recording');row['raw_outputs_complete']=True;row['raw_outputs']=[dict(path=str(p))]
   def validate(self,row):
    assert Path(row['raw_outputs'][0]['path']).read_bytes()==b'recording'
    if not consumer.numeric_contract(row):raise RuntimeError('recorded newcounter refusal')
   def close(self):self.closed=True
  leaf=Leaf()
  def save(rows,error):path.write_text(json.dumps(dict(records=rows,error=error)))
  try:consumer.consume(leaf,save)
  except RuntimeError as e:assert'newcounter'in str(e)
  else:raise AssertionError('Bad firstON counter did not stop')
  q=json.loads(path.read_bytes());assert leaf.closed and leaf.query_count==2 and len(q['records'])==2 and q['records'][1]['raw_outputs_complete']and q['error']
 return dict(passed=True,GPU=0,Native_dlopen=False,Runtime=False,Slurm=False,new_CABI_actual_libffi_highctx=True,old_query_operands_and_loader_bytes_equal=True,new_native_counter_negatives=negatives,nonzero_finite_original_diagnostic_limits_preserved=True,actual_new_consumer_persist_before_counter_STOP=True,old_suites_rerun=False)
if __name__=='__main__':print(json.dumps(run(),indent=2))
