import ast,pathlib,json,hashlib,threading,time,ctypes,types,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;p=r/'Deep_reference_sink_component_r8968.py';s=p.read_text();node=next(v for v in ast.parse(s).body if isinstance(v,ast.FunctionDef)and v.name=='invoke');mod=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
class Tensor:
 def __init__(self,ptr,strides):self.ptr=ptr;self.strides=strides
 def __getitem__(self,k):return self
 def data_ptr(self):return self.ptr
 def stride(self,k=None):return self.strides[k]if k is not None else self.strides
 def float(self):return self
vals=[0]*4;scratch=[0]*4;active=0;kind='swa';captured=[]
def fn(*a):
 cr,cw={'swa':(1,1),'hca128':(16,4),'csa4':(512,512)}[kind];assert len(a)==24 and list(a[:7])==[0,2048,8,2048,cr,128,cw];assert list(a[7:15])==[4096,512,1,512,1,512,1,1];captured.append([v.value for v in a[15:]])
 copies={'swa':512,'hca128':524,'csa4':896}[kind]if active else 2048
 for i,v in enumerate([2048,copies,2048-copies,active]):vals[i]+=v
 for i,v in enumerate([0,1,0,3]):scratch[i]+=v
 return 0
g=dict(state={},threading=threading,time=time,ctypes=ctypes,counts=lambda d:vals.copy(),scratch_counts=lambda d:scratch.copy(),torch=types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda d:None,current_stream=lambda:types.SimpleNamespace(cuda_stream=9)),isfinite=lambda x:types.SimpleNamespace(all=lambda:True)),check=lambda c:None,fn=fn)
exec(compile(mod,str(p),'exec'),g);rows=[]
for kind in ['swa','hca128','csa4']:
 cr,cw={'swa':(1,1),'hca128':(16,4),'csa4':(512,512)}[kind];g['state'][0]=dict(args=(Tensor(1,[4096,512,1]),Tensor(2,[512,1]),Tensor(3,[512,1]),Tensor(4,[1]),Tensor(5,[128,1]),Tensor(6,[cw,1]),Tensor(7,[2,1])),outs=[Tensor(8+i,[4096,512,1])for i in range(4)],thread=threading.get_native_id(),cr=cr,cw=cw,kind=kind)
 for which,arm in enumerate([0,1,1,0]):
  active=arm;z=g['invoke'](0,which);assert captured[-1][:7]==[1,2,3,4,5,6,7]and captured[-1][-1]==9;rows.append(dict(kind=kind,arm=arm,**z))
assert 'check(lib.deep_reference_scratch_mode_r8953(1))'in s and 'check(lib.deep_reference_sink_mode_r8967(arm))'in s and 'deep_latent_prefill_mode_r8933(0)'in s
cpp=r/'deep_latent_reference_sink_r8967.cpp';base=r/'deep_latent_reference_scratch_r8954.cpp';ops=lambda t:[line[line.index('ex::submit<'):]for line in t.splitlines()if 'ex::submit<'in line];assert ops(cpp.read_text())==ops(base.read_text())
out=dict(hardware='BW1000',level='CPU_CONSUMER_CONTRACT',status='PASSED',CPU_only=True,no_GPU_dlopen=True,actual_entry_AST=True,ABI_args=24,faithful_fixtures=3,fixed_ABBA=[0,1,1,0],observed_calls=rows,scratchON_both_arms=True,original_fused_OFF=True,actual_five_Ops_arguments_text_identical=True,entry_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),CPP_SHA256=hashlib.sha256(cpp.read_bytes()).hexdigest());dst=r/'Deep-reference-sink-consumer-CPU-r8968.json';dst.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps({'passed':True,'calls':len(rows),'SHA256':hashlib.sha256(dst.read_bytes()).hexdigest()}))
