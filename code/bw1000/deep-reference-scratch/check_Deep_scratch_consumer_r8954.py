import ast,pathlib,json,hashlib,threading,time,ctypes,types
r=pathlib.Path.cwd();p=r/'Deep_reference_scratch_component_r8954.py';s=p.read_text();tree=ast.parse(s);node=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='invoke');mod=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
class Tensor:
 def __init__(self,ptr,strides):self.ptr=ptr;self.strides=strides
 def __getitem__(self,k):return self
 def data_ptr(self):return self.ptr
 def stride(self,k=None):return self.strides[k]if k is not None else self.strides
 def float(self):return self
vals=[0]*4;captured=[];active=0
def fn(*a):
 assert len(a)==24 and list(a[:7])==[0,2048,8,2048,512,128,512];assert list(a[7:15])==[4096,512,1,512,1,512,1,1];captured.append([v.value for v in a[15:]])
 vals[active]+=1;vals[3 if active else 2]+=3 if active else 6144;return 0
g=dict(state={0:dict(args=(Tensor(1,[4096,512,1]),Tensor(2,[512,1]),Tensor(3,[512,1]),Tensor(4,[1]),Tensor(5,[128,1]),Tensor(6,[512,1]),Tensor(7,[2,1])),outs=[Tensor(8+i,[4096,512,1])for i in range(4)],thread=threading.get_native_id(),cr=512,cw=512)},threading=threading,time=time,ctypes=ctypes,counts=lambda d:vals.copy(),torch=types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda d:None,current_stream=lambda:types.SimpleNamespace(cuda_stream=9)),isfinite=lambda x:types.SimpleNamespace(all=lambda:True)),check=lambda c:None,fn=fn)
exec(compile(mod,str(p),'exec'),g);observed=[]
for which,arm in enumerate([0,1,1,0]):
 active=arm;observed.append(g['invoke'](0,which));assert captured[-1][:7]==[1,2,3,4,5,6,7]and captured[-1][-1]==9
assert [v['counter_delta']for v in observed]==[[1,0,6144,0],[0,1,0,3],[0,1,0,3],[1,0,6144,0]];assert 'deep_latent_prefill_mode_r8933(0)'in s and 'original_fused_call_counts' in s
base=next(v['text']for v in json.loads((r/'Deep908-source-binding-r8945.json').read_bytes())['sources']if v['source'].endswith('/deep_latent_reference_r681.cpp'));cpp=(r/'deep_latent_reference_scratch_r8954.cpp').read_text();ops=lambda z:[v[v.index('ex::submit<'):]for v in z.splitlines()if'ex::submit<'in v];assert ops(base)==ops(cpp)
out=dict(hardware='BW1000',level='CPU_CONSUMER_CONTRACT',status='PASSED',CPU_only=True,no_GPU_dlopen=True,actual_entry_AST=True,ABI_args=24,fixed_ABBA=[0,1,1,0],observed_counter_deltas=[v['counter_delta']for v in observed],same_operands_output_shape_stride_stream=True,original_fused_OFF=True,actual_five_Ops_arguments_text_identical=True,entry_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),CPP_SHA256=hashlib.sha256((r/'deep_latent_reference_scratch_r8954.cpp').read_bytes()).hexdigest());q=r/'Deep-reference-scratch-consumer-CPU-r8954.json';q.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
