import pathlib,json,hashlib,ctypes,os,time,statistics
import numpy as np
import torch
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('deep-smallT-packed-r9228-'+tag+'.json');assert not path.exists();manifest=json.loads((inc/'manifest.json').read_bytes());cp=r/'candidates/deep-smallT-packed-r9228/CPU-build.json';assert sha(cp)==manifest['CPU_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['library_SHA256']==manifest['DSO_SHA256']and sha(cpu['library'])==manifest['DSO_SHA256'];assert all(sha(f)==h for f,h in cpu['reused_objects_SHA256'].items());assert all(sha(f)==v['SHA256']for f,v in cpu['actual_DTK_bindings'].items());prov=json.loads((inc/'fixture-provenance.json').read_bytes());torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);lib=ctypes.CDLL(cpu['library']);fn=lib.modern_deep_latent_strided_r681;fn.argtypes=[ctypes.c_int]+[ctypes.c_size_t]*6+[ctypes.c_int64]*8+[ctypes.c_void_p]*9;fn.restype=ctypes.c_int;lib.modern_deep_latent_error.restype=ctypes.c_char_p;lib.modern_deep_latent_shutdown.restype=ctypes.c_int
for n in ['deep_latent_prefill_mode_r8933','deep_reference_scratch_mode_r8953','deep_reference_packed_batch_mode_r8996b','deep_mode1_sink_r9207','deep_smallt_packed_mode_r9228','deep_scalar_cast_mode_r9054','deep_values_reuse_mode_r9083']:getattr(lib,n).argtypes=[ctypes.c_int]
for n in ['deep_smallt_packed_counts_r9228','deep_mode1_sink_counts_r9207']:getattr(lib,n).argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
lib.deep_smallt_packed_get_r9228.restype=ctypes.c_int
for n in ['modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681']:getattr(lib,n).argtypes=[ctypes.c_int];getattr(lib,n).restype=ctypes.c_uint64
report={'completed':False,'normal':False,'step':tag,'DSO_SHA256':manifest['DSO_SHA256'],'fixtures':[],'scope':'Actual existing modern_deep_latent_strided_r681 outerdispatcher; no manual referencecall/providerforce. T2/4/7 truefallbackcandidate; T8/16/32 fusedzero-hitcontrols; B1original. Real1083promptpasts/EOSsurvivorrows,Ncounts; syntheticuniqueKV/RMSRoPE; fullblockinglatentcompoundwall+eventbracket. NotModel/taskquality/perf.','correction':'r9222 reference-only audit missed outerT8..64fused dispatch; preserved/superseded. precision5changescollective notattentionfusedgate.'}
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
def check(c):
 if c:raise RuntimeError(lib.modern_deep_latent_error().decode())
def ct(n,count):
 a=(ctypes.c_uint64*count)();assert getattr(lib,n)(0,a,count)==0;return list(a)
def diff(a,b):return [v-u for u,v in zip(a,b)]
try:
 assert lib.deep_smallt_packed_get_r9228()==0;assert lib.deep_smallt_packed_mode_r9228(2)==1
 for n,v in [('deep_latent_prefill_mode_r8933',0),('deep_reference_scratch_mode_r8953',1),('deep_reference_packed_batch_mode_r8996b',1),('deep_mode1_sink_r9207',1),('deep_scalar_cast_mode_r9054',1),('deep_values_reuse_mode_r9083',0)]:check(getattr(lib,n)(v))
 specs=[]
 for T in [1,2,4,8,16,32]:
  for kind in ['swa','hca128','csa4']:specs.append((kind,list(range(T)),prov['past'][:T],'first_decode'))
 tail=prov['eos_survivor7']
 for kind in ['hca128','csa4']:specs.append((kind,tail['rows'],tail['past'],'actual_EOS7_step'+str(tail['step'])))
 for number,(kind,logical,past,label)in enumerate(specs):
  T=len(past);ratio={'swa':0,'hca128':128,'csa4':4}[kind];cw=1 if not ratio else max((v+1)//ratio for v in past)if kind=='hca128'else 512;nc=[0 if not ratio else min(cw,(v+1)//ratio)for v in past];cr=max(1,sum(nc));sw=128;si=np.arange(T*sw,dtype=np.int64).reshape(T,sw);ci=np.full((T,cw),-1,np.int64);lens=np.zeros((T,2),np.int32);lens[:,0]=128;lens[:,1]=nc;co=0;gen_np=np.random.default_rng(922800+number)
  for i,n in enumerate(nc):
   if n:ci[i,:n]=co+(gen_np.permutation(n)if kind=='csa4'else np.arange(n));co+=n
  assert co==sum(nc)and len(set(logical))==T;N=[128+n for n in nc];runs=[];i=0
  while i<T:
   end=i+1
   while end<T and N[end]==N[i]:end+=1
   if end-i>=2:runs.append((i,end-i,N[i]))
   i=end
  gen=torch.Generator(device='cuda');gen.manual_seed(922800+number)
  def norm(shape,positions=None):
   x=torch.randn(shape,device='cuda',dtype=torch.float32,generator=gen);x=x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
   if positions is not None:
    pos=torch.tensor(positions,device='cuda').float();freq=torch.pow(10000.,-torch.arange(0,64,2,device='cuda').float()/64);angle=pos[:,None]*freq[None,:];a=x[...,-64::2].clone();b=x[...,-63::2].clone();c=angle.cos();s=angle.sin()
    if len(shape)==3:c=c[:,None,:];s=s[:,None,:]
    x[...,-64::2]=a*c-b*s;x[...,-63::2]=a*s+b*c
   return x.to(torch.bfloat16)
  query=norm((T,8,512),past);swa=norm((T*sw,512),sum([list(range(v+1-sw,v+1))for v in past],[]));comp=norm((cr,512));sink=torch.linspace(-.5,.5,8,device='cuda',dtype=torch.float32);args=[query,swa,comp,sink,torch.from_numpy(si).cuda(),torch.from_numpy(ci).cuda(),torch.from_numpy(lens).cuda()];clones=[v.clone()for v in args];torch.cuda.synchronize();plan=[('verify',0),('verify',1),('restore',0)]if T==1 else[('warm',0),('warm',1)]+[('measure',v)for v in [0,1,1,0,0,1,1,0]]+[('restore',0)];row={'label':label,'kind':kind,'T':T,'logical_rows':logical,'past':past,'N':N,'N_histogram':{str(n):N.count(n)for n in sorted(set(N))},'run_lengths':runs,'cr':cr,'cw':cw,'operands':'syntheticuniqueKV, query/SWA RMSRoPE, compressedRMS','metadata_SHA256':hashlib.sha256(si.tobytes()+ci.tobytes()+lens.tobytes()).hexdigest(),'measurements':[]};report['fixtures'].append(row);outputs=[];save()
  for phase,arm in plan:
   check(lib.deep_smallt_packed_mode_r9228(arm));back=torch.full((query.numel()+128,),42.,device='cuda',dtype=torch.bfloat16);output=back[:query.numel()].view_as(query);torch.cuda.synchronize();before=ct('deep_smallt_packed_counts_r9228',9);sinkbefore=ct('deep_mode1_sink_counts_r9207',5);f0=int(lib.modern_deep_latent_fused_calls_r681(0));r0=int(lib.modern_deep_latent_fallback_calls_r681(0));report['active_call']={'fixture':number,'kind':kind,'T':T,'past':past,'logical_rows':logical,'N':N,'phase':phase,'arm':arm,'smallT_before':before};save();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);stream=torch.cuda.current_stream();start.record(stream);tick=time.perf_counter();ptr=lambda x:ctypes.c_void_p(x.data_ptr());call=[0,T,8,T*sw,cr,sw,cw,*query.stride(),*swa.stride(),*comp.stride(),sink.stride(0),*[ptr(v)for v in args],ptr(output),ctypes.c_void_p(stream.cuda_stream)];rc=fn(*call);report['active_call']['native_returncode']=rc;check(rc);end.record(stream);end.synchronize();wall=time.perf_counter()-tick;gpu=start.elapsed_time(end);counts=diff(before,ct('deep_smallt_packed_counts_r9228',9));sd=diff(sinkbefore,ct('deep_mode1_sink_counts_r9207',5));fd=int(lib.modern_deep_latent_fused_calls_r681(0))-f0;rd=int(lib.modern_deep_latent_fallback_calls_r681(0))-r0
   if T>=8:expected=[0]*9;es=[0]*5;assert fd==1 and rd==0
   else:
    chosen=bool(arm and T>=2 and runs);queries=sum(v[1]for v in runs)if chosen else 0;nrun=len(runs)if chosen else 0;expected=[1,int(T>=2),int(chosen),int(not chosen),T-queries,nrun,queries,3*int(chosen),queries-nrun];es=[0,0,nrun,queries,queries-nrun]if chosen else[0]*5;assert fd==0 and rd==1
   assert counts==expected and sd==es,(counts,expected,sd,es);assert torch.isfinite(output.float()).all()and bool((back[-128:]==42).all())and all(torch.equal(a,b)for a,b in zip(args,clones));row['measurements'].append({'phase':phase,'arm':arm,'wall_seconds':wall,'event_bracket_ms':gpu,'smallT_counts':counts,'sink_counts':sd,'fused_calls':fd,'fallback_calls':rd,'finite':True,'readonly':True,'canary':True,'driver_free_total':list(torch.cuda.mem_get_info())});outputs.append(output);report['active_call']=None;save();del start,end
  ref=outputs[0 if T==1 else 2].float();row['numerical']=[{'phase':a['phase'],'arm':a['arm'],'maxabs':float((b.float()-ref).abs().max()),'relative_RMS':float((b.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-9))}for a,b in zip(row['measurements'],outputs)];row['numerical_contract_passed']=all(v['relative_RMS']<=0.01 for v in row['numerical']);assert row['numerical_contract_passed'];save();del args,clones,outputs,query,swa,comp,sink,back,output
 report['loaded_paths']=[v for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(s in v for s in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so','_infinilm.cpython'])];report['completed']=True
except BaseException as e:report['error']=repr(e);save();raise
finally:
 try:check(lib.deep_smallt_packed_mode_r9228(0));check(lib.modern_deep_latent_shutdown());report['normal']=True
 except BaseException as e:report['shutdown_error']=repr(e);save();raise
 finally:save()
