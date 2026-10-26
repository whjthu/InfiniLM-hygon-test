import os,pathlib,json,ctypes,time,statistics,hashlib,traceback,threading
import torch
r=pathlib.Path('/refactor');b=r/'candidates/GLM-ragged-B32-r9172';proof=json.loads((b/'CPU-build.json').read_text());fixture=json.loads((b/'fixture.json').read_text());V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
q={'completed':False,'scope':'B17/B31/B32 completeattentionhelper QK/scale392/Softmax/Cast/PV/workspace/copy/drain; syntheticuniqueResidentKV32x640 real681Pmetadata; notModel/taskquality','producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'fixture_scope':fixture['scope'],'fixture_SHA256':proof['fixture_SHA256'],'library_SHA256':proof['library_SHA256'],'source_SHA256':proof['source_SHA256'],'trials':[],'QA':[]};out=r/'results'/('GLM-ragged-B32-r9172-'+q['producer']+'.json');assert not out.exists();ctx=None
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save():t=out.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(out)
try:
 torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;assert proof['completed']and sha(proof['library'])==proof['library_SHA256'];assert sha(b/'fixture.json')==proof['fixture_SHA256'];assert all(sha(p)==h for p,h in proof['dependencies'].items());lib=ctypes.CDLL(proof['library']);lib.glm_ragged_create_r9135.argtypes=[I];lib.glm_ragged_create_r9135.restype=V;lib.glm_ragged_run_r9135.argtypes=[V,Z,ctypes.POINTER(Z),V,V,V,V,I];lib.glm_ragged_destroy_r9135.argtypes=[V,ctypes.POINTER(I)];lib.glm_ragged_error_r9135.restype=ctypes.c_char_p;lib.glm_ragged_report_r9135.restype=ctypes.c_char_p
 def ck(c):
  if c:raise RuntimeError((c,lib.glm_ragged_error_r9135().decode()))
 ctx=lib.glm_ragged_create_r9135(0);assert ctx,lib.glm_ragged_error_r9135();q['created']=True;q['node']=os.uname().nodename;q['free_before']=torch.cuda.mem_get_info()[0];save()
 # Wholelist rejection, originalContext remains usable. No submit on these paths.
 z=(Z*2)(0,0);assert lib.glm_ragged_run_r9135(ctx,1,z,None,None,None,None,1)!=0;z=(Z*2)(0,641);assert lib.glm_ragged_run_r9135(ctx,1,z,None,None,None,None,1)!=0;assert lib.glm_ragged_run_r9135(ctx,33,z,None,None,None,None,1)!=0;z=(Z*4)(0,20,0,30);assert lib.glm_ragged_run_r9135(ctx,2,z,None,None,None,None,1)!=0;z=(Z*2)(32,20);assert lib.glm_ragged_run_r9135(ctx,1,z,None,None,None,None,1)!=0;q['QA'].append({'preSubmit_badP0_P641_B33_duplicate_row_row32_rejected':True});lib.glm_ragged_Bcounts_r9136.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z];sentinel=(ctypes.c_uint64*68)(*([99]*68));assert lib.glm_ragged_Bcounts_r9136(0,sentinel,68)!=0 and list(sentinel)==[99]*68;q['QA'].append({'old_Bhist_n68_rejected_without_write':True})

 bad=[]
 def foreign():bad.append(lib.glm_ragged_run_r9135(ctx,0,None,None,None,None,None,1))
 t=threading.Thread(target=foreign);t.start();t.join();assert bad and bad[0]!=0;q['QA'].append({'foreign_owner_rejected':True})
 q['summaries']={}
 bankgen=torch.Generator(device='cpu').manual_seed(917200);tag=torch.arange(32).view(32,1,1,1).float()*.003
 keybank=(torch.randn((32,640,8,256),generator=bankgen)*.1+tag).to(torch.bfloat16).cuda();valuebank=(torch.randn((32,640,8,256),generator=bankgen)*.1+tag*2).to(torch.bfloat16).cuda();querybank=(torch.randn((32,8,256),generator=bankgen)*.1).to(torch.bfloat16).cuda();bank_original=[keybank.clone(),valuebank.clone()]
 cases=[('primary-B%d-mixed'%B,fixture['P_by_B'][str(B)],True)for B in [17,31,32]]+[('mature-B16',fixture['P_by_B']['16'],False),('B1-original601',[fixture['P_by_B']['16'][0]],False),('B8-equalP-oldcohort',[107]*8,False)]

 for ci,(label,P,primary)in enumerate(cases):
  assert len(set(P))>1 if 'mixed'in label or'tail'in label else True
  B=len(P);logical=fixture['logical_rows'][:B];query=querybank[logical].contiguous();key=keybank;value=valuebank;original=[query.clone(),*bank_original];guard=torch.full((B*8*256+128,),123,dtype=torch.bfloat16,device='cuda');result=guard[64:-64].view(B,8,256);meta=(Z*(2*B))(*[v for row,n in zip(logical,P)for v in [row,n]]);torch.cuda.synchronize();ref=None;rows=[];modes=[0,1,0,1,1,0,1,0,0,1,0]if primary else[0,1,0]

  # CPU-diagnostic reference usesTorchF32 QK/PV and explicit originalBF16 rounding; notactualtaskquality.
  expected=[];off=0
  for i,n in enumerate(P):
   Q=query[i].float();K=key[logical[i],:n].float().permute(1,2,0);VV=value[logical[i],:n].float().permute(1,0,2);dots=torch.bmm(Q[:,None,:],K).to(torch.bfloat16);scaled=(dots.float()/16).to(torch.bfloat16).float();prob=torch.softmax(scaled,-1).to(torch.bfloat16);expected.append(torch.bmm(prob.float(),VV).to(torch.bfloat16).squeeze(1));off+=n
  oracle=torch.stack(expected).cpu()
  for j,mode in enumerate(modes):
   guard.fill_(123);torch.cuda.synchronize();start=time.perf_counter();ck(lib.glm_ragged_run_r9135(ctx,B,meta,V(query.data_ptr()),V(key.data_ptr()),V(value.data_ptr()),V(result.data_ptr()),mode));driver_ms=(time.perf_counter()-start)*1000;rep=json.loads(lib.glm_ragged_report_r9135());actual=result.cpu();finite=bool(torch.isfinite(actual.float()).all());canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all());readonly=all(torch.equal(a.view(torch.uint8),z.view(torch.uint8))for a,z in zip([query,key,value],original));assert finite and canary and readonly
   if ref is None:ref=actual.clone()
   diff=actual.float()-ref.float();den=max(float(ref.float().square().mean().sqrt()),1e-30);rel=float(diff.square().mean().sqrt())/den;err=actual.float()-oracle.float();oracle_rel=float(err.square().mean().sqrt())/max(float(oracle.float().square().mean().sqrt()),1e-30);assert rel<=.01 and oracle_rel<=.02,('error',label,mode,rel,oracle_rel)
   route=3 if B==1 else 2 if len(set(P))==1 and B in[8,16]else 1 if mode and len(set(P))>1 else 0;assert rep['route']==route and rep['actual_branch_delta']==[int(i==route)for i in range(4)]
   row={**rep,'label':label,'index':j,'measurement':primary and 2<=j<10,'restore':j==len(modes)-1,'driver_full_CAPI_ms':driver_ms,'finite':finite,'canary':canary,'inputs_readonly':readonly,'relRMS_vs_baseline':rel,'relRMS_vs_F32diagnostic':oracle_rel,'maxabs_vs_baseline':float(diff.abs().max()),'output_SHA256':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),'free_after':torch.cuda.mem_get_info()[0]};rows.append(row);q['trials'].append(row);save()
  if primary:
   pairs=[(rows[a],rows[a+1])for a in [2,4,6,8]];ratios=[]
   for a,z in pairs:
    off,on=(a,z)if a['mode']==0 else(z,a);ratios.append(off['driver_full_CAPI_ms']/on['driver_full_CAPI_ms'])
   q['summaries'][label]={'label':label,'B':B,'P':P,'four_pair_driver_speedups':ratios,'medians':{str(mode):{m:statistics.median(t[m]for t in rows if t['measurement']and t['mode']==mode)for m in ['driver_full_CAPI_ms','whole_attention_native_ms']}for mode in [0,1]},'stable_allfourpairs_positive':all(v>1 for v in ratios),'notModelgain':True};save()
 expired=I();ck(lib.glm_ragged_destroy_r9135(ctx,ctypes.byref(expired)));ctx=None;assert expired.value==1;q['normal_Runtime_expired']=True;check=torch.ones(32,device='cuda');assert (check+1).sum().item()==64;torch.cuda.synchronize();q.update(completed=True,normal_close=True,Torch_postclose=True,free_postclose=torch.cuda.mem_get_info()[0]);save()
except BaseException as exc:q['error']=repr(exc);q['traceback']=traceback.format_exc();save();raise
finally:
 if ctx:
  expired=I();q['finally_destroy_code']=lib.glm_ragged_destroy_r9135(ctx,ctypes.byref(expired));q['finally_expired']=bool(expired.value);save()
