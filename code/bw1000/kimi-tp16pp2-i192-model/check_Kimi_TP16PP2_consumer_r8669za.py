"""Run the actual new forward consumer with a CPU fake CAPI; no GPU/library import."""
import ast,pathlib,json,ctypes,time,hashlib,numpy as np
r=pathlib.Path('.');p=r/'kimi_TP16PP2_I192_protocol_r8669za.py';tree=ast.parse(p.read_text());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run');f=next(x for x in run.body if isinstance(x,ast.FunctionDef)and x.name=='forward');f.body=[x for x in f.body if not isinstance(x,ast.Nonlocal)];mod=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]));cases=[]
for node in range(4):
 for candidate in [False,True]:
  for N in [8,16]:
   for token_only in [False,True]:
    counters=[[0,0]for _ in range(8)];stage=node//2;moe=45 if stage==0 else 47;tail=stage==1;emitter=2;vocab=32;calls=[]
    class Lib:
     def modern_kimi_vocab_head_get_r8341(self):return 0
     def modern_kimi_engine_forward_r2475(self,ctx,tokens,B,flags,*ptrs):
      assert flags==(7 if token_only else 3)and not flags&64;assert tokens==B==N
      calls.append(flags)
      out=np.ctypeslib.as_array(ctypes.cast(ptrs[7],ctypes.POINTER(ctypes.c_int64)),shape=(B,));out[:]=1
      if not token_only:
       z=np.ctypeslib.as_array(ctypes.cast(ptrs[6],ctypes.POINTER(ctypes.c_uint16)),shape=(B*vocab,)).reshape(B,vocab);z[:]=0;z[:,1]=0x3f80
      ctypes.cast(ptrs[8],ctypes.POINTER(ctypes.c_int32))[0]=(int(tail)if not token_only else 0)+(2 if node==emitter else 0)
      if candidate and N==16:
       for c in counters:c[0]+=moe;c[1]+=16*moe
      return 0
    def check(code):assert code==0
    g=dict(lib=Lib(),ctx=None,np=np,ctypes=ctypes,time=time,hashlib=hashlib,seq=0,counts=lambda:[[0]*4 for _ in range(8)],custom_counts=lambda:[list(v)for v in counters],candidate=candidate,policy=1,moe=moe,g={'memory':lambda:[(8*1024**3,64*1024**3)]*8},report={'counter_dispatch':[],'local_forward_samples':[]},vocab=vocab,check=check,tail=tail,node=node,emitter=emitter,pipeline=True)
    exec(compile(mod,str(p),'exec'),g);selected,seconds,digest=g['forward']([[1]]*N,[512]*N,list(range(1,N+1)),'CPU-case',token_only,True);assert len(calls)==1;assert(selected==[1]*N)if node==emitter else selected is None;assert len(g['report']['counter_dispatch'])==1;cases.append({'node':node,'candidate':candidate,'N':N,'flags':calls[0],'counter':g['report']['counter_dispatch'][0]['I192_expected']})
out={'passed':True,'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'ActualconsumerCPUfakeCAPI32cases/flags3or7/no64/TP16stage45or47/customhitN16/otherNfallback/tailnode2emitter/threepublishedflags/actualhead0; notGPU/Model/performance','cases':cases};(r/'Kimi-TP16PP2-consumer-CPU-r8669za.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(cases))
