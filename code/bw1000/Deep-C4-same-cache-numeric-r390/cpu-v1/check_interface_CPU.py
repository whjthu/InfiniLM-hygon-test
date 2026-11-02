"""Actual ctypes packet readers/consumer with Fake C ABI leaves, zero dlopen."""
import pathlib,importlib.util,ctypes as C,json,types
S=pathlib.Path(__file__).resolve().parent
def load(n,p):sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
fixtures=load('fixture390',S/'check_consumer_CPU.py');api=load('interface390',S/'interface.py');consumer=fixtures.consumer;oracle=fixtures.oracle
def check(rc):assert rc==0
calls=[]
def prepare(ctx,past,x,q,pos,stream,out):
 assert past==ctx['record']['identity_before']['past'];ctx['stage']='before';C.cast(out,C.POINTER(C.c_uint64))[0]=ctx['record']['identity_before']['generation'];calls.append('prepare');return 0
def score(ctx,token,mode):
 assert ctx['stage']==('before'if mode==0 else'after_sdk');ctx['stage']='after_sdk'if mode==0 else'after_paged';calls.append('score'+str(mode));return 0
def commit(ctx,token):assert ctx['stage']=='after_paged';ctx['stage']='post_commit';calls.append('commit');return 0
def identity(ctx,token):
 name={'before':'identity_before','after_sdk':'identity_after_sdk','after_paged':'identity_after_paged','post_commit':'identity_post_commit'}[ctx['stage']]
 assert token==ctx['record'][name]['generation'];return json.dumps(ctx['record'][name]).encode()
def info(ctx,token,which,shape,size,dtype):
 f=ctx['record'][ctx['stage']][which]
 if f is None:C.cast(size,C.POINTER(C.c_size_t))[0]=0;shape[0]=0;return 0
 shape[0]=len(f['shape'])
 for i,v in enumerate(f['shape']):shape[i+1]=v
 C.cast(size,C.POINTER(C.c_size_t))[0]=len(f['raw']);C.cast(dtype,C.POINTER(C.c_int))[0]=-1;return 0
def read(ctx,token,which,out,size):
 data=ctx['record'][ctx['stage']][which]['raw'];assert size==len(data);C.memmove(out,int(data.ctypes.data),size);return 0
lib=types.SimpleNamespace()
for n in ['error','id_size','id','create','keys','load','finish','reset','prefix','destroy','shutdown']:setattr(lib,'deep_c4_lease_'+n+'_r390',lambda *args:0)
for n,fn in [('prepare',prepare),('score',score),('commit',commit),('identity',identity),('proof_info',info),('proof_read',read)]:setattr(lib,'deep_c4_lease_'+n+'_r390',fn)
f=api.configure(lib);previous=None;checks=[]
for past in [8575,8576]:
 data=fixtures.fixtures(past,previous);packets=[];contexts=[];tokens=[]
 for row in data:
  ctx={'record':row,'stage':'before'};token,packet=api.begin(f,check,ctx,past,None,None,None,None);packet['rank']=row['rank'];api.score(f,check,ctx,token,0,packet);api.score(f,check,ctx,token,1,packet);contexts.append(ctx);tokens.append(token);packets.append(packet)
 assert calls.count('commit')==(0 if past==8575 else 8),'No commit before all8 scores checked'
 assert consumer.validate_same_cache(packets,oracle,previous,committed=False)['passed'];checks.append('actualCABIpacket_and_precommitconsumer_'+str(past))
 for ctx,token,packet in zip(contexts,tokens,packets):api.commit(f,check,ctx,token,packet)
 assert consumer.validate_same_cache(packets,oracle,previous)['passed'];checks.append('actualCABIpacket_postcommit_nextstate_'+str(past));previous=packets
assert calls.count('prepare')==calls.count('score0')==calls.count('score1')==calls.count('commit')==16
print(json.dumps(dict(passed=True,checks=checks,prepare_calls=16,SDK_replay_calls=16,paged_replay_calls=16,commit_calls=16,actual_interface_and_consumer=True,CABI_leaves_fake_only=True,Native_dlopen=False,GPU=False)))
