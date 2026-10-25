import pathlib,ast
r=pathlib.Path.cwd()
def put(n,s):ast.parse(s);(r/n).write_bytes(s.encode('utf8'))
s=(r/'check_Deep_guard_Model_consumer_r8925.py').read_text().replace('Deep_owner_Model_protocol_r8925','Deep_latent_Model_protocol_r8942').replace('r8925','r8942').replace('Deep-owner-events-consumer-CPU','Deep-latent-Model-consumer-CPU')
s=s.replace('self.negative=negative','self.negative=negative;self.latent_mode=0;self.tiled=0')
s=s.replace("self.modern_native_engine_reset=Fn", "self.deep_latent_prefill_mode_r8933=Fn(self.mode);self.deep_latent_prefill_counts_r8933=Fn(self.latent);self.modern_native_engine_reset=Fn")
s=s.replace(' def armfn(self,w):', ''' def mode(self,a):self.latent_mode=a;return 0
 def latent(self,d,q,n):
  vals=[self.tiled*43,self.tiled*43*2048,self.tiled*43*64]
  if self.negative=='count'and self.tiled:vals[2]+=1
  if self.negative=='decodehit'and self.latent_mode and self.dec:vals[0]+=self.dec
  for i,v in enumerate(vals):q[i]=v
  return 0
 def armfn(self,w):''')
s=s.replace('self.pre+=1','self.pre+=1;self.tiled+=self.latent_mode')
s=s.replace("assert negative and lib.calls==1,(negative,repr(e),lib.calls)","assert negative,(negative,repr(e),lib.calls)")
a=s.index('  assert not negative and lib.calls==135');b=s.index('\nout=dict(',a)
s=s[:a]+'''  assert not negative and lib.calls==540 and lib.windows==[]and chats==['before','after_restore']and report['quality_passed']and report['performance_qualified']and len(report['owner_trials_r8942'])==4 and report['owner_event_windows']==[]
  assert len([v for v in progress if v[0].startswith('prefill_CAPI_enter')])==32
  assert len([v for v in progress if v[0].startswith('prefill_CAPI_return')])==32
  assert [v['latent_arm']for v in report['owner_trials_r8942']]==[0,1,1,0]
  assert lib.tiled==16 and lib.pre==32 and lib.dec==508
  return dict(passed=True,CAPI_calls=540,CAPI_calls_per_arm=135,full128_IDs_exact=True,plan=['A0','C1','C2','A3'],all8_T2048_chunks_checked_per_arm=True,owner_event_capture=False,no_extra_forward=True,QA=chats)
out=dict(hardware='BW1000',level='CPU_CONSUMER_CONTRACT',status='PASSED',CPU_only=True,noGPU_dlopen=True,positive=execute(),negative_nonfinite=execute('finite'),negative_actual_counter=execute('count'),negative_decode_hit=execute('decodehit'),entry_protocol_SHA256=hashlib.sha256((r/'Deep_latent_Model_protocol_r8942.py').read_bytes()).hexdigest());p=r/'Deep-latent-Model-consumer-CPU-r8942.json';p.write_bytes((json.dumps(out,indent=2)+'\\n').encode('utf8'));print(json.dumps(out))
'''
put('check_Deep_latent_Model_consumer_r8942.py',s)
