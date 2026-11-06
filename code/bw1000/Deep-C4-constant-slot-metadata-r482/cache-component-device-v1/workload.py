"""Frozen real-cache workload; arrays are finite packed E4M3 bytes, not weights."""
from pathlib import Path
import hashlib,time,json
import numpy as np
ORDER=[0,1,1,0,0,1] # Three complete paired units AB / BA / AB.
NAMES=['prefix1','prefix2124','prefix8192','reverse8192','page_edges_duplicates','reset_reseed_prefix2124']
H=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def values(layer,epoch):
 slot=np.arange(8192,dtype=np.uint64)[:,None];channel=np.arange(128,dtype=np.uint64)[None,:]
 v=(((slot*17+channel*13+layer*7+epoch*29)%120)|(((slot+channel+layer+epoch)%2)<<7)).astype(np.uint8)
 scales=(np.float32(1)+(np.arange(8192,dtype=np.uint32)+layer+epoch*3)%31/np.float32(32)).astype(np.float32).reshape(8192,1)
 assert not np.any((v&127)==127)and np.isfinite(scales).all()and (scales>0).all();return v,scales
def raw_expected(v,s,written):
 out=np.full((128,8448),0xa5,np.uint8)
 for page in range(128):
  n=max(0,min(64,written-page*64))
  if n:out[page,:n*128]=v[page*64:page*64+n].reshape(-1);out[page,8192:8192+n*4]=s[page*64:page*64+n].view(np.uint8).reshape(-1)
 return out.reshape(-1)
def ids(name):
 if name=='prefix1':return np.arange(1,dtype=np.int64)
 if name in ['prefix2124','reset_reseed_prefix2124']:return np.arange(2124,dtype=np.int64)
 if name=='prefix8192':return np.arange(8192,dtype=np.int64)
 if name=='reverse8192':return np.arange(8191,-1,-1,dtype=np.int64).copy()
 if name=='page_edges_duplicates':return np.array([0,63,64,127,128,4095,4096,8191,0,8191],np.int64)
 raise ValueError(name)
def run_phase(api,mode,directory,phase,clock=lambda:None,save=lambda x:None):
 d=Path(directory);d.mkdir(exist_ok=False);q={'mode':mode,'phase':phase,'completed':False,'passed':False,'planned_measured_gathers':126,'planned_warm_gathers':21,'measured':[],'warm':[],'input_pins':[],'raw_audits':[],'selected_raw':[],'guards_checked':0,'reset_checked':0,'seed_calls':0};start=time.monotonic();data={i:values(i,0)for i in range(21)}
 def checkpoint():clock();save(q)
 def audit(layer,expected,label):
  q['current_operation']={'kind':'raw_audit','layer':layer,'label':label};clock();actual=api.raw(layer);assert actual.dtype==np.uint8 and actual.size==128*8448 and np.array_equal(actual,expected),('rawcache/untouched state',label,layer);q['raw_audits'].append({'layer':layer,'label':label,'bytes':actual.nbytes,'SHA256':H(actual)});checkpoint()
 def seed(layer,begin,n,epoch,label):
  q['current_operation']={'kind':'seed','layer':layer,'label':label};clock();v,s=data[layer];vp,sp=v[begin:begin+n],s[begin:begin+n];before=(H(vp),H(sp));binding=json.loads((Path(__file__).parent/'INPUT_BINDING.json').read_bytes());pin=next(a for a in binding['input_pins']if(a['layer'],a['epoch'],a['count'])==(layer,epoch,n));assert before==(pin['values_SHA256'],pin['scales_SHA256']);api.seed(layer,begin,vp,sp);assert (H(vp),H(sp))==before;q['input_pins'].append({'layer':layer,'epoch':epoch,'begin':begin,'count':n,'values_SHA256':before[0],'scales_SHA256':before[1],'label':label});q['seed_calls']+=1;checkpoint()
 def check_guard(layer,kind):q['current_operation']={'kind':'guard','layer':layer,'guard':kind};clock();api.guard(layer,kind);q['guards_checked']+=1;checkpoint()
 for layer in range(21):
  audit(layer,np.full(128*8448,0xa5,np.uint8),'constructor_A5')
  for kind in range(5):check_guard(layer,kind)
  seed(layer,0,2124,0,'partial2124');audit(layer,raw_expected(*data[layer],2124),'partial_untouched');check_guard(layer,4)
  seed(layer,0,8192,0,'full8192');audit(layer,raw_expected(*data[layer],8192),'full_seed')
 def gather(layer,name,warm=False):
  q['current_operation']={'kind':'warm_gather'if warm else'measured_gather','layer':layer,'case':name};clock();idx=ids(name);v,s=data[layer];before=api.observe();t=time.monotonic();out,scale,seconds=api.gather(layer,idx);elapsed=time.monotonic()-t
  assert np.array_equal(out,v[idx])and np.array_equal(scale.view(np.uint32),s[idx].view(np.uint32)),('actualGPU gather address/value/scale',layer,name)
  after=api.observe();delta=[after[9+i]-before[9+i]for i in range(5)];prefix=name in ['prefix1','prefix2124','prefix8192','reset_reseed_prefix2124'];expected=[0,1,int(bool(mode and prefix)),idx.size*8 if mode and prefix else 0,int(not(mode and prefix))];assert delta==expected,('actual metadata path counters',delta,expected)
  item={'layer':layer,'case':name,'count':int(idx.size),'IDs_SHA256':H(idx),'values_SHA256':H(out),'scales_SHA256':H(scale),'seconds':seconds,'Python_API_wall_s':elapsed,'counter_delta':delta,'GPU_values_scales_and_CPU_canaries_passed':True}
  q['warm'if warm else'measured'].append(item);q['current_operation']={'kind':'idle_after_confirmed_gather'}
  if not warm and layer in [0,20]and name in ['prefix8192','reset_reseed_prefix2124']:
   for field,a in [('values',out),('scales',scale)]:
    p=d/('phase%d-layer%d-%s.%s.bin'%(phase,layer,name,field));a.tofile(p);q['selected_raw'].append({'name':p.name,'path':str(p),'phase':phase,'layer':layer,'case':name,'field':field,'dtype':str(a.dtype),'shape':list(a.shape),'bytes':a.nbytes,'SHA256':H(a)})
  checkpoint()
 for layer in range(21):gather(layer,'prefix2124',warm=True);audit(layer,raw_expected(*data[layer],8192),'after_warm')
 for name in NAMES:
  if name=='reset_reseed_prefix2124':
   for layer in range(21):
    prior=raw_expected(*data[layer],8192);api.reset(layer);q['reset_checked']+=1;audit(layer,prior,'reset_does_not_write_raw');check_guard(layer,4);data[layer]=values(layer,1);seed(layer,0,8192,1,'reset_reseed8192');audit(layer,raw_expected(*data[layer],8192),'reset_rewritten')
  for layer in range(21):gather(layer,name);audit(layer,raw_expected(*data[layer],8192),'after_'+name)
 obs=api.observe();assert obs[2:9]==[21,128,128*8448,63,147,147,21],obs
 expected=[1376256 if mode else 0,147,105 if mode else 0,2446920 if mode else 0,42 if mode else 147];assert obs[9:14]==expected,(obs[9:14],expected)
 assert len(q['measured'])==126 and len(q['warm'])==21 and q['guards_checked']==147 and q['reset_checked']==21 and q['seed_calls']==63 and len(q['selected_raw'])==8
 q.update(completed=True,passed=True,observation=obs,elapsed_schedule_s=time.monotonic()-start,raw_audit_count=len(q['raw_audits']),primary_case='prefix2124',primary_gather_enclosure_s=sum(x['seconds'][0]for x in q['measured']if x['case']=='prefix2124'),primary_API_wall_s=sum(x['Python_API_wall_s']for x in q['measured']if x['case']=='prefix2124'))
 save(q);return q
