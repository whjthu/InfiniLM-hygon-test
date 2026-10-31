"""Shared same-residency owner evidence. Buffered HTTP is not client token latency."""
import pathlib,json,time,hashlib,math
from source_pin_r281 import verify
D=pathlib.Path(__file__).resolve().parent
NATIVE_SHA='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
COMMIT='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
def bind_identity(path,expected,service_report):
 verify(expected);p=pathlib.Path(path);q=json.loads(p.read_bytes());assert q['Native_SHA256']==NATIVE_SHA and(q['TP'],q['PP'],q['cards'],q['rows'],q['capacity'])==(8,1,8,1,32768);assert q['source_manifest_SHA256']==expected and q['prepared_entry_identity']['upstream_commit']==COMMIT
 assert pathlib.Path(service_report).name=='Deep-latest-full-owner-r281-'+q['actual_step']+'.json'
 return q

def bound_report(path,args):
 q=json.loads(pathlib.Path(path).read_bytes());assert q['actual_step']==args.expected_step and q['Native_SHA256']==NATIVE_SHA and q['source_manifest_SHA256']==args.manifest_sha
 return q

def row(path,args,rid,index,session,budget,remaining):
 end=time.monotonic()+8
 while time.monotonic()<end:
  remaining();rows=[r for r in bound_report(path,args)['requests']if r['request_id']==rid];assert len(rows)<=1
  if rows and rows[0].get('normal_reset_after')and rows[0].get('final_metadata'):break
  time.sleep(.05)
 else:raise RuntimeError('Completed same-residency owner receipt unavailable')
 r=rows[0];a=r['admission'];h=a['history_render_audit'];assert r['admitted_index']==index and a['diagnostic_user']==session and r['max_new_tokens']==budget and r['normal_reset_before']and r['normal_reset_after'];assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native'];assert r['prompt_tokens']==h['prompt_tokens']and r['prompt_tokens']+budget<=32768
 ids=r['generated_token_ids'];assert ids and len(ids)==r['actual_generated_tokens']and all(type(v)is int and 0<=v<129280 for v in ids);assert sum(v['tokens']for v in r['Native_continuation']if v['kind']=='prefill')==r['prompt_tokens'];assert r['native_finish_reason']in ['eos_token','length']
 if r['native_finish_reason']=='eos_token':assert ids[-1]==1
 else:assert len(ids)==budget and ids[-1]!=1
 ticks=r['backend_token_monotonic'];assert len(ticks)==len(ids)and all(math.isfinite(v)for v in ticks)and all(x<=y for x,y in zip(ticks,ticks[1:]))and a['accepted_monotonic']<=r['started_monotonic']<=ticks[0]<=ticks[-1]<=r['ended_monotonic'];return r

def percentile(v,p):
 if not v:return None
 v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)

def backend_metrics(r):
 ticks=[t for t,i in zip(r['backend_token_monotonic'],r['generated_token_ids'])if i!=1];n=len(ticks);gaps=[b-a for a,b in zip(ticks,ticks[1:])];a=r['admission']['accepted_monotonic'];s=r['started_monotonic'];return {'backend_emitter_visible_tokens':n,'backend_emitter_decode_tok_s':(n-1)/(ticks[-1]-ticks[0])if n>1 and ticks[-1]>ticks[0]else None,'backend_emitter_ITL_p95_s':percentile(gaps,.95),'backend_emitter_ITL_p99_s':percentile(gaps,.99),'queue_s':s-a,'backend_emitter_first_wait_from_admission_s':ticks[0]-a if n else None,'backend_emitter_first_wait_from_Native_start_s':ticks[0]-s if n else None,'Native_forward_host_summary':r['Native_forward_host_summary'],'client_incremental_metrics_qualified':False,'not_GPU_busy_or_kernel_attribution':True}

def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return[mid-half,mid+half]
