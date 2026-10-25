"""Real HTTP/SSE client concurrency driver. Does not start a model/server."""
import concurrent.futures,json,time,random,threading,urllib.request,urllib.error
from quality_bench_r9100 import serving_metrics,describe

def run_load(endpoints,scenario,prompts,cards,max_tokens=64,seed=20261026,deadline_seconds=300):
 assert endpoints and cards>0 and prompts;count=scenario['requests'];C=scenario.get('concurrency',1);assert 1<=C<=64
 start=time.perf_counter();deadline=start+deadline_seconds;rng=random.Random(seed);offsets=[0.]*count;poisson=scenario['kind']=='poisson'
 if poisson:
  for i in range(1,count):offsets[i]=offsets[i-1]+rng.expovariate(scenario['request_rate'])
 records=[{'id':scenario['name']+'-'+str(i),'prompt':prompts[i%len(prompts)]['prompt'],'source_id':prompts[i%len(prompts)]['id'],'planned_offset':offsets[i],'arrival':None,'status':'planned','events':[],'text_events':[],'text':''}for i in range(count)]
 def client(rec):
  wait=start+rec['planned_offset']-time.perf_counter()if poisson else 0
  if wait>0:threading.Event().wait(min(wait,max(0,deadline-time.perf_counter())))
  rec['arrival']=time.perf_counter()
  if rec['arrival']>=deadline:rec.update(status='not_sent_before_deadline',finish=time.perf_counter());return
  rec['status']='inflight';endpoint=endpoints[int(rec['id'].rsplit('-',1)[1])%len(endpoints)];rec['endpoint']=endpoint
  body={'model':scenario.get('model','infinilm-kimi-k3'),'request_id':rec['id'],'messages':[{'role':'user','content':rec['prompt']}],'max_tokens':max_tokens,'temperature':0,'stream':True};req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=max(.1,deadline-time.perf_counter()))as response:
    rec['HTTP_status']=response.status
    for raw in response:
     if time.perf_counter()>=deadline:rec['status']='deadline';break
     if not raw.startswith(b'data: '):continue
     raw=raw[6:].strip()
     if raw==b'[DONE]':rec['status']='completed'if rec.get('stop_reason')in ('stop','length','EOS')else'failed';break
     e=json.loads(raw)
     if 'error'in e:rec.update(status='failed',error=e['error']);continue
     c=e['choices'][0];piece=c.get('delta',{}).get('content','');now=time.perf_counter()
     if piece:rec['text']+=piece;rec['text_events'].append({'time':now,'text':piece})
     if c.get('finish_reason'):rec['stop_reason']=c['finish_reason'];continue
     if 'token_id'in e and not e.get('text_only',False):rec['events'].append({'time':now,'kind':'committed','visible':True,'token_id':int(e['token_id']),'server_ready':e.get('server_ready')})
     if 'server_queue_seconds'in e:rec['server_queue_seconds']=e['server_queue_seconds']
    if rec['status']=='inflight':rec['status']='connection_closed_unfinished'
  except urllib.error.HTTPError as e:rec.update(status='rejected',HTTP_status=e.code,error=e.read().decode(errors='replace'))
  except Exception as e:rec.update(status='failed',error=repr(e))
  rec['finish']=time.perf_counter()
 with concurrent.futures.ThreadPoolExecutor(max_workers=count if poisson else C)as pool:
  futures=[pool.submit(client,r)for r in records]
  for f in futures:f.result()
 end=time.perf_counter()
 for r in records:
  stamps=[e['time']for e in r['events']];r['user_decode_tok_s']=(len(stamps)-1)/(stamps[-1]-stamps[0])if len(stamps)>1 and stamps[-1]>stamps[0]else None;r['TTFT_s']=stamps[0]-r['arrival']if stamps else None;r['first_nonempty_text_TTFT_s']=r['text_events'][0]['time']-r['arrival']if r['text_events']else None;r['ITL_s']=[b-a for a,b in zip(stamps,stamps[1:])];r['visible_token_count']=len(stamps);r['short_output_below32']=len(stamps)<32
 transitions=sorted([(r['arrival'],1)for r in records if r['status']!='not_sent_before_deadline']+[(r['finish'],-1)for r in records if r['status']!='not_sent_before_deadline'],key=lambda z:(z[0],z[1]));active=peak=0
 for stamp,delta in transitions:active+=delta;peak=max(peak,active)
 m=serving_metrics(records,start,end,cards);m['actual_peak_inflight_clients']=peak;m['first_nonempty_text_TTFT_s']=describe([r['first_nonempty_text_TTFT_s']for r in records if r['first_nonempty_text_TTFT_s']is not None]);m['server_queue_seconds']=describe([r['server_queue_seconds']for r in records if 'server_queue_seconds'in r]);m['queue_unobserved_requests']=sum('server_queue_seconds'not in r for r in records);m['scheduled_requests']=count;m['actually_sent_requests']=sum(r['status']!='not_sent_before_deadline'for r in records);m['short_output_below32_count']=sum(r['short_output_below32']for r in records)
 return {'scenario':scenario,'requests':records,'metrics':m,'native_scheduler':'B1serial unless separately bound nativebatch producer; client C is not native batch','clock':'single clientprocess perf_counter actualHTTPframes; no crossnode subtraction','Model_performance_qualified':False,'bind_native_endpoint_required':True}

def total_cards(topology):
 return sum(sum(int(r.get(k,0))for k in ('target_cards','draft_cards','aux_cards'))for r in topology['replicas'])

def main():
 import argparse,pathlib
 p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--topology',required=True);p.add_argument('--output',required=True);p.add_argument('--endpoints',nargs='+');p.add_argument('--scenario',action='append');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();trace=json.loads(pathlib.Path(a.trace).read_text());top=json.loads(pathlib.Path(a.topology).read_text());cards=total_cards(top);assert cards>0;out={'executed':False,'Model_performance_qualified':False,'topology':top,'all_service_cards':cards,'DP_accounting':'all target/draft/aux sets counted; scaling is not nativebatch efficiency','results':[]}
 if not a.prepare_only:
  assert a.endpoints and len(a.endpoints)==len(top['replicas']);out['executed']=True
  for sc in trace['scenarios']:
   if a.scenario and sc['name']not in a.scenario:continue
   out['results'].append(run_load(a.endpoints,sc,trace['prompts'],cards,trace['max_tokens'],trace['seed'],trace['scenario_deadline_seconds']))
 dest=pathlib.Path(a.output);assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
