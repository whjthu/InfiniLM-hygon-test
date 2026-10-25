"""Local HTTP/SSE resident-model baseline. Model callback runs only on caller main thread.
run_batch(items, emit): emit(index,token_id,text_fragment,eos=False) once per real native token.
No native-model result is established by this module's CPU selftest.
"""
import concurrent.futures,json,queue,random,threading,time,urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from quality_bench_r9100 import serving_metrics

def run_local_service_benchmark(run_batch,prompts,max_batch=4,max_new_tokens=64,request_rate=.5,seed=20261026,deadline_seconds=300):
    assert prompts and 1<=max_batch<=32 and request_rate>0
    jobs=queue.Queue();wire=[];start=time.perf_counter();deadline=start+deadline_seconds
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path!='/v1/chat/completions':self.send_error(404);return
            try:
                n=int(self.headers.get('Content-Length','0'));assert 0<n<=1048576
                data=json.loads(self.rfile.read(n));messages=data['messages'];assert isinstance(messages,list)
                prompt=messages[-1]['content'];assert messages[-1]['role']=='user' and isinstance(prompt,str)
                cap=int(data.get('max_tokens',max_new_tokens));assert 0<cap<=max_new_tokens
                task={'id':data['request_id'],'prompt':prompt,'max_new_tokens':cap,'stream_queue':queue.Queue(),'arrival_server':time.perf_counter()};jobs.put(task)
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers()
                while True:
                    left=deadline-time.perf_counter()
                    if left<=0:raise TimeoutError('service deadline')
                    obj=task['stream_queue'].get(timeout=left)
                    if obj is None:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();break
                    self.wfile.write(('data: '+json.dumps(obj,ensure_ascii=False)+'\n\n').encode());self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):return
            except Exception as e:
                try:self.wfile.write(('data: '+json.dumps({'error':repr(e)})+'\n\n').encode());self.wfile.flush()
                except Exception:pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start();url='http://127.0.0.1:'+str(server.server_port)+'/v1/chat/completions'
    rng=random.Random(seed);arrivals=[0.]
    for _ in prompts[1:]:arrivals.append(arrivals[-1]+rng.expovariate(request_rate))
    records=[{'id':str(i),'planned_arrival_offset':a,'arrival':None,'status':'planned','events':[],'prompt':p}for i,(a,p)in enumerate(zip(arrivals,prompts))]
    def client(rec):
        wait=start+rec['planned_arrival_offset']-time.perf_counter()
        if wait>0:threading.Event().wait(min(wait,max(0,deadline-time.perf_counter())))
        rec['arrival']=time.perf_counter();rec['status']='inflight'
        body={'model':'resident-infinilm','request_id':rec['id'],'messages':[{'role':'user','content':rec['prompt']}],'max_tokens':max_new_tokens,'temperature':0,'stream':True}
        req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=max(.1,deadline-time.perf_counter())) as response:
                for raw in response:
                    if not raw.startswith(b'data: '):continue
                    raw=raw[6:].strip()
                    if raw==b'[DONE]':rec['status']='completed';break
                    event=json.loads(raw)
                    if 'error' in event:raise RuntimeError(event['error'])
                    choice=event['choices'][0]
                    if choice.get('finish_reason'):rec['stop_reason']=choice['finish_reason'];continue
                    rec['events'].append({'time':time.perf_counter(),'kind':'committed','visible':True,'token_id':event['token_id'],'text':choice['delta']['content'],'server_ready':event['server_ready']})
        except Exception as e:rec['status']='failed';rec['error']=repr(e)
        rec['finish']=time.perf_counter()
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts));futures=[pool.submit(client,rec)for rec in records]
    batches=[]
    try:
        while time.perf_counter()<deadline and not all(f.done() for f in futures):
            try:batch=[jobs.get(timeout=min(.1,max(.001,deadline-time.perf_counter())))]
            except queue.Empty:continue
            collect_until=min(deadline,time.perf_counter()+.02)
            while len(batch)<max_batch:
                try:batch.append(jobs.get(timeout=max(.0001,collect_until-time.perf_counter())))
                except queue.Empty:break
                if time.perf_counter()>=collect_until:break
            began=time.perf_counter();stopped=[False]*len(batch);emitted=[0]*len(batch)
            def emit(index,token_id,text_fragment,eos=False):
                assert 0<=index<len(batch)
                if stopped[index]:return
                if eos:stopped[index]=True;batch[index]['stream_queue'].put({'choices':[{'delta':{},'finish_reason':'stop'}]});return
                emitted[index]+=1
                batch[index]['stream_queue'].put({'object':'chat.completion.chunk','token_id':int(token_id),'server_ready':time.perf_counter(),'choices':[{'index':0,'delta':{'content':text_fragment},'finish_reason':None}]})
            try:
                run_batch(batch,emit)
                for i,task in enumerate(batch):
                    if not stopped[i]:task['stream_queue'].put({'choices':[{'delta':{},'finish_reason':'length'}]})
                    task['stream_queue'].put(None)
            except BaseException as e:
                for task in batch:task['stream_queue'].put({'error':repr(e)});task['stream_queue'].put(None)
                raise
            batches.append({'ids':[q['id']for q in batch],'start':began,'end':time.perf_counter(),'queue_seconds':[began-q['arrival_server']for q in batch],'emitted_counts':emitted})
        for f in futures:
            try:f.result(timeout=max(.01,deadline-time.perf_counter()))
            except concurrent.futures.TimeoutError:pass
    finally:
        server.shutdown();server.server_close();pool.shutdown(wait=False,cancel_futures=True)
    end=time.perf_counter()
    for rec in records:
        if rec['arrival'] is None:rec['arrival']=end;rec['status']='not_sent_before_deadline'
    metrics=serving_metrics(records,start,end,8)
    return {'scope':'ActuallocalhostHTTP/SSE caller must disclose modeladapter; cohortserialbaseline,notcontinuousbatching;distinctprompts/naturalEOS/maxcap','request_rate':request_rate,'arrival_seed':seed,'max_batch':max_batch,'max_new_tokens':max_new_tokens,'metrics':metrics,'requests':records,'batches':batches,'clock_scope':'ClientHTTPstreamingperf_counter,serverreadyseparate;includesqueueandtransport','production_qualification':False}

def selftest():
    def adapter(items,emit):
        for i,_ in enumerate(items):emit(i,10,'ok');emit(i,1,'',True)
    r=run_local_service_benchmark(adapter,['a','b','c'],request_rate=1000,deadline_seconds=5)
    assert r['metrics']['completed']==3 and r['metrics']['useful_visible_output_tokens']==3
    return {'passed':True,'CPU_only':True,'actualHTTP_SSE':True,'native_model_called':False,'metrics_not_model_performance':True}
if __name__=='__main__':print(json.dumps(selftest()))
