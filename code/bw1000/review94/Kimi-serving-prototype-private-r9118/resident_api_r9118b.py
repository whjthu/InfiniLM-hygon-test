"""Optional localhost OpenAI-compatible SSE subset; native work only on owner main thread.
Call next_request() at complete case/transaction boundaries; emit_tokens() receives ONLY
verified committed IDs after leader collect. finish_request() follows native commit cleanup.
No file polling is advertised as streaming. No sampling or continuous batching claim.
"""
import json,queue,threading,time,uuid,re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from incremental_text_r9118 import IncrementalTextDecoder
class ResidentAPI:
    def __init__(self,tokenizer,*,encode_messages=None,eos_ids=(),pad_ids=(),queue_capacity=8,max_prompt_tokens=512,max_tokens=256,min_tokens=2,request_timeout=900,model='infinilm-kimi-k3',port=0):
        assert queue_capacity>0 and 1<=min_tokens<=max_tokens and max_prompt_tokens<=512
        self.tokenizer=tokenizer;self.encoder=encode_messages or(lambda messages:tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True));self.eos_ids=set(eos_ids);self.pad_ids=set(pad_ids);self.jobs=queue.Queue(maxsize=queue_capacity);self.max_prompt_tokens=max_prompt_tokens;self.max_tokens=max_tokens;self.min_tokens=min_tokens;self.request_timeout=request_timeout;self.model=model;self.owner=threading.get_ident();self.active=None;self.records={};self.lock=threading.Lock();self.closed=False
        outer=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args):pass
            def error(self,status,message):
                b=json.dumps({'error':{'message':message,'type':'invalid_request_error'if status==400 else 'service_error'}}).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.send_header('Connection','close');self.end_headers();self.wfile.write(b);self.wfile.flush()
            def do_POST(self):
                if self.path!='/v1/chat/completions':self.error(404,'unknown endpoint');return
                self.connection.settimeout(30);task=None;headers=False
                try:
                    if self.headers.get('Transfer-Encoding'):raise ValueError('chunked request bodies unsupported')
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=65536:raise ValueError('request body must be 1..65536 bytes')
                    data=json.loads(self.rfile.read(n));task=outer.validate(data)
                    with outer.lock:
                        if outer.closed:self.error(503,'server draining');return
                        if task['id']in outer.records:raise ValueError('duplicate request_id')
                        try:outer.jobs.put_nowait(task)
                        except queue.Full:self.error(503,'request queue full');return
                        outer.records[task['id']]=task
                    self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers();headers=True
                    self.wfile.write(('data: '+json.dumps({'id':task['id'],'object':'chat.completion.chunk','model':outer.model,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})+'\n\n').encode());self.wfile.flush()
                    while True:
                        left=task['deadline']-time.perf_counter()
                        if left<=0:raise TimeoutError('request deadline')
                        obj=task['output'].get(timeout=left)
                        if obj is None:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();task['transport_done']=True;break
                        self.wfile.write(('data: '+json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n\n').encode());self.wfile.flush()
                except queue.Empty:
                    # The producer may still be loading/prefilling: resume until deadline.
                    if task is not None:outer.transport_error(task,'transport wait expired')
                except (BrokenPipeError,ConnectionResetError,OSError)as e:
                    if task is not None:outer.transport_error(task,'client disconnected: '+type(e).__name__)
                except (ValueError,KeyError,TypeError)as e:
                    if not headers:self.error(400,str(e))
                    elif task is not None:outer.transport_error(task,str(e))
                except Exception as e:
                    if not headers:self.error(500,str(e))
                    elif task is not None:outer.transport_error(task,repr(e))
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler);self.server.daemon_threads=True;self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url='http://127.0.0.1:'+str(self.server.server_port)+'/v1/chat/completions'
    def validate(self,data):
        if not isinstance(data,dict):raise ValueError('JSON object required')
        allowed={'model','messages','max_tokens','temperature','top_p','n','stream','request_id'}
        if set(data)-allowed:raise ValueError('unsupported fields: '+','.join(sorted(set(data)-allowed)))
        if data.get('model',self.model)!=self.model:raise ValueError('unknown model')
        if data.get('stream')is not True:raise ValueError('stream=true required by this prototype')
        if data.get('temperature',0)!=0 or data.get('top_p',1)!=1 or data.get('n',1)!=1:raise ValueError('only greedy temperature=0,top_p=1,n=1 supported')
        cap=data.get('max_tokens',self.max_tokens)
        if isinstance(cap,bool)or not isinstance(cap,int)or not self.min_tokens<=cap<=self.max_tokens:raise ValueError('max_tokens outside configured range')
        messages=data.get('messages')
        if not isinstance(messages,list)or len(messages)!=1 or not isinstance(messages[0],dict)or set(messages[0])!={'role','content'}or messages[0]['role']!='user'or not isinstance(messages[0]['content'],str)or not messages[0]['content']:raise ValueError('one nonempty plain user message required')
        ids=list(self.encoder(messages))
        if not 1<=len(ids)<=self.max_prompt_tokens or len(ids)+cap>1024:raise ValueError('prompt or prompt+output exceeds native capacity; no truncation')
        if any(isinstance(t,bool)or not isinstance(t,int)or t<0 for t in ids):raise ValueError('invalid encoded token IDs')
        rid=data.get('request_id',uuid.uuid4().hex)
        if not isinstance(rid,str)or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',rid):raise ValueError('invalid request_id')
        now=time.perf_counter();return {'id':rid,'input_ids':ids,'max_tokens':cap,'arrival_server':now,'deadline':now+self.request_timeout,'output':queue.Queue(maxsize=cap+16),'cancelled':threading.Event(),'state':'queued','events':[],'transport_done':False,'prompt':messages[0]['content']}
    def transport_error(self,task,error):
        task['cancelled'].set();task['transport_error']=error
    def assert_owner(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('native adapter operations require owner main thread')
    def next_request(self,timeout=.1):
        self.assert_owner()
        if self.active is not None:raise RuntimeError('previous native transaction not finished')
        end=time.perf_counter()+max(0,timeout)
        while True:
            try:task=self.jobs.get(timeout=max(0,end-time.perf_counter()))
            except queue.Empty:return None
            if task['cancelled'].is_set()or time.perf_counter()>=task['deadline']:task['state']='cancelled_before_start';continue
            task['state']='active';task['started']=time.perf_counter();task['decoder']=IncrementalTextDecoder(self.tokenizer,self.eos_ids,self.pad_ids);task['useful_tokens']=0;self.active=task;return {'id':task['id'],'input_ids':task['input_ids'].copy(),'max_tokens':task['max_tokens']}
    def to_case(self,request,*,mode='adaptive',draft='dflash2',K=7):
        if mode not in ('ar','oldspec','state','adaptive')or draft not in ('dflash2','modal')or K not in (2,4,7):raise ValueError('unsupported resident case policy')
        return {'schema':1,'action':'case','id':request['id'],'mode':mode,'draft':draft,'K':0 if mode=='ar'else K,'cap':request['max_tokens'],'input_ids':request['input_ids'].copy()}
    def put(self,task,event):
        if task['cancelled'].is_set():return False
        try:task['output'].put_nowait(event);return True
        except queue.Full:self.transport_error(task,'stream buffer full');return False
    def emit_tokens(self,request_id,token_ids,*,kind='committed',final=False):
        self.assert_owner();task=self.active
        if task is None or task['id']!=request_id:raise RuntimeError('no matching active request')
        if kind in ('draft','replay','pad'):return not task['cancelled'].is_set()
        for i,token in enumerate(token_ids):
            eos=int(token)in self.eos_ids;piece,count=task['decoder'].push(token,kind=kind,eos=eos,final=final and i==len(token_ids)-1)
            now=time.perf_counter()
            if count:
                task['useful_tokens']+=1
                if task['useful_tokens']>task['max_tokens']:raise RuntimeError('native adapter exceeded output cap')
                task['events'].append({'time':now,'token_id':int(token),'kind':'committed','visible':True})
            if count or piece:self.put(task,{'id':request_id,'object':'chat.completion.chunk','model':self.model,'server_ready':now,'server_queue_seconds':task['started']-task['arrival_server'],'text_only':not count,**({'token_id':int(token)}if count else{}),'choices':[{'index':0,'delta':{'content':piece},'finish_reason':None}]})
            if eos:break
        return not task['cancelled'].is_set()
    def finish_request(self,request_id,reason='stop',error=None):
        self.assert_owner();task=self.active
        if task is None or task['id']!=request_id:raise RuntimeError('no matching active request')
        if reason not in ('stop','length','error','cancelled'):raise ValueError('invalid finish reason')
        tail=task['decoder'].finish()
        if tail:self.put(task,{'id':request_id,'text_only':True,'choices':[{'index':0,'delta':{'content':tail},'finish_reason':None}]})
        if error:self.put(task,{'error':{'message':str(error),'type':'generation_error'}})
        self.put(task,{'id':request_id,'object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':reason}],'usage':{'completion_tokens':task['useful_tokens']}});self.put(task,None);task.update(state='cancelled'if task['cancelled'].is_set()else'failed'if error else'completed',finished=time.perf_counter(),stop_reason=reason);self.active=None
    def close(self):
        self.assert_owner()
        if self.active is not None:raise RuntimeError('finish native transaction before server shutdown')
        with self.lock:self.closed=True
        while True:
            try:task=self.jobs.get_nowait()
            except queue.Empty:break
            task['state']='cancelled_shutdown';self.transport_error(task,'server shutdown')
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
