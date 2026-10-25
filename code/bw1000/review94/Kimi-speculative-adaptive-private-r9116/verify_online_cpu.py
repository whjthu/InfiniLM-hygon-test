"""Execute the actual four-node protocol over real loopback binary sockets.
Native model/draft math is mocked; no claim of GPU quality/performance.
"""
import pathlib,tempfile,threading,time,socket,os,json,sys,types,ctypes as C,traceback
import numpy as np
from wire import listen,recv,send
from target_protocol import run
class Hub:
    def __init__(self):self.cv=threading.Condition();self.data={}
    def collect(self,key,node,value):
        with self.cv:
            d=self.data.setdefault(key,{})
            if node in d:raise AssertionError(('duplicate',key,node))
            d[node]=value;self.cv.notify_all()
            if not self.cv.wait_for(lambda:len(d)==4,timeout=25):raise TimeoutError((key,list(d)))
            return[d[i]for i in range(4)]
class Fn:
    def __init__(self,f):self.f=f
    def __call__(self,*a):return self.f(*a)
def view(p,typ,n):return np.ctypeslib.as_array(C.cast(p,C.POINTER(typ)),shape=(n,))
class Native:
    def __init__(self,node):
        self.node=node;self.active=1;self.shadow=9;self.past=0;self.phase=0;self.generation=0;self.records=[];self.capture=False;self.retained=0;self.width=0;self.replays=0;self.state_on=False;self.state_replays=0;self.pair_AR_calls=0;self.prefixes={1:[],9:[]};self.last_ids=[];self.modal=False;self.boundary=False
        for name,method in [('state_mode_r9114','state_mode'),('state_replay_r9114','state_replay'),('state_stats_r9114','stats'),('live_taps_begin_r9112','begin6'),('live_taps_begin_r9046','begin'),('live_taps_count_r9046','count'),('live_taps_info_r9046','info'),('live_taps_copy_r9046','copy'),('pair_mode_r9007','mode'),('pair_rows_r9006','rows'),('pair_prefill_r9006','prefill'),('pair_forward_r9006','forward'),('pair_commit_r9006','commit'),('pair_ordinary_forward_r9007','ordinary')]:setattr(self,'modern_kimi_'+name,Fn(getattr(self,method)))
    def state_mode(self,ctx,enabled):
        assert self.phase in [0,1,4];self.state_on=bool(enabled);return 0
    def stats(self,ctx,p,n):view(p,C.c_uint64,n)[:]=[self.state_replays]*n;return 0
    def state_replay(self,ctx,row):
        assert self.state_on and self.phase==3;self.prefixes[self.shadow]=self.prefixes[self.active]+self.last_ids[:self.retained];self.past+=self.retained;self.active,self.shadow=self.shadow,self.active;self.phase=1;self.state_replays+=1;return 0
    def begin(self,ctx,epoch,on):self.modal=False;self.epoch=epoch;self.capture=on;self.records=[];return 0
    def begin6(self,ctx,epoch,on):self.modal=True;self.epoch=epoch;self.capture=on;self.records=[];return 0
    def count(self,ctx,p):C.cast(p,C.POINTER(C.c_size_t))[0]=len(self.records);return 0
    def info(self,ctx,index,p,n):
        layer,past,N,raw=self.records[index];view(p,C.c_int64,6)[:]=[self.epoch,layer,past,N,7168,len(raw)];return 0
    def copy(self,ctx,index,p,n):raw=self.records[index][3];assert len(raw)==n;C.memmove(p,raw,n);return 0
    def mode(self,ctx,ordinary,p):self.active,self.shadow,self.past,self.phase=1,9,0,0;self.prefixes={1:[],9:[]};self.boundary=False;self.generation+=1;C.cast(p,C.POINTER(C.c_uint64))[0]=self.generation;return 0
    def rows(self,ctx,row,p):view(p,C.c_int32,4)[:]=[self.active,self.shadow,self.past,self.phase];return 0
    def emit(self,ids,past,logits,out,flags,count):
        if past==0 and len(ids)==512 and ids[0]==0:self.boundary=True
        limit=126 if self.boundary else 127
        selected=[min(int(v)+1,limit)for v in ids]if count>1 else[min(int(ids[-1])+1,limit)];l=view(logits,C.c_uint16,count*128).reshape(count,128);l[:]=0
        for i,t in enumerate(selected):l[i,t]=0x3f80
        if self.node==2:view(out,C.c_int64,count)[:]=selected
        view(flags,C.c_int32,1)[0]=int(self.node>=2)+2*int(self.node==2)
        if self.capture:
            for layer in {0:[19,37],2:([54,66,78,90]if self.modal else[66,78,90])}.get(self.node,[]):self.records.append((layer,past,len(ids),bytes(len(ids)*7168*2)))
        return 0
    def ordinary(self,ctx,g,N,B,f,row,past,total,off,ids,pos,logits,out,flags):return self.emit(view(ids,C.c_int64,N),int(view(past,C.c_int32,1)[0]),logits,out,flags,1)
    def prefill(self,ctx,N,row,ids,pos,logits,out,flags):self.past=N;self.phase=1;self.prefixes[self.active]=list(view(ids,C.c_int64,N));return self.emit(view(ids,C.c_int64,N),0,logits,out,flags,1)
    def forward(self,ctx,N,B,f,read,write,row,past,total,off,ids,pos,logits,out,flags):
        old=int(view(past,C.c_int32,1)[0]);assert old+N<=1024;assert old==self.past and self.phase in[1,3]
        if self.phase==3:assert not self.state_on and N==self.retained;self.prefixes[self.shadow]=self.prefixes[self.active]+list(view(ids,C.c_int64,N));self.past+=N;self.active,self.shadow=self.shadow,self.active;self.phase=1;self.replays+=1
        else:
            assert len(self.prefixes[self.active])==old;self.last_ids=list(view(ids,C.c_int64,N));self.prefixes[self.shadow]=self.prefixes[self.active]+self.last_ids;self.phase=2;self.width=N
            if N==1:self.pair_AR_calls+=1
            elif self.state_on:time.sleep(.0002)
        return self.emit(view(ids,C.c_int64,N),old,logits,out,flags,N)
    def commit(self,ctx,row,a,continuation):
        assert self.phase==2 and 0<=a<self.width
        if not continuation:self.phase=4
        elif a+1<self.width:self.retained=a+1;self.phase=3
        else:self.past+=a+1;self.active,self.shadow=self.shadow,self.active;self.phase=1
        return 0
class Tok:
    def apply_chat_template(self,messages,**kw):self.answer='3'if'robe'in messages[0]['content']else'18';return[1,2,3,4]
    def decode(self,ids,**kw):return'#### '+self.answer
sys.modules['transformers']=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok()))
def port():
    with socket.socket()as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
p1,p2,p3=port(),port(),port();nonce='cpu-r9046';os.environ.update(KIMI_ENGINE_HEAD='127.0.0.1',KIMI_FEATURE_PORT_R9046=str(p1),KIMI_DRAFT_HOST_R9046='127.0.0.1',KIMI_DRAFT_PORT_R9046=str(p2),KIMI_DRAFT_NONCE_R9046=nonce,KIMI_MODAL_HOST_R9116='127.0.0.1',KIMI_MODAL_PORT_R9116=str(p3),KIMI_MODAL_NONCE_R9116=nonce)
server=listen('127.0.0.1',p2);draft_record={'cycles':0,'partial_proposals':0,'closed':False};errors=[]
def fake_draft(server,record,modal=False):
    try:
        conn,_=server.accept();h,b=recv(conn);assert h=={'cmd':'hello','nonce':nonce};send(conn,{'ready':True,'block_size':16 if modal else 8,'greedy_only':True});past=0
        while True:
            m,b=recv(conn)
            if m['cmd']=='close':send(conn,{'closed':True,'seq':m['seq']});record['closed']=True;break
            if m['cmd']=='reset':past=0;send(conn,{'reset':True,'seq':m['seq']});continue
            assert m['past']==past and len(b)==m['rows']*(43008 if modal else 35840)*2;past+=m['rows'];a=m['anchor'];K=m['K'];props=[min(a+i+1,127)for i in range(K)];record['cycles']+=1
            if record['cycles']%3==0 and props[0]<120:props[0]+=3;record['partial_proposals']+=1
            send(conn,{'seq':m['seq'],'proposals':props,'greedy_only':True,'draft_query_s':0.001,'append_H2D_s':0.0001})
        conn.close();server.close()
    except BaseException as e:errors.append(('draft',repr(e),traceback.format_exc()))
modalserver=listen('127.0.0.1',p3);modal_record={'cycles':0,'partial_proposals':0,'closed':False};t=threading.Thread(target=fake_draft,args=(server,draft_record),daemon=True);mt=threading.Thread(target=fake_draft,args=(modalserver,modal_record,True),daemon=True);t.start();mt.start();hub=Hub();reports={};natives={};start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='kimi-r9046-')as temp:
    r=pathlib.Path(temp);(r/'incoming').mkdir();(r/'results/cpuqueue').mkdir(parents=True);os.environ['KIMI_CASE_QUEUE_R9116']='cpuqueue'
    for i,strategy in enumerate(['adaptive','state','oldspec','ar']):(r/'results/cpuqueue'/('q%04d.json'%i)).write_text(json.dumps({'schema':1,'action':'case','id':'queued-'+strategy,'mode':strategy,'K':7,'cap':24,'input_ids':[8,9,10,11]}))
    for i,strategy in enumerate(['state','adaptive']):(r/'results/cpuqueue'/('q%04d.json'%(i+4))).write_text(json.dumps({'schema':1,'action':'case','id':'modal-'+strategy,'mode':strategy,'draft':'modal','K':7,'cap':24,'input_ids':[8,9,10,11]}))
    (r/'results/cpuqueue/q0006.json').write_text(json.dumps({'schema':1,'action':'case','id':'capacity512-state','mode':'state','K':7,'cap':512,'input_ids':[0]*512}))
    (r/'results/cpuqueue/q9999.json').write_text(json.dumps({'schema':1,'action':'stop'}))
    (r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').write_text(json.dumps({'natural_input_ids':[[1,2,3,4],[4,5,6,7]],'eos_token_ids':[127]}))
    def worker(node):
        try:
            lib=Native(node);natives[node]=lib;report={};control=types.SimpleNamespace(collect=lambda key,value:hub.collect(key,node,value));run({'lib':lib,'ctx':node+1,'r':r,'control':control,'report':report,'save':lambda:None,'node':node,'tail':node>=2,'emitter':2,'vocab':128,'model':r,'experiment_started_r9046':time.monotonic()});reports[node]=report
        except BaseException as e:errors.append((node,repr(e),traceback.format_exc()))
    threads=[threading.Thread(target=worker,args=(n,),daemon=True)for n in range(4)]
    for th in threads:th.start()
    for th in threads:th.join(timeout=45)
    t.join(timeout=3);mt.join(timeout=3)
    assert not errors,errors
    assert len(reports)==4 and draft_record['closed']and draft_record['partial_proposals']>0 and modal_record['closed']and modal_record['cycles']>0
    assert all(z['serving_protocol_complete']for z in reports.values())
    assert all(len(z['speculative_quality_cases'])==24 for z in reports.values())
    assert all(q['quality']['correct']for z in reports.values()for q in z['speculative_quality_cases']if'quality'in q)
    assert all(v.replays>0 and v.state_replays>0 and v.pair_AR_calls>0 for v in natives.values())
    assert all(q['adaptive_fallback_cycle']==2 for z in reports.values()for q in z['speculative_quality_cases']if q['strategy']=='adaptive')
    for z in reports.values():
        for q in z['speculative_quality_cases']:
            if q['adaptive_fallback_cycle']:
                assert all(c.get('pair_AR_continuation')for c in q['cycles'][2:])
                assert not any(c['draft_RPC_s']for c in q['cycles'][2:])
    assert reports[2]['speculative_quality_cases'][-1]['generated_tokens']==512 and reports[2]['speculative_quality_cases'][-1]['truncated']
    q={'status':'ACTUAL_FOUR_NODE_PROTOCOL_CPU_MOCK_PASS','GPU':False,'math_mocked':True,'loopback_wire_real':True,'nodes':4,'cases_per_node':24,'capacity512_last_cycle_no_overrun':True,'optional_modal_six_taps':True,'modal_draft':modal_record,'state_replays':[v.state_replays for v in natives.values()],'pair_AR_calls':[v.pair_AR_calls for v in natives.values()],'queue_append_cases':7,'automatic_AR_calibration':True,'fallback_cycle':2,'draft':draft_record,'native_partial_replays':[v.replays for v in natives.values()],'cases_K':[q['K']for q in reports[2]['speculative_quality_cases']],'no_whole_sequence_identity_gate':True,'quality_scoring_smoke_exercised':True,'elapsed_s':time.perf_counter()-start};pathlib.Path('CPU-online-protocol-receipt-r9116.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
