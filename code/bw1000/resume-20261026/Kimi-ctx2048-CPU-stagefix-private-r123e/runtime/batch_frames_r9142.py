"""One logical multi-request RPC; bounded per-request binary continuation frames."""
from batch_wire_r9128 import send,recv,MAX_BODY
FEATURE_BYTES=35840*2

def send_query(sock,meta,bodies):
 reqs=meta['requests'];assert 1<=len(reqs)<=32 and len(reqs)==len(bodies);assert len({q['request']for q in reqs})==len(reqs)
 for q,b in zip(reqs,bodies):assert 1<=q['rows']<=1024 and len(b)==q['rows']*FEATURE_BYTES and len(b)<=MAX_BODY
 send(sock,{**meta,'feature_frames':len(bodies)})
 for i,b in enumerate(bodies):send(sock,{'cmd':'query_batch_part','seq':meta['seq'],'index':i,'request':reqs[i]['request']},b)

def receive_bodies(sock,meta):
 reqs=meta['requests'];assert 1<=len(reqs)<=32 and meta['feature_frames']==len(reqs) and len({q['request']for q in reqs})==len(reqs);bodies=[]
 for i,q in enumerate(reqs):
  h,b=recv(sock);assert h=={'cmd':'query_batch_part','seq':meta['seq'],'index':i,'request':q['request']} and len(b)==q['rows']*FEATURE_BYTES;bodies.append(b)
 return bodies

def request_group(rpc,active,effective,epoch):
 import time
 start=time.perf_counter();needed=[q for q in active if effective[q.logical]>1];records={q.logical:{'logical':q.logical,'proposals':[],'RPC_s':0.,'draft_query_s':0.,'serial_queue_wait_s':0.}for q in active};extra={}
 if needed:
  fields=[{'request':q.logical,'past':q.draft_past,'rows':q.append_rows,'anchor':q.tokens[-1],'K':effective[q.logical]-1}for q in needed];reply,wall=rpc('query_batch',[q.features for q in needed],epoch=epoch,requests=fields);assert reply['epoch']==epoch and reply['actual_model_batch_C']==len(needed);assert [v['request']for v in reply['requests']]==[q.logical for q in needed]
  for q,v in zip(needed,reply['requests']):
   assert v['context_tokens']==q.past and len(v['proposals'])==effective[q.logical]-1;records[q.logical]={'logical':q.logical,'proposals':v['proposals'],'RPC_s':wall/len(needed),'draft_query_s':reply['draft_batch_query_s']/len(needed),'append_H2D_s':reply.get('append_batch_H2D_s',0.)/len(needed),'serial_queue_wait_s':0.,'batch_shared_RPC_s':wall,'cost_attribution':'shared group cost divided actualC; not perrequest independent timer'}
  extra={'actual_model_batch_C':len(needed),'logical_query_RPCs':1,'binary_feature_frames':len(needed),'shared_RPC_s':wall,'shared_query_s':reply['draft_batch_query_s']}
 return {'requests':[records[q.logical]for q in active],'group_wall_s':time.perf_counter()-start,'batch_draft':extra}
