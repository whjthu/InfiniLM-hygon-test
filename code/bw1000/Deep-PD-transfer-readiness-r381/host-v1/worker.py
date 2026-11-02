"""One CPU host ticket across two nodes; no Native, GPU, model, or KV qualification."""
from pathlib import Path
import os,socket,time,json,hashlib,traceback,re
import state_wire as wire

NODES=['a02r4n12','a02r4n13'];IPS=['12.1.2.72','12.1.2.73'];PORT=49181
sha=lambda b:hashlib.sha256(b).hexdigest()
def blocks():
 return [dict(key='synthetic_host_bytes',shard=i,nbytes=(1<<20)+17*i,
  layout_SHA256=sha(wire.canonical(dict(dtype='uint8',shape=[(1<<20)+17*i],stride=[1],scope='HOST_ONLY_NOT_KV'))))for i in range(8)]
def pattern(index,offset,count):
 unit=bytes((i+index*37)%256 for i in range(256));start=offset%256
 return (unit*((start+count+255)//256))[start:start+count]
def atom(path,value):
 p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)
def identity(run,nonce):
 return dict(model_SHA256=sha(b'NO_MODEL_CPU_ONLY'),ABI_SHA256=sha(wire.canonical(blocks())),prompt_IDs_SHA256=sha(b'NO_PROMPT_CPU_ONLY'),request_nonce=nonce,epoch=1,past=1024,TP=8,next_token=1,output_budget=128)

class HostSink:
 def __init__(self):self.values=[];self.verified=False;self.publish_count=0;self.aborted=False;self.written=[]
 def prepare(self,ident,descriptors):self.values=[bytearray(b['nbytes'])for b in descriptors];self.written=[0]*len(descriptors)
 def write(self,index,offset,part):self.values[index][offset:offset+len(part)]=part;self.written[index]+=len(part)
 def verify(self):
  assert len(self.values)==8 and all(v==pattern(i,0,len(v))for i,v in enumerate(self.values)),'Host pattern mismatch'
  self.verified=True
 def publish(self):
  assert self.verified and not self.aborted and self.publish_count==0
  self.publish_count=1
 def abort(self):self.values=[];self.aborted=True;self.verified=False
 def close(self):self.values=[]

def execute(role,run,root,addresses,port,deadline_unix,nonce,actual_step,source_SHA,*,inject_wrongpattern=False):
 """Shared actual entry, also exercised with loopback in the explicitly local CPU check."""
 assert role in ['P','D']and re.fullmatch('[a-z0-9-]+',run)
 root=Path(root);results=root/'results';control=root/'control';results.mkdir(exist_ok=True);control.mkdir(exist_ok=True)
 report=results/('Deep381-hostTCP-'+run+'-'+role+'.json');ready=control/('Deep381-hostTCP-'+run+'-listening.json');assert not report.exists()
 q=dict(schema=1,completed=False,passed=False,role=role,run=run,node=socket.gethostname(),actual_step=actual_step,source_manifest_SHA256=source_SHA,
  GPU=0,Native_dlopen=False,model_loaded=False,actual_KV_state=False,Native_verify_called=False,host_pattern_verification_only=True,
  request_nonce_SHA256=sha(nonce.encode()),protocol_wire_SHA256=sha(Path(wire.__file__).read_bytes()),single_connection_no_retry=True,staged_shards=0)
 save=lambda:atom(report,q);save();sock=listener=None;sink=None;begin=time.monotonic();phase='before_socket'
 try:
  assert deadline_unix-time.time()>0,'Original CPU deadline exhausted before socket'
  deadline=time.monotonic()+deadline_unix-time.time()
  ident=identity(run,nonce);desc=blocks();q['payload_bytes']=sum(v['nbytes']for v in desc)
  if role=='D':
   phase='listen';listener=socket.socket();listener.bind((addresses[1],port));listener.listen(1);wire.remaining(listener,deadline)
   assert not ready.exists();atom(ready,dict(run=run,request_nonce_SHA256=q['request_nonce_SHA256'],source_manifest_SHA256=source_SHA,address=addresses[1],port=port,listener_PID=os.getpid(),actual_step=actual_step))
   sock,peer=listener.accept();assert peer[0]==addresses[0];listener.close();listener=None;q['peer_IP']=peer[0]
   phase='receive_verify_publish';sink=HostSink();ticket=wire.ImportTicket(ident,desc,sink);q['transfer']=ticket.receive(sock,deadline)
   q.update(staged_shards=8,host_pattern_verify_passed=sink.verified,whole_ticket_host_publish_count=sink.publish_count,CPU_host_ticket_published=ticket.published)
  else:
   phase='bounded_listening_marker'
   while True:
    assert time.time()<deadline_unix,'Original CPU deadline exhausted awaiting listening marker'
    try:mark=json.loads(ready.read_bytes())
    except FileNotFoundError:time.sleep(.05);continue
    assert mark['run']==run and mark['source_manifest_SHA256']==source_SHA and mark['request_nonce_SHA256']==q['request_nonce_SHA256']and mark['address']==addresses[1]and mark['port']==port
    break
   phase='single_connect';sock=socket.socket();sock.bind((addresses[0],0));wire.remaining(sock,deadline);sock.connect((addresses[1],port));q['local_IP']=sock.getsockname()[0];q['peer_IP']=sock.getpeername()[0]
   def read_block(index,offset,count):
    data=pattern(index,offset,count)
    return (bytes([data[0]^1])+data[1:])if inject_wrongpattern and index==7 and offset==0 else data
   phase='export_wait_complete_ticket_ACK';q['transfer']=wire.export_state(sock,ident,desc,read_block,deadline);q['all8shards_publish_ACK_received']=True
  q.update(passed=True,completed_phase='complete_ticket_ACK',no_Model_or_KV_gain_claim=True)
 except BaseException as e:q.update(error=repr(e),error_phase=phase,traceback=traceback.format_exc());raise
 finally:
  if sink is not None:q.update(sink_aborted=sink.aborted,publish_count_on_exit=sink.publish_count,host_written_bytes_per_shard=sink.written);sink.close();q['all_host_staging_released']=not sink.values
  if sock is not None:sock.close()
  if listener is not None:listener.close()
  q.update(completed=True,normal_socket_close=True,local_complete_entry_wall_s=time.monotonic()-begin);save()
 return q

def main():
 d=Path(__file__).resolve().parent;m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());expected=os.environ['DEEP_R381_CPU_MANIFEST'];assert sha((d/'SOURCE_MANIFEST.json').read_bytes())==expected
 assert all(sha((d/n).read_bytes())==h for n,h in m['files_SHA256'].items())and m['GPU_GO']is False
 rank=int(os.environ['SLURM_PROCID']);assert rank in [0,1]and socket.gethostname()==NODES[rank]
 k=Path('/sys/class/kfd/kfd/proc');assert k.exists()and not any(p.name.isdigit()for p in k.iterdir()),'Candidate node has active KFD owners'
 q=execute(['P','D'][rank],os.environ['DEEP_R381_CPU_RUN'],Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014'),IPS,PORT,float(os.environ['DEEP_R381_CPU_DEADLINE']),os.environ['DEEP_R381_CPU_NONCE'],os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],expected)
 print(json.dumps(dict(role=q['role'],passed=q['passed'],GPU=0,Native=False,actual_step=q['actual_step'])))
if __name__=='__main__':main()
