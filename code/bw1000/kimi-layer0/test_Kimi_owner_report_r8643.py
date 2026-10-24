"""CPU stress of the actual report publisher: no Engine/GPU/performance fixture."""
import pathlib,threading,json,time,tempfile,hashlib,sys,ast
from kimi_owner_report_r8643 import OwnerSnapshotReport

def one(inject):
 with tempfile.TemporaryDirectory(prefix='Kimi-report-r8643-')as tmp:
  path=pathlib.Path(tmp)/'report.json';counts={'serialized':0,'observer':0};bad=[]
  def validate(v):
   assert len(v['workers'])==8
   for i,w in enumerate(v['workers']):
    assert w['rank']==i and 'unpublished_marker'not in w
    for s in w.get('steps',[]):assert s['rank']==i and s['complete']and len(s['data'])==s['ordinal']+1
  def encode(v):
   pieces=[]
   for i,piece in enumerate(json.JSONEncoder().iterencode(v)):
    pieces.append(piece)
    if i%128==0:time.sleep(.0001)
   text=''.join(pieces);validate(json.loads(text));counts['serialized']+=1;return text
  report=OwnerSnapshotReport(path,{'phase':'start','workers':[{'rank':i,'normal_close':False}for i in range(8)]},encode=encode);report.flush();bar=threading.Barrier(8,timeout=10);stop=threading.Event();errors=[]
  def observer():
   try:
    while not stop.is_set():
     snap=report.snapshot();validate(snap);snap['workers'][0]['injected_external_alias']=True;report.flush();counts['observer']+=1;time.sleep(.0005)
   except BaseException as e:bad.append(repr(e))
  t=threading.Thread(target=observer);t.start()
  def worker(rank):
   own={'rank':rank,'normal_close':False,'steps':[]}
   try:
    for i in range(16):
     own['steps'].append({'rank':rank,'ordinal':i,'data':list(range(i+1)),'complete':True});report.publish_worker(rank,own)
     own['unpublished_marker']=i;time.sleep(0);del own['unpublished_marker']
     if inject and rank==3 and i==7:raise RuntimeError('injected_owner_failure')
     bar.wait()
   except BaseException as e:own['error']=repr(e);errors.append(rank);bar.abort();report.publish_worker(rank,own)
   finally:own['normal_close']=True;report.publish_worker(rank,own)
  owners=[threading.Thread(target=worker,args=(i,))for i in range(8)];start=time.monotonic()
  for q in owners:q.start()
  for q in owners:q.join(20)
  assert not any(q.is_alive()for q in owners),'owner deadlock'
  stop.set();t.join(10);assert not t.is_alive()and not bad,bad
  report.publish_top(phase='closed');final=json.loads(path.read_text());validate(final);assert all(w['normal_close']for w in final['workers']);assert all('injected_external_alias'not in w for w in final['workers'])
  if inject:assert len(errors)==8 and 'injected_owner_failure'in final['workers'][3]['error']and all('error'in w for w in final['workers'])
  else:assert not errors and all(len(w['steps'])==16 for w in final['workers'])
  return {'injected_exception':inject,'writers':8,'serializer_forced_yield':True,'serialized_JSON_frames':counts['serialized'],'independent_snapshot_observations':counts['observer'],'all8_normalclose_recorded':True,'errors_expected':bool(inject),'all_expected_errors_preserved':not inject or len(errors)==8,'isolated_deep_snapshots':True,'deadlock':False,'wall_s':time.monotonic()-start}
if __name__=='__main__':
 r=pathlib.Path(__file__).resolve().parent;ast.parse((r/'kimi_embedding_layer0_runner_r8643.py').read_text());out={'CPU_only':True,'GPU':False,'native_or_model':False,'actual_publisher_module_SHA256':hashlib.sha256((r/'kimi_owner_report_r8643.py').read_bytes()).hexdigest(),'actual_consumer_SHA256':hashlib.sha256((r/'kimi_embedding_layer0_runner_r8643.py').read_bytes()).hexdigest(),'cases':[one(False),one(True)],'scope':'Actual publisher concurrent mutation/snapshot/exception lifecycle only; not numerical or throughput evidence','passed':True};p=r/'Kimi-owner-report-stress-r8643.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
