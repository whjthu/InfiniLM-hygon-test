"""Compose exact r480 public overlay with bounded physical FIFO admission."""
from pathlib import Path
import importlib.util,json,hashlib,argparse
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def build(parent,clock,output,parent_sha,clock_sha):
 parent,clock,output=Path(parent),Path(clock),Path(output)
 assert sha(parent/'SOURCE_MANIFEST.json')==parent_sha and sha(clock/'SOURCE_MANIFEST.json')==clock_sha
 m=json.loads((clock/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(clock/n)==h
 spec=importlib.util.spec_from_file_location('qualified_clock_overlay',clock/'build_overlay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.build(parent,output,parent_sha)
 def replace(name,old,new):
  p=output/'public_deep_model'/name;s=p.read_text();assert s.count(old)==1,(name,old);p.write_bytes(s.replace(old,new).encode())
 replace('prepared_factory.py','class SelectedDeepBridge(AdmissionMixin, TextDeliveryMixin, bridge_class):','from .queue_policy import QueuePolicyMixin\n    class SelectedDeepBridge(QueuePolicyMixin, AdmissionMixin, TextDeliveryMixin, bridge_class):')
 replace('cli.py','provider=DeepProvider(config,prepared)',"prepared['engine'].configure_queue_policy(config)\n provider=DeepProvider(config,prepared)")
 replace('worker.py','   if stop_intent.is_set():break','   if stop_intent.is_set():engine.defer_stopped_dequeued(req);break')
 replace('worker.py','   context.reset()\n   with engine._lock:', '   context.reset()\n   engine.close_never_owned_after_reset()\n   with engine._lock:')
 replace('worker.py','  else:\n   with engine._lock:','  else:\n   engine.close_never_owned_after_reset(primary)\n   with engine._lock:')
 replace('worker.py',"    admission=copy.deepcopy(req.metadata)","    if stop_intent.is_set():engine.defer_stopped_dequeued(req);break\n    if not engine.claim_queued(req):continue\n    admission=copy.deepcopy(req.metadata)")
 replace('worker.py',"  t=path.with_suffix('.tmp');t.write_text","  report['queue_ledger']=list(prepared['engine'].queue_ledger)\n  t=path.with_suffix('.tmp');t.write_text")
 replace('worker.py',"fixed_profile=report['fixed_profile']))","fixed_profile=report['fixed_profile'],queue_policy=prepared['engine'].queue_policy_ready()))")
 replace('provider.py',"fixed_profile=row['fixed_profile']","fixed_profile=row['fixed_profile'],queue_policy=row['queue_policy']")
 (output/'public_deep_model/queue_policy.py').write_bytes((Path(__file__).parent/'queue_policy.py').read_bytes())
 m=json.loads((output/'SOURCE_MANIFEST.json').read_bytes());m.update(schema='public-bounded-queue-r481',CPU_GO=False,GPU_GO=False,Model_GO=False,queue_default=False,clock_parent_manifest_SHA256=clock_sha)
 m['files_SHA256']={str(p.relative_to(output)).replace('\\','/'):sha(p)for p in output.rglob('*')if p.is_file()and p.name!='SOURCE_MANIFEST.json'};(output/'SOURCE_MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('clock');p.add_argument('output');p.add_argument('--parent-sha',required=True);p.add_argument('--clock-sha',required=True);a=p.parse_args();build(a.parent,a.clock,a.output,a.parent_sha,a.clock_sha)
