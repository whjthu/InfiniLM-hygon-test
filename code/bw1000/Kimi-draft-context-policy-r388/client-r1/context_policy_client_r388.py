"""335 command/abort unchanged; explicit388 ON handshake, no fallback."""
from pathlib import Path
import hashlib,importlib.util,json
from types import SimpleNamespace

OFF='off';ON='projected-only-r384'
WORKER_MF='b9a74a6bf1ea7e0922bcaa15764f84022864bd1ea0a78e593c7ee8b80d2a37f0'
PUBLISHER_SHA='feec885f60ee034144285511e52361661df0e8be21073be091e4e02fc8b8f4af'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def bundle(root):
 root=Path(root);directory=root/'incoming/Kimi-draft-context-policy-r388';manifest=directory/'SOURCE_MANIFEST.json'
 assert sha(manifest)==WORKER_MF;m=json.loads(manifest.read_bytes())
 assert all(sha(directory/n)==h for n,h in m['files_SHA256'].items())
 def host(path):return root/Path(path).relative_to('/refactor')if path.startswith('/refactor/')else Path(path)
 assert all(sha(host(p))==h for p,h in m['dependencies_SHA256'].items())
 policy=load(directory/'policy.py','worker_policy388_for_target')
 olddir=root/'incoming/Kimi-long-draft-memory-floor-r362b'
 assert sha(olddir/'worker.py')==policy.OLD_WORKER_SHA256 and sha(olddir/'SOURCE_MANIFEST.json')==policy.OLD_MANIFEST_SHA256
 old=load(olddir/'worker.py','old_worker362b_for_target388')
 return SimpleNamespace(policy=policy,old=old,manifest_SHA256=WORKER_MF)

def client_type(original,bound,mode=OFF):
 if mode==OFF:return original
 if mode!=ON:raise ValueError('Explicit supported context policy required')
 assert sha(original.__init__.__code__.co_filename)==PUBLISHER_SHA
 assert bound.manifest_SHA256==WORKER_MF and bound.policy.ON==ON
 assert original.__init__.__globals__['RUNTIME_SHA']==bound.old.RUNTIME_SHA and original.__init__.__globals__['NATIVE_SHA']==bound.old.TARGET_NATIVE_SHA
 send=original.__init__.__globals__['send'];recv=original.__init__.__globals__['recv']
 class PolicyDraftClient(original):
  def __init__(self,connection,nonce):
   if type(nonce)is not str or len(nonce)<32:raise ValueError('Explicit session nonce required')
   self.connection=connection;self.sequence=0;self.poisoned=self.closed=False
   self.policy_binding=dict(context_policy=ON,context_source_SHA256=bound.policy.CONTEXT_SHA256,worker_entry_manifest_SHA256=WORKER_MF,protocol=bound.policy.PROTOCOL_ON,runtime_manifest_SHA256=bound.old.RUNTIME_SHA,target_Native_SHA256=bound.old.TARGET_NATIVE_SHA)
   try:
    hello=bound.policy.expected_hello(bound.old,nonce,ON,WORKER_MF);send(connection,hello)
    meta,body=recv(connection);expected=bound.policy.ready(bound.old,ON,hello)
    if body or set(meta)!=set(expected)or any(meta[k]!=v or type(meta[k])is not type(v)for k,v in expected.items()):raise ValueError('Exact388 ON ready policy/source/nonce-session identity required')
   except BaseException:self.poisoned=True;connection.close();raise
 assert PolicyDraftClient.command is original.command and PolicyDraftClient.abort is original.abort
 return PolicyDraftClient
