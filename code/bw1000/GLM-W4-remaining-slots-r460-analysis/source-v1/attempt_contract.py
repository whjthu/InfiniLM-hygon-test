"""Explicit existing0/1/6 identity binding; stdlib/no Runtime or remote calls."""
import json,math,re
from pathlib import Path
BINDINGS={int(k):v for k,v in json.loads((Path(__file__).parent/'ATTEMPT_BINDINGS.json').read_bytes()).items()}
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def nodes(value):
 if '['not in value:return set(value.split(','))
 match=re.fullmatch(r'([^[]+)\[([^]]+)\]',value);assert match,value;out=set()
 for item in match[2].split(','):
  if '-'in item:
   lo,hi=item.split('-');out.update(match[1]+str(i).zfill(len(lo))for i in range(int(lo),int(hi)+1))
  else:out.add(match[1]+item)
 return out
def bind_attempt(store,spec,watch):
 b=BINDINGS[spec['slot']];slot=b['slot'];prefix=b['label']+'-'+b['run'];claim=watch['resource_claim']
 normalize=lambda value:R+value[len('/refactor'):]if value.startswith('/refactor/')else value
 assert normalize(spec['watch_path'])==R+'/results/'+prefix+'-watch.json'
 assert normalize(spec['client_path'])==R+'/results/'+prefix+'-client/result.json'
 assert normalize(spec['run_directory'])==R+'/results/GLM-W4-fewshot-quality-r444/'+b['run']
 assert watch['slot']==slot and claim['run']==b['run']and claim['label']==b['label']and claim['slot']==claim['shard_index']==slot and claim['shards']==8 and claim['nodes']==b['nodes']
 assert claim['HTTP']==claim['Native']==64 and claim['cards']==16 and(claim['TP'],claim['PP'])==(8,2)
 assert watch['deployment_source_SHA256']==b['deployment']and(watch['actual_GPU'],watch['actual_client'],watch['CPU_keeper'])==(b['GPU'],b['client'],b['keeper'])
 assert math.isclose(watch['original_GPU_start_unix'],b['epoch'],abs_tol=1e-6,rel_tol=0)and math.isclose(watch['first_all16_ready_unix'],b['first_ready'],abs_tol=1e-6,rel_tol=0)
 assert watch['startup_accepted']is True # Historical acceptance; no elapsed-time re-gate.
 records=[row.split('|')for row in store.raw(spec['accounting_path']).decode().splitlines()if'|'in row]
 expected={b['GPU']:(prefix,set(b['nodes']),29400),b['client']:(prefix+'-client',{b['nodes'][1]},28200),b['keeper']:(prefix+'-keeper',{'a14r2n20'},32400)}
 normal=True
 for step,(label,where,limit)in expected.items():
  found=[row for row in records if row[0]==step];assert len(found)==1;row=found[0]
  assert row[1]==label and nodes(row[5])==where and row[2].split()[0].rstrip('+')in{'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}and row[4].isdigit()and 0<=int(row[4])<=limit
  normal=normal and row[2:4]==['COMPLETED','0:0']
 post_errors=[]
 try:
  post=watch.get('post')or watch.get('post_after_failure')or[];assert len(post)==2 and{p['node']for p in post}==set(b['nodes'])
  for step in{p['actual_step']for p in post}:
   found=[row for row in records if row[0]==step];assert len(found)==1 and found[0][1]in{prefix+'-post',prefix+'-failure-post'}and found[0][2:4]==['COMPLETED','0:0']and nodes(found[0][5])==set(b['nodes'])
 except(AssertionError,KeyError,ValueError,TypeError)as error:post_errors.append(repr(error))
 if b['legacy_identity_not_archived']:return dict(terminal_accounting_normal=normal,post_accounting_normal=not post_errors,post_accounting_errors=post_errors,client_identity='NOT_RECORDED_IN_ORIGINAL886_ARCHIVE; original singleton binding retained',PID_independent_proof='UNKNOWN')
 identpath=R+'/results/'+prefix+'-client-identity.json';identity=store.data(identpath)
 assert type(identity['PID'])is int and identity['PID']>0 and identity['node']==b['nodes'][1]and identity['actual_CPU']==b['client']and identity['GPU']==identity['Native']==0
 assert identity['client']=='/refactor/incoming/GLM-W4-fewshot-quality-r444-client-v1/client-v1/quality_client.py'and identity['client_SHA256']==b['client_source_SHA256']
 argv=identity['argv']
 def argument(name):assert argv.count(name)==1;return argv[argv.index(name)+1]
 assert argument('--expected-step')==b['GPU']and argument('--run')==b['run']and argument('--shard-index')==str(slot)and argument('--shards')=='8'and argument('--native-offset')=='0'
 assert argument('--provider-source-sha')=='eaf5ae8c5333e0601ce2bec7fdd27e41e0a110b534660e27f55c362915cb96a0'and argument('--config-sha')==b['config_SHA256']
 assert math.isclose(float(argument('--gpu-start-epoch')),b['epoch'],abs_tol=1e-6,rel_tol=0)and math.isclose(float(argument('--deadline-unix')),b['epoch']+28200,abs_tol=1e-6,rel_tol=0)
 deploy=R+'/incoming/GLM-W4-remaining-slots-r460/deployment-v1/'
 assert __import__('hashlib').sha256(store.raw(deploy+'SOURCE_MANIFEST.json')).hexdigest()==b['deployment']
 assert __import__('hashlib').sha256(store.raw(deploy+'GPU_MANIFEST-initial-two.json')).hexdigest()==b['GPU_manifest_SHA256']
 assert __import__('hashlib').sha256(store.raw(deploy+'config-slot%d.json'%slot)).hexdigest()==b['config_SHA256']
 return dict(terminal_accounting_normal=normal,post_accounting_normal=not post_errors,post_accounting_errors=post_errors,client_identity='VERIFIED_FROM_ORIGINAL_SOURCE_LAUNCH_RECORD_AND_EXACT_SACCT',PID=identity['PID'],node=identity['node'],actual_CPU=identity['actual_CPU'],client_code_SHA256=identity['client_SHA256'],PID_scope='Original execv identity; no invented independent PID/startticks telemetry')
