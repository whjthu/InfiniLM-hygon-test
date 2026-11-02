"""Fixed PD benchmark geometry and original-epoch bounds; no model/device imports."""
from pathlib import Path
import hashlib,json,re,time

PLAN_SHA='ff0902853e1d1eb78b1cbb732f3d9e1b70f36581c4f970485dbd9de279366680'
ROLE_NODES={'P':'a02r4n12','D':'a02r4n13'}
ROLE_IPS={'P':'12.1.2.72','D':'12.1.2.73'}
CONTROL_PORTS={'P':49282,'D':49283};STATE_PORT=49181
GPU_HARD_S=3600;NATIVE_S=3540;CPU_HARD_S=3900
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_plan(path):
 assert sha(path)==PLAN_SHA,'Frozen final PD workload differs'
 p=json.loads(Path(path).read_bytes())
 assert p['planned_external_requests']==15 and p['planned_role_request_lifecycles']['total_stage_lifecycles']==22
 assert p['output_budget_including_P_first_token']==128 and p['context']==32768 and p['fixture']['actual_prompt_tokens']==8489
 return p

def request_groups(plan,nonce):
 groups=[];ordinal=0
 for g in plan['planned_groups']:
  requests=[]
  for index in range(g['requests']):
   mode=g['mode'];role=g['co_nodes'][index]if mode=='co'else'P'
   requests.append(dict(index=ordinal,epoch=ordinal+1,request_nonce=nonce(),group=g['id'],phase=g['phase'],mode=mode,co_role=role,
    arrival_offset_s=g.get('arrival_offsets_s',[0.]*g['requests'])[index],output_budget=plan['output_budget_including_P_first_token'],past=8489,prompt_IDs_SHA256=plan['fixture']['actual_prompt_token_ids_SHA256']))
   ordinal+=1
  groups.append(dict(id=g['id'],phase=g['phase'],mode=g['mode'],requests=requests))
 assert ordinal==plan['planned_external_requests']
 return groups

def check_live(original_epoch,cancel_file):
 if Path(cancel_file).exists():raise RuntimeError('Original PD run canceled; no request/state retry')
 if time.time()>=original_epoch+NATIVE_S:raise TimeoutError('Original PD Native deadline exhausted')

def public_request(value):
 result={k:v for k,v in value.items()if k!='request_nonce'}
 if'request_nonce'in value:result['request_nonce_SHA256']=hashlib.sha256(value['request_nonce'].encode()).hexdigest()
 return result

def run_identity(run,nonce):
 assert re.fullmatch('[a-z0-9-]+',run)and re.fullmatch('[0-9a-f]{64}',nonce)
 return dict(run=run,session_nonce_SHA256=hashlib.sha256(nonce.encode()).hexdigest())
