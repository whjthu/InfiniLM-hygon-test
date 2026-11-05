"""Pure immutable C1 package/config/receipt identity; no Native imports."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,sys,socket
D=Path(__file__).resolve().parent;DEPLOY=D.parent
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PACKAGE=R/'candidates/public-bounded-queue-r481-CPU-v1/source'
LABEL='Deep-r481-bounded-queue'
PROVIDER_SHA='473238545ee485d1558d8c5a2b326b62298b2682004a4c28d2c6a49c778675d8'
CPU_SOURCE_SHA='cf17247dfba2758b85e6e07251228a57d2814a1ecc56b5d7e3575e0f14ca834e'
NATIVE_SHA='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
GPU_SECONDS=21600;NATIVE_SECONDS=21000;SERVICE_SECONDS=20400;CLEANUP_SECONDS=22080;KEEPER_SECONDS=22200
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
HOST_PYTHON='/usr/local/python3.12/bin/python3'
def verify_host_python():
 b=read(D/'HOST_PYTHON_BINDING.json');node=socket.gethostname().split('.')[0];assert node in b['node_identity'],'UnqualifiedHOSTinterpreter node';v=b['node_identity'][node];assert list(sys.version_info[:3])==v['version']and list(sys.version_info[:2])>=b['minimum_version'];assert str(Path(sys.executable).resolve())==v['canonical']and SHA(Path(sys.executable).resolve())==v['SHA256'];return dict(node=node,**v)
def hostpath(p):
 p=str(p)
 return R/p[len('/refactor/'):]if p.startswith('/refactor/')else Path(p)
def read(p):return json.loads(Path(p).read_bytes())
def resource_claim(files,private_plan):
 member_hash=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return dict(schema=1,GPU_GO=True,Model_GO=True,qualification=False,source_member_set_SHA256=member_hash,provider_source_SHA256=PROVIDER_SHA,CPU1068_binding_SHA256=SHA(DEPLOY/'CPU1068_BINDING.json'),private_plan=private_plan,GPU_parent=910374,instances=read(D/'MATRIX.json')['instances'],per_instance=dict(cards=8,cores=32,host_MiB=128000,TP=8,PP=1,rows=1,context=32768,Native_SHA256=NATIVE_SHA,device_floor_GiB=16,HTTP_planned=20,warm=0,GPU_seconds=GPU_SECONDS,Native_seconds=NATIVE_SECONDS,service_seconds=SERVICE_SECONDS,cleanup_seconds=CLEANUP_SECONDS,keeper_seconds=KEEPER_SECONDS),keeper=dict(parent=910407,node='a14r2n20',cores_each=2,MiB_each=8192),no_retry_or_reload=True,first_pair='A',B_gate='BothAfirstfullserialEOS+normalownerreset, notanswercorrectness')

def verify(digest,name):
 verify_host_python()
 assert SHA(DEPLOY/'SOURCE_MANIFEST.json')==digest;m=read(DEPLOY/'SOURCE_MANIFEST.json')
 for n,h in m['files_SHA256'].items():
  p=DEPLOY/n;assert p.resolve().is_relative_to(DEPLOY.resolve())and SHA(p)==h,n
 assert SHA(DEPLOY/'GPU_RESOURCE_CLAIM.json')==m['resource_claim_SHA256']and read(DEPLOY/'GPU_RESOURCE_CLAIM.json')==resource_claim(m['files_SHA256'],m['private_plan'])
 assert m['provider_source_SHA256']==PROVIDER_SHA and SHA(PACKAGE/'SOURCE_MANIFEST.json')==PROVIDER_SHA
 for n,h in read(PACKAGE/'SOURCE_MANIFEST.json')['files_SHA256'].items():
  p=PACKAGE/n;assert p.resolve().is_relative_to(PACKAGE.resolve())and SHA(p)==h,n
 b=read(DEPLOY/'CPU1068_BINDING.json');receipt=hostpath(b['receipt']);assert SHA(receipt)==b['receipt_SHA256'];q=read(receipt)
 assert q['actual_step']=='910407.1068'and q['completed']and q['passed']and q['normal_CLI_nonserve']and q['source_manifest_SHA256']==CPU_SOURCE_SHA and q['derived_manifest_SHA256']==PROVIDER_SHA
 assert not any(q[k]for k in ['Native','GPU','dlopen','weights','new_encode'])
 matrix=read(D/'MATRIX.json');assert name in matrix['instances'];v=matrix['instances'][name]
 assert v['pair']in['A','B']and type(v['queue_policy'])is bool and v['node']in['a02r4n01','a02r4n04','a02r4n05','a02r4n07']and re.fullmatch(r'[A-Za-z0-9_.-]+',v['run'])
 config_path=DEPLOY/v['config'];c=read(config_path);expected=read(PACKAGE/'config.template.json')
 expected.update(execute_authorized=True,experimental_profile=None,runtime_directory='/refactor/results/Deep-bounded-queue-r481',node=v['node'],Native_seconds=NATIVE_SECONDS,service_seconds=SERVICE_SECONDS,bounded_queue_enabled=v['queue_policy'],physical_pending_capacity=2,queue_wait_seconds=5.,token_clock_enabled=True)
 assert c==expected and(c['rows'],c['context'],c['HC_mode'],c['Native_SHA256'])==(1,32768,0,NATIVE_SHA)
 plan=hostpath(m['private_plan']['path']);assert SHA(plan)==m['private_plan']['SHA256'];p=read(plan);assert p['ordinals']==[0,1,128,129]and len(p['requests'])==20 and len(p['records'])==4
 for i,row in enumerate(p['requests']):
  assert row['index']==i and row['body']['max_tokens']==4096 and row['body']['stream']is True and row['ordinal']in p['ordinals']
  assert row['actual_body_SHA256']==hashlib.sha256(json.dumps(row['body'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 return m,v,c,config_path

def identity(v,step,epoch,config_path):
 return dict(run=v['run'],actual_step=step,source_manifest_SHA256=PROVIDER_SHA,provider_config_SHA256=SHA(config_path),gpu_start_epoch=epoch,Native_seconds=NATIVE_SECONDS)
def server_config(v,step,epoch,c,config_path):
 q=dict(c,**identity(v,step,epoch,config_path));q["runtime_directory"]=str(hostpath(c["runtime_directory"]));return q
def folder(c,v):return hostpath(c['runtime_directory'])/v['run']
def atomic(path,value):
 p=Path(path);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def exclusive(path,value):
 with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
