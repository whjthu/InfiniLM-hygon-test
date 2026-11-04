"""Exact current C1 PP2 layout, consumed before device initialization and create.

The historical source/receipt/CABI audit remains in production preengine. This
module contains current topology/config arguments, without a device or loader.
"""
from pathlib import Path
import base64,ctypes,hashlib,json,os,re,socket
from .cache_policy_r452 import from_environment
BOUNDS=(0,38,78)

# Only the eight frozen r444 node pairs, supplied by the SHA-bound config.
ALLOWED_NODE_PAIRS_R444 = (('a02r4n14', 'a02r4n15'), ('a02r4n04', 'a02r4n05'), ('a02r3n03', 'a02r3n07'), ('a02r3n15', 'a02r3n16'), ('a14r2n08', 'a14r2n09'), ('a14r2n10', 'a14r2n12'), ('a02r4n12', 'a02r4n13'), ('a02r4n01', 'a02r4n07'))

def validated_nodes_r444(nodes):
 assert type(nodes)is list and len(nodes)==2 and all(type(n)is str for n in nodes)
 pair=tuple(nodes);assert pair in ALLOWED_NODE_PAIRS_R444 and len(set(pair))==2
 return pair
def validate_layout(environ,config,hostname):
 assert environ['GLM_PP_SIZE_R3603']=='2'and environ['GLM_CACHE_ROWS_R3627']=='1'and environ['GLM_CACHE_CAP_R3627']=='32768'
 policy=from_environment(environ,int(environ['GLM_STAGE_R5040']))
 assert environ['GLM_BOUNDED_ROPE_R4279']==environ['GLM_SHARED_CONTROL_R4279']=='1'
 assert environ['GLM_SHARED_PREFIX_VALIDATE_R5814']=='0'and environ['GLM_ACTUAL_CAPTURE_R5164']=='0'and environ['USE_WEIGHT_CACHE_R3604']=='0'
 assert environ['LD_PRELOAD']=='/refactor/candidates/blas-policy-r271/libblas_policy_r271.so'
 assert '/install/hygon-glm-5f91ad7-r3672/lib:'in environ['LD_LIBRARY_PATH']
 stage=int(environ['GLM_STAGE_R5040']);assert stage in (0,1)and int(environ['SLURM_PROCID'])==stage
 assert hostname.split('.')[0]==validated_nodes_r444(json.loads(environ['GLM_EXPECTED_NODES_R444']))[stage]
 assert config['num_hidden_layers']==78 and config['hidden_size']==6144
 assert len(config['mlp_layer_types'])==len(config['indexer_types'])==78
 assert all(config['indexer_types'][i]=='full'for i in BOUNDS[:-1])
 return dict(TP=8,PP=2,rows=1,capacity=32768,stage=stage,node=hostname.split('.')[0],bounds=list(BOUNDS),layer_begin=BOUNDS[stage],layer_end=BOUNDS[stage+1],all_serving_owners=16,no_resident_V=bool(policy.noV_flag),cache_policy=policy.metadata())
def read_layout(environ=None,config_path=None,hostname=None):
 env=os.environ if environ is None else environ;p=Path('/old/models/ZhipuAI-GLM-5.3/config.json')if config_path is None else Path(config_path);raw=p.read_bytes();q=validate_layout(env,json.loads(raw),socket.gethostname()if hostname is None else hostname);q.update(config_path=str(p),config_SHA256=hashlib.sha256(raw).hexdigest(),metadata_only=True);return q
def create_context(lib,rank,stage,rows,capacity,config,payload,environ=None,hostname=None):
 env=os.environ if environ is None else environ;layout=validate_layout(env,config,socket.gethostname()if hostname is None else hostname)
 assert type(rank)is int and 0<=rank<8 and stage==layout['stage']and(rows,capacity)==(1,32768)
 assert len(payload['tp'])==2
 # Original public constructor's eight arguments and temporary ctypes buffers;
 # no additional sync, tensor, precision, group, or mathematical operation.
 buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k]if k!='tp'else payload[k][stage],validate=True))for k in ['world','tp','control']]
 return lib.modern_glm_model_create(rank,stage,rows,capacity,json.dumps(config).encode(),*buffers)
