"""Exact two-replica identity/time policy, checked before device imports."""
from pathlib import Path
import hashlib,json,math,os
PAIRS=(('a02r4n12','a02r4n13'),('a02r4n14','a02r4n15'))
POLICY='695b1a12bfd2929851b3ffa89a9267564f9fd9dc0045c4a9adfbcf7142b21584'
BOUNDS=(13500,14100,14400,15000)
def nodepair(environ=None):
 env=os.environ if environ is None else environ;index=env['GLM_R418_REPLICA_INDEX']
 if index not in('0','1'):raise ValueError('Exactly two approved replica identities')
 return PAIRS[int(index)]
def validate(environ=None,source_directory=None,profile_directory=None):
 env=os.environ if environ is None else environ;nodes=nodepair(env)
 D=Path(__file__).resolve().parent if source_directory is None else Path(source_directory)
 S=Path('/refactor/incoming/GLM-NoV-W4-normal-profile-r417')if profile_directory is None else Path(profile_directory)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 expected=env['GLM_SOURCE_MANIFEST_SHA'];assert len(expected)==64 and sha(D/'SOURCE_MANIFEST.json')==expected
 for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(S/'SOURCE_MANIFEST.json')==env['GLM_R417_SOURCE_SHA256']==POLICY
 for n,h in json.loads((S/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(S/n)==h,n
 assert env['GLM_PP_SIZE_R3603']=='2'and env['GLM_CACHE_ROWS_R3627']=='1'and env['GLM_CACHE_CAP_R3627']=='32768'
 assert env['GLM_V_ON_DEMAND_R403']=='1'and env['GLM_SERVICE_PROFILE_R417']in('glm-nov16-w2','glm-nov16-w4')
 assert env['GLM_R417_DEFAULT_PROFILE']==env['GLM_SERVICE_PROFILE_R417']
 assert env['GLM_PIPELINE_TAIL_NODE']==nodes[1]
 declared=tuple(int(env[k])for k in ['GLM_R418_SERVICE_SECONDS','GLM_R418_NATIVE_SECONDS','GLM_R418_GPU_SECONDS','GLM_R418_KEEPER_SECONDS']);assert declared==BOUNDS
 epoch=float(env['GLM_R290_GPUSTART_UNIX']);assert math.isfinite(epoch)and epoch>0
 return dict(loader_manifest_SHA256=expected,service_profile_manifest_SHA256=POLICY,replica_index=int(env['GLM_R418_REPLICA_INDEX']),nodes=list(nodes),original_GPU_start_unix=epoch,service_deadline_unix=epoch+13500,Native_deadline_unix=epoch+14100,GPU_deadline_unix=epoch+14400,keeper_deadline_unix=epoch+15000,Native_cleanup_reserve_s=600,original_8GiB_preweight_512MiB_runtime_floor_unchanged=True,device_initialized_by_check=False)
