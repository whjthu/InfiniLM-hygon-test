"""Actual final-shell environment -> frozen Settings/binding CPU leaf, no Slurm."""
from pathlib import Path
import ctypes as C,hashlib,json,os,sys,time
D=Path(__file__).resolve().parent;runtime=D.parent/'runtime-v2';sys.path.insert(0,str(runtime))
import bootstrap
from native_binding import declare,SIGNATURES
from check_entry_CPU import RecordingLibrary
from contract import validate_config,frozen_fixtures,layout
import numpy as np
def main():
 assert os.environ['KIMI410_PREFLIGHT_CPU_ONLY']=='1'and all(os.environ.get(k)==''for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert 'torch'not in sys.modules,'Torch shell metadata lookup was a separate process, not a device Runtime'
 settings=bootstrap.Settings(os.environ);pins=json.loads((runtime/'bindings.json').read_bytes())
 claim=Path(os.environ['KIMI410_PUBLISHED_MODEL_CLAIM']);assert bootstrap.sha(claim)==os.environ['KIMI410_PUBLISHED_MODEL_CLAIM_SHA256']and json.loads(claim.read_bytes())['Model_GO']is False
 assert os.environ['KIMI410_PREFLIGHT_FINAL_ARGC']=='4'and os.environ['KIMI410_PREFLIGHT_FINAL_COMMAND']=='python -B -u /refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v2/bootstrap.py'
 bootstrap.verify_bundle(runtime,os.environ['KIMI410_RUNTIME_SOURCE_SHA256'])
 specs=json.loads((runtime/'CAPI_PROVIDERS.json').read_bytes())['symbols'];assert set(specs)==set(SIGNATURES)and len(specs)==20
 provider_files={v['path']:v['SHA256']for v in specs.values()};assert len(provider_files)==2
 for path,digest in provider_files.items():assert bootstrap.sha(path)==digest,path
 for k,v in pins['required_environment'].items():assert os.environ.get(k)==v,(k,'actual production shell output')
 for k,v in dict(INFINILM_KIMI_BOUNDED_BF16_R6054='0',INFINILM_KIMI_N4_TILE_R8167='0',INFINILM_KIMI_W2_PAIR_R9280='0',KIMI_TRUE_DRAFT_BATCH_R9142='0').items():assert os.environ[k]==v,k
 model=Path(pins['model_view']['path']);cfg=json.loads((model/'config.json').read_bytes());validate_config(cfg)
 fp=json.loads((runtime/'FIXTURE_PINS.json').read_bytes());fixtures=frozen_fixtures(fp['frozen_inputs_path'],fp)
 mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];outputs=json.loads((runtime/'CPU_recorded_outputs.json').read_bytes())['outputs']
 raw=declare(RecordingLibrary(settings.node,mapping,outputs));ctx=raw.modern_kimi_engine_create_ordinary_tp8_pp4_r410(json.dumps(cfg).encode(),settings.node,settings.head.encode(),settings.port)
 assert ctx==0x100000000+settings.node+1;raw.reset_count=1
 ids=fixtures[0]['prompt_tokens'][:2];buffers=[np.asarray([1],np.int32),np.asarray([0],np.int32),np.asarray([2],np.int32),np.asarray([0,2],np.int32),np.asarray(ids,np.int64),np.arange(2,dtype=np.int64),np.full((1,163840),0x7fc0,np.uint16),np.full(1,-1,np.int64),np.full(1,-1,np.int32)]
 rc=raw.modern_kimi_engine_forward_ordinary_r410(ctx,0,2,1,3,*[C.c_void_p(a.ctypes.data)for a in buffers]);assert rc==0 and len(raw.forward_calls)==1;assert raw.modern_kimi_engine_destroy_r2475(ctx)==0 and raw.destroyed==1
 q=dict(completed=True,passed=True,CPU_only=True,Slurm_commands=0,GPU=0,Native_dlopen=False,Runtime_created=False,model_payload_loaded=False,node=settings.node,actual_target_shell_sha256=hashlib.sha256((runtime/'target_inside.sh').read_bytes()).hexdigest(),actual_ordinary_Settings=True,static_20map_files_SHA_checked=True,dynamic_dladdr_qualified=False,CPU836_failed_no_dynamic_receipt=True,actual_layout_contract=layout(settings.node),CABI_factory_4arg_and_context64=True,CABI_forward_14arg_9real_CPU_buffers=True,production_preengine_replaced=False,whole_preengine_CPU826_inherited=True,required_environment_observed={k:os.environ[k]for k in pins['required_environment']},typed_original_epoch=settings.start,typed_native_deadline=settings.deadline,typed_native_s=settings.native_s,scope='Final exec is the only recording leaf. Exact frozen shell exports, actual Settings and CABI ABI consumers; no real Native/Model/device, no inherited GPU qualification.')
 Path(os.environ['KIMI410_PREFLIGHT_RESULT']).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(node=settings.node,passed=True,CPU_only=True,Slurm_commands=0)))
if __name__=='__main__':main()
