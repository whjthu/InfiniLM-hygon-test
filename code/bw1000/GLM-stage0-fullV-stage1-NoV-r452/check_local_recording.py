"""Only new HOST policy/census/ready/collector consumers; all Native leaves recorded."""
from pathlib import Path
import contextlib,copy,ctypes,hashlib,importlib.util,io,json,os,sys,tempfile
from unittest.mock import patch
D=Path(__file__).resolve().parent;P=D/'provider';sys.path.insert(0,str(P))
from public_glm_native.cache_policy_r452 import for_config,global_metadata,validate_census,VALUE_BYTES
from public_glm_native.worker import configure_process
from public_glm_native.model_admission_r404 import constructor_snapshot,NATIVE_SHA
from public_glm_native.owner_gates import NativeGates
from public_glm_native.protocol import Run
from public_glm_native import service_policy
from public_glm_native.topology_r404 import validate_layout
from public_glm_native.cli import main as actual_cli
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reject(fn):
    try:fn()
    except(AssertionError,ValueError,RuntimeError):return
    raise AssertionError('Negative case accepted')

def census(policy):
    layers=[];total={}
    for layer in range(policy.layer_begin,policy.layer_end):
        shapes={'latent':[1,32768,512],'rotary':[1,32768,64],'projected_key':[1,32768,8,256]}
        if policy.noV_flag==0:shapes['projected_value']=[1,32768,8,256]
        tensors=[]
        for k,shape in shapes.items():
            n=2
            for v in shape:n*=v
            tensors.append(dict(category=k,shape=shape,bytes=n,unique_start_address=True));total[k]=total.get(k,0)+n
        layers.append(dict(layer=layer,tensors=tensors))
    return dict(layer_begin=policy.layer_begin,layer_end=policy.layer_end,rows=1,capacity=32768,loaded=False,cache_layers=layers,cache_category_bytes=total,cache_unique_start_bytes=sum(total.values()))

def run():
    results=[];mixed=json.loads((P/'config.mixed.template.json').read_bytes());default=json.loads((P/'config.template.json').read_bytes());sha=H(P/'SOURCE_MANIFEST.json')
    assert[for_config(default,s).noV_flag for s in range(2)]==[1,1]
    legacy={k:v for k,v in default.items()if k not in['cache_policy_r452','cache_stage_flags_r452']};assert for_config(legacy,0).noV_flag==1
    assert[for_config(mixed,s).noV_flag for s in range(2)]==[0,1]
    spec=importlib.util.spec_from_file_location('r452_collector',D/'collect_cache_identity.py');collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    spec=importlib.util.spec_from_file_location('r452_env',P/'cache_env.py');envmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(envmod)
    owners=[];constructors=[];records=[];base=dict(mixed,run='CPU.recording-only',actual_step='recording.0',source_manifest_SHA256=sha,provider_config_SHA256=H(P/'config.mixed.template.json'),gpu_start_epoch=0.)
    for stage in range(2):
        c=dict(base,stage=stage);p=for_config(c,stage)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):assert envmod.main(['--config',str(P/'config.mixed.template.json'),'--config-sha',H(P/'config.mixed.template.json'),'--stage',str(stage)])==0
        assert output.getvalue().strip()==str(p.noV_flag)
        cli_output=io.StringIO()
        with contextlib.redirect_stdout(cli_output):rc=actual_cli(['--config',str(P/'config.mixed.template.json'),'--config-sha',H(P/'config.mixed.template.json'),'--source-sha',sha,'--stage',str(stage),'--run','CPU.source-only','--actual-step','recording.0','--gpu-start-epoch','0'])
        assert rc in(None,0)and json.loads(cli_output.getvalue())['source_and_schema_valid']is True
        for rank in range(8):
            env=dict(SLURM_PROCID=str(stage),SLURM_JOB_ID='recording',SLURM_STEP_ID='0',INFINILM_OWNER_CPU_WIDTH_R2392='4',GLM_EXACT_BANKS_R4188='0',GLM_V_ON_DEMAND_R403=str(p.noV_flag),GLM_CACHE_POLICY_R452=p.policy,LD_PRELOAD='/refactor/candidates/blas-policy-r271/libblas_policy_r271.so',LD_LIBRARY_PATH='/refactor/install/hygon-glm-5f91ad7-r3672/lib:')
            with patch.dict(os.environ,env,clear=True),patch.object(os,'sched_getaffinity',lambda _:set(range(32)),create=True):
                configure_process(c)
                model=dict(num_hidden_layers=78,hidden_size=6144,mlp_layer_types=['moe']*78,indexer_types=['full']*78)
                layout=validate_layout(os.environ,model,c['nodes'][stage]);assert layout['cache_policy']==p.metadata()
                report=dict(cache_policy=p.metadata());raw=census(p)
                class Function:
                    def __call__(self,ctx):assert ctx==0x1234567887654321;return json.dumps(raw).encode()
                lib=type('RecordingNative',(),dict(modern_glm_constructor_census_r4158=Function()))()
                with tempfile.TemporaryDirectory()as tmp:
                    path=Path(tmp)/'constructor.json';constructor_snapshot(lib,0x1234567887654321,stage,rank,report,report_path=path,provider_reader=lambda _:dict(library_SHA256=NATIVE_SHA,recording_not_dlopen=True));record=json.loads(path.read_bytes())
                assert report['constructor_cache_verified']and report['constructor_noV_verified']==bool(p.noV_flag)
                class StopAfterBank(Exception):pass
                observed=[];gate=NativeGates(torch=None,rank=rank,stage=stage,stream=None,config=c,publish_failure=lambda _:None)
                def counts(lib,name,n,ctx=None):
                    observed.append(name)
                    if name=='modern_glm_exact_banks_stats_r4188':return[p.exact_bank_count,0,p.exact_bank_count,0]
                    raise StopAfterBank()
                gate._counts=counts
                try:gate.before_load(type('B',(),dict(lib=lib))(),1,report)
                except StopAfterBank:pass
                assert observed==['modern_glm_exact_banks_stats_r4188','modern_glm_service_memory_stats_r4279']
                assert report['exact_bank_dispatch_r4188'][0]==(152 if stage==0 else 120)
                assert 'all16_preweight_8GiB_passed'not in report # original memory/CCL leaf not executed
                report.update(source_manifest_SHA256=sha,completed=True,normal_shutdown=True,worker_shutdown_completed=True,post_close_Torch_liveness=True,all16_preweight_8GiB_passed=True,all16_loaded_and_postload_4GiB_passed=True,service_requests=[dict(cache_policy=p.metadata())])
                owners.append(report);constructors.append(record)
                bad=dict(env,GLM_V_ON_DEMAND_R403=str(1-p.noV_flag))
                with patch.dict(os.environ,bad,clear=True):reject(lambda:configure_process(c))
            affinity=dict(actual_owner_report=dict(owners=[dict(device=rank,changes=1,verified_mask=','.join(str(x)for x in range(rank*4,rank*4+4)))]))
            records.append(dict(identity={},stage=stage,rank=rank,Native_SHA256=NATIVE_SHA,config_SHA256=c['config_SHA256'],loaded=True,postload_4GiB=True,service_agreement=True,affinity=affinity,cache_policy=p.metadata()))
        results.append(dict(stage=stage,all8_actual_configure_layout_census=True,projected_value_bytes=p.projected_value_bytes,exact_bank_count=p.exact_bank_count,Native_executed=False))
    with tempfile.TemporaryDirectory()as tmp:
        c=dict(base,runtime_directory=tmp);r=Run(c)
        for q in records:q['identity']=r.identity()
        assert r.validate_ready_records(records)==records
        broken=copy.deepcopy(records);broken[3]['cache_policy']=for_config(c,1).metadata();reject(lambda:r.validate_ready_records(broken))
    ready=dict(cache_policy=global_metadata(mixed),Native_SHA256=NATIVE_SHA)
    assert collector.validate_originals(mixed,sha,owners,constructors,records,ready)['passed']
    broken=copy.deepcopy(constructors);broken[0]['census']['cache_layers'][0]['tensors'][-1]['bytes']-=2;reject(lambda:collector.validate_originals(mixed,sha,owners,broken,records,ready))
    bad=copy.deepcopy(mixed);bad['cache_stage_flags_r452']=[0,0];reject(lambda:for_config(bad,0))
    bad=census(for_config(mixed,1));bad['cache_layers'][0]['tensors'].append(dict(category='projected_value',shape=[1,32768,8,256],bytes=VALUE_BYTES,unique_start_address=True));reject(lambda:validate_census(bad,for_config(mixed,1)))
    token=service_policy._REQUEST_PROFILE.set('glm-nov16-w4')
    try:reject(lambda:service_policy.request_profile({},'glm-mixedV16-w4'))
    finally:service_policy._REQUEST_PROFILE.reset(token)
    assert service_policy.profile_for_window('glm-mixedV16-w4',2)=='glm-mixedV16-w2'
    assert not any(n in sys.modules for n in['torch','_infinilm','flash_attn'])
    return dict(passed=True,source_only=True,recording_only=True,Native_dlopen=False,GPU=0,weights=False,compile=False,actual_worker_CLI_layout_census_ready_collector_new_boundaries=True,stages=results,unchanged_CCL_and_memory_admission_not_executed=True,negative_checks=dict(conflicting_constructor_flag_rejected=True,wrong_stage_V_census_rejected=True,wrong_V_bytecount_rejected=True,wrong_TP_ready_policy_rejected=True,wrong_config_stage_flags_rejected=True,request_cache_family_switch_rejected=True),physical_fit_peak_Model_quality_or_performance_proven=False)

if __name__=='__main__':print(json.dumps(run(),indent=2))
