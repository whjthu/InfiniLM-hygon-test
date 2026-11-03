"""Actual Bash/configure_process/purelayout on a genuine >=32CPU entry.

Only hostname and CAPI result/communicator bytes are recorded leaves. No
Torch/SDK import, Native dlopen, model, weights or GPU work is performed.
"""
import base64
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

S=Path(__file__).resolve().parents[1];D=S/'deployment-v1';P=S/'provider-v2'
R=S.parents[1]
B=R/'candidates/GLM-W4-fewshot-quality-r444-nodes-CPU-v1'
sys.path.insert(0,str(D));sys.path.insert(0,str(P))
import controller as C
from preflight import check as preflight
from public_glm_native.worker import configure_process
from public_glm_native.topology_r404 import read_layout,create_context,validate_layout,validated_nodes_r444


def main():
    B.mkdir(parents=True,exist_ok=True);out=B/'CPU-result.json';temp=out.with_suffix('.tmp')
    original=dict(os.environ);mask=sorted(os.sched_getaffinity(0));model=Path('/old/models/ZhipuAI-GLM-5.3/config.json');raw=model.read_bytes();config_object=json.loads(raw)
    q=dict(completed=False,passed=False,actual_CPU=original['SLURM_JOB_ID']+'.'+original['SLURM_STEP_ID'],
        actual_cpu_hostname=socket.gethostname(),actual_entry_mask=mask,SLURM_CPUS_PER_TASK=original.get('SLURM_CPUS_PER_TASK'),
        Native_dlopen=False,SDK_executed=False,GPU=0,Model=False,weights_loaded=False,compile=False,
        cases=[],preflight=[],errors=[],recorded_leaves=['prospective GPU-node hostname','CAPI create callable and communicator ID bytes'],
        actual_model_config_SHA256=hashlib.sha256(raw).hexdigest(),CPU922_HTTP_and_CPU918_encoding_repeated=False)
    def save():temp.write_text(json.dumps(q,indent=2)+'\n');temp.replace(out)
    save();primary=None
    try:
        assert len(mask)>=32, 'Do not manufacture the original real entrymask qualification'
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        deploy_sha=C.sha(D/'SOURCE_MANIFEST.json');source_sha=C.sha(S/'nodes-CPU-v1/SOURCE_MANIFEST.json')
        q.update(source_manifest_SHA256=source_sha,deployment_manifest_SHA256=deploy_sha,provider_manifest_SHA256=C.sha(P/'SOURCE_MANIFEST.json'))
        all_cases=[]
        for index in range(8):
            slot=C.configure_slot(index);m=C.verify(deploy_sha);config=json.loads(C.CONFIG.read_bytes())
            proof=preflight(D,P,m,slot);q['preflight'].append(proof);all_cases+=proof['planned_case_ids'];save()
            for stage in[0,1]:
                case=dict(slot=index,stage=stage,passed=False,recorded_hostname_leaf=slot['nodes'][stage],CAPI_recording_calls=0)
                q['cases'].append(case);save()
                env=C.gpu_environment(original,1.,m)
                # Real CPU Slurm job/step remain unchanged. Only the prospective
                # GPU role's PROCID is an explicit source-contract leaf.
                env.update(SLURM_JOB_ID=original['SLURM_JOB_ID'],SLURM_STEP_ID=original['SLURM_STEP_ID'],SLURM_PROCID=str(stage),GLM_R444_ENV_ONLY='1')
                proc=subprocess.run(['bash',str(D/'inside.sh')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
                case.update(Bash_returncode=proc.returncode,Bash_stderr=proc.stderr.decode(),Bash_stdout_SHA256=hashlib.sha256(proc.stdout).hexdigest());save()
                assert proc.returncode==0,proc.stderr.decode()
                marker=b'GLM_R444_ENV_BEGIN\0';assert proc.stdout.count(marker)==1
                values=proc.stdout.split(marker,1)[1].split(b'\0');assert values[-1]==b'';values.pop();assert len(values)%2==0
                captured={values[i].decode():values[i+1].decode()for i in range(0,len(values),2)}
                config.update(stage=stage,run=slot['run'],source_manifest_SHA256=m['provider_source_SHA256'],
                    provider_config_SHA256=m['config_SHA256'],actual_step=q['actual_CPU'],gpu_start_epoch=1.)
                os.environ.clear();os.environ.update(original);os.environ.update(captured)
                configure_process(config)  # unchanged genuine >=32 affinity guard
                expected_nodes=json.loads(os.environ['GLM_EXPECTED_NODES_R444']);assert expected_nodes==slot['nodes']
                layout=read_layout(os.environ,model,slot['nodes'][stage])
                assert layout['config_SHA256']==config['config_SHA256']and(layout['layer_begin'],layout['layer_end'])==((0,38)if stage==0 else(38,78))
                captured['GLM_EXPECTED_NODES_R444']=os.environ['GLM_EXPECTED_NODES_R444']
                case['Bash_then_actual_configure_env']=captured;case['layout']=layout
                calls=[];identifiers={k:base64.b64encode((k+'-recorded-comm').encode()).decode()for k in ['world','control']}
                identifiers['tp']=[base64.b64encode(('tp%d-recorded-comm'%s).encode()).decode()for s in range(2)]
                class RecordingCAPI:
                    def modern_glm_model_create(self,*args):
                        assert len(args)==8 and args[:4]==(0,stage,1,32768)
                        assert json.loads(args[4])==config_object
                        for value,key in zip(args[5:],['world','tp','control']):
                            expected=base64.b64decode(identifiers[key][stage]if key=='tp'else identifiers[key])
                            assert isinstance(value,ctypes.Array)and value.raw==expected+b'\0'
                        calls.append(dict(rank=args[0],stage=args[1],rows=args[2],capacity=args[3],argument_count=len(args)))
                        return 'recorded-not-a-Native-context'
                result=create_context(RecordingCAPI(),0,stage,1,32768,config_object,identifiers,os.environ,slot['nodes'][stage])
                assert result=='recorded-not-a-Native-context'and len(calls)==1
                assert sorted(os.sched_getaffinity(0))==mask
                case.update(passed=True,CAPI_recording_calls=len(calls),original_eight_argument_create_contract=calls[0],actual_entry_mask_after=sorted(os.sched_getaffinity(0)));save()
        assert len(all_cases)==len(set(all_cases))==256
        q['fixed_eight_shards_cover_all256_once_and512paired']=True
        rejected=[]
        try:validate_layout(os.environ,config_object,'not-a-frozen-node')
        except AssertionError:rejected.append('wrong hostname')
        else:raise AssertionError('Wrong prospective hostname was accepted')
        try:validated_nodes_r444(['a02r4n12','a02r4n12'])
        except AssertionError:rejected.append('foreign/duplicate config nodes')
        else:raise AssertionError('Wrong nodes were accepted')
        q.update(passed=True,negative_contracts=rejected,actual_same_configure_and_layout_consumed=True,
            genuine_at_least32CPU_entry=True,necessary_create_args_recording_only=True,Native_math_changed=False,GPU_node_qualification=False)
    except BaseException as error:
        primary=error;q['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:
        os.environ.clear();os.environ.update(original)
        q['original_environment_restored']=dict(os.environ)==original;q['entry_mask_unchanged']=sorted(os.sched_getaffinity(0))==mask
        q['forbidden_modules_present']=[n for n in ['torch','_infinilm','transformers']if n in sys.modules]
        q['forbidden_maps']=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(n in line for n in ['libGLM_V_on_demand_Model','libGLM_PP4_registry','_infinilm.cpython','libinfinirt.so','libinfiniop.so'])]
        if q['forbidden_modules_present']or q['forbidden_maps']:q['passed']=False
        q['completed']=True;save()
    print(json.dumps(dict(passed=q['passed'],actual_CPU=q['actual_CPU'],path=str(out),SHA256=hashlib.sha256(out.read_bytes()).hexdigest())))
    if primary:raise primary
    return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
