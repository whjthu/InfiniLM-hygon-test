"""Root-only metadata binding AFTER actual generic independent terminal. No model/weights read."""
import argparse
import json
from pathlib import Path
from effect_contract import *
from dev_inputs import TARGET

def bind(output):
    source=verify('READ')
    domain=json.loads(read(DOMAIN/'continuous-independent-terminal.json',DOMAIN_GATE))
    generic_raw=read(GENERIC_OP/'independent-terminal.json',GENERIC_GATE)
    generic=json.loads(generic_raw)
    need(domain['passed'] is domain['independent_passed'] is domain['accounting_C0'] is True and
         domain['actual_updates']==25 and domain['fresh_initial_named_SHA256']==NAMED,'actual851 domain terminal required')
    need(generic['state']=='GEMMA_GENERIC64_CONTROL25_INDEPENDENT_TERMINAL_V1' and
         generic['passed'] is generic['independent_passed'] is generic['accounting_C0'] is True and
         generic['owned_cleanup_passed'] is True and generic['actual_updates']==25 and generic['actual_exposures']==800 and
         generic['request_id']=='2e96e8db20ee44c1a6a6a277e66f0d5e' and
         generic['initialization_mode']=='TARGET_CREATED_DOMAIN_INITIAL_REUSE' and
         generic['fresh_initial_named_SHA256']==NAMED,'generic actual complete independent25 required; NOT_READY is never promoted')
    dpath=DOMAIN_OP/'checkpoint-25-files.json';gpath=GENERIC_OP/'checkpoint-25-files.json'
    dcp=json.loads(read(dpath,DOMAIN_CP))
    gpin=generic['checkpoint_manifests']['25']
    need(gpin==GENERIC_CP,'actual1080 original checkpoint25 manifest differs')
    gcp=json.loads(read(gpath,gpin))
    cp={arm:checkpoint(value,root,req) for arm,value,root,req in (
        ('domain25',dcp,DOMAIN/'continuous/train/checkpoint-25','247fa3682a114542ac3640fbc726aaae'),
        ('generic25',gcp,GENERIC/'continuous/train/checkpoint-25','2e96e8db20ee44c1a6a6a277e66f0d5e'))}
    initpin=domain['rank_receipts'][0]['initialization']
    initial=json.loads(read(DOMAIN/'continuous/train/continuous-witness/rank-00/initialization.json',initpin))
    need(initial['named']['SHA256']==NAMED and initial['mode']=='TARGET_FRESH_DOMAIN_ONLY' and
         initial['target_created'] is True and initial['reference_initial_read'] is False,'actual851 initial origin required')
    stage_raw=read(GENERIC/'stage-complete.json');stage=json.loads(stage_raw)
    need(stage['passed'] is True and stage['training_request']==generic['request_id'] and
         stage['source_manifest']==generic['source_manifest'] and stage['reference_initial']['named_SHA256']==NAMED,
         'actual generic source/stage provenance differs')
    model=json.loads(read(D/'MODEL_BINDING.json'))
    weights={}
    for name,expected in model['files'].items():
        path=str(Path(model['target_model_directory'])/name)
        value=stage['reads'][path]
        need({k:value[k] for k in ('bytes','SHA256')}=={k:expected[k] for k in ('bytes','SHA256')},'actual16 model source differs')
        weights[path]=value
    framework_raw=read(DOMAIN/'fresh-framework-files.json')
    runtime_raw=read(DOMAIN/'runtime-binding.json')
    need(pin(framework_raw)['SHA256']==FRAMEWORK_SHA and pin(runtime_raw)['SHA256']==RUNTIME_SHA,'original qualified396/runtime changed')
    stage_dev=read(Path(TARGET)/'stage-complete.json',dict(bytes=968,SHA256='9411e7c43788145a56e3fe1ca20cd163bf0b0f9329ff3f3642a9a8cffec3dd61'))
    value=dict(state='ACTUAL_THREE_ARM_GEMMA25_BINDING',request=REQUEST,source_manifest=source,
        domain_terminal=DOMAIN_GATE,generic_terminal=pin(generic_raw),generic_terminal_publication=generic['publication'],
        checkpoints=cp,checkpoint_manifests=dict(domain25=dict(path=str(dpath),**DOMAIN_CP),generic25=dict(path=str(gpath),**gpin)),
        initial=dict(path=str(DOMAIN/'continuous/train/continuous-witness/rank-00/target-fresh-initial.pt'),
                     **initial['artifacts']['target-fresh-initial.pt']),initial_named_SHA256=NAMED,
        framework_root=str(DOMAIN/'framework'),framework=pin(framework_raw),runtime=pin(runtime_raw),model=weights,
        model_root=model['target_model_directory'],same_compute_host=stage['host'],same_compute_boot_id=stage['boot_id'],
        dev_input_stage=pin(stage_dev),data_root=TARGET,generation_cap=256,planned_each=36,planned_total=108,
        captions_unmodified_format_confound=True,domain_actual_tokens='UNKNOWN_ORIGINAL_NOT_MEASURED',
        generic_actual_shifted_supervised_tokens=generic['actual_shifted_supervised_tokens'],
        actual_inference_runs=0,dev_gold_read=False,blind_read=False,formal17=False)
    return dump(output,value)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(bind(a.output),sort_keys=True))
