"""Root-only metadata binding AFTER actual question independent terminal. No model/weights read."""
import argparse
import json
from pathlib import Path
from effect_contract import *
from blind_inputs import TARGET, check_bundle
from asset_witness import INIT_IDENTITY, INIT_PIN, INPUT_PIN, check_original

def bind_value():
    source=verify('READ')
    witness=check_original()
    domain=json.loads(read(DOMAIN/'continuous-independent-terminal.json',DOMAIN_GATE))
    question_raw=read(QUESTION_OP/'independent-terminal.json',QUESTION_GATE)
    question=json.loads(question_raw)
    need(domain['passed'] is domain['independent_passed'] is domain['accounting_C0'] is True and
         domain['actual_updates']==25 and domain['fresh_initial_named_SHA256']==NAMED,'actual851 domain terminal required')
    need(question['state']=='GEMMA_QUESTION_ONLY_CONTROL25_INDEPENDENT_TERMINAL_V1' and
         question['passed'] is question['independent_passed'] is question['accounting_C0'] is True and
         question['owned_cleanup_passed'] is True and question['actual_updates']==25 and question['actual_exposures']==800 and
         question['request_id']=='2ea01ac64fa07c326d825f0e073b254b' and
         question['initialization_mode']=='TARGET_CREATED_DOMAIN_INITIAL_REUSE' and
         question['fresh_initial_named_SHA256']==NAMED,'question actual complete independent25 required; NOT_READY is never promoted')
    dpath=DOMAIN_OP/'checkpoint-25-files.json';gpath=QUESTION_OP/'checkpoint-25-files.json'
    dcp=json.loads(read(dpath,DOMAIN_CP))
    gpin=question['checkpoint_manifests']['25']
    need(gpin==QUESTION_CP,'actual1208 question-only checkpoint25 manifest differs')
    gcp=json.loads(read(gpath,gpin))
    cp={arm:checkpoint(value,root,req) for arm,value,root,req in (
        ('domain25',dcp,DOMAIN/'continuous/train/checkpoint-25','247fa3682a114542ac3640fbc726aaae'),
        ('question25',gcp,QUESTION/'continuous/train/checkpoint-25','2ea01ac64fa07c326d825f0e073b254b'))}
    initpin=domain['rank_receipts'][0]['initialization']
    initial=json.loads(read(DOMAIN/'continuous/train/continuous-witness/rank-00/initialization.json',initpin))
    need(initial['named']['SHA256']==NAMED and initial['mode']=='TARGET_FRESH_DOMAIN_ONLY' and
         initial['target_created'] is True and initial['reference_initial_read'] is False,'actual851 initial origin required')
    stage_raw=read(QUESTION/'stage-complete.json',{'bytes': 150566, 'SHA256': '02c31f7cc5b5f5c6dcdc24e173fc2ac3ba18c2acf44c9c1a597827f7d3afff65'});stage=json.loads(stage_raw)
    need(stage['passed'] is True and stage['training_request']==question['request_id'] and
         stage['source_manifest']==question['source_manifest'] and stage['reference_initial']['named_SHA256']==NAMED,
         'actual question source/stage provenance differs')
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
    _,input_manifest=check_bundle(Path(TARGET))
    initial_artifact=initial['artifacts']['target-fresh-initial.pt']
    need(initial_artifact==INIT_PIN and input_manifest==INPUT_PIN,'exact authenticated .3538 initial/input pin differs')
    value=dict(state='ACTUAL_THREE_ARM_GEMMA25_BINDING',request=REQUEST,source_manifest=source,
        domain_terminal=DOMAIN_GATE,question_terminal=pin(question_raw),question_terminal_publication=question['publication'],
        checkpoints=cp,checkpoint_manifests=dict(domain25=dict(path=str(dpath),**DOMAIN_CP),question25=dict(path=str(gpath),**gpin)),
        initial=dict(path=str(DOMAIN/'continuous/train/continuous-witness/rank-00/target-fresh-initial.pt'),
                     file_identity=INIT_IDENTITY,**initial_artifact),initial_named_SHA256=NAMED,
        independent_asset_witness=witness,
        framework_root=str(DOMAIN/'framework'),framework=pin(framework_raw),runtime=pin(runtime_raw),model=weights,
        model_root=model['target_model_directory'],same_compute_host=stage['host'],same_compute_boot_id=stage['boot_id'],
        input_manifest=input_manifest,data_root=TARGET,generation_cap=256,planned_each=36,planned_total=108,
        same_train_FINAL_format_across_adapters=True,domain_actual_tokens='UNKNOWN_ORIGINAL_NOT_MEASURED',
        question_actual_shifted_supervised_tokens=question['actual_shifted_supervised_tokens'],
        actual_inference_runs=0,blind_gold_read=False,dev_inputs_read=False,formal17=False)
    return value

def bind(output):
    return dump(output,bind_value())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(bind(a.output),sort_keys=True))
