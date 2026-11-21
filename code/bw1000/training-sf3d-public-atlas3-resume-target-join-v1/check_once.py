"""Real3 joins plus labelled synthetic metadata-only resume compatibility cases."""
import copy
import hashlib
import json
from pathlib import Path
import time

from target_join3 import G,binding,targets_for_group,local_path,checked
from resume_labels import load_native_contract_code,training_label_identity,domain_contract,check_resume_metadata

D=Path(__file__).resolve().parent
OUT=G/'local-only/training-sf3d-public-atlas3-resume-target-join-v1-original'


def pin(p):
    raw=p.read_bytes();return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def main():
    began=time.monotonic();OUT.mkdir()
    r={'schema':'SF3D_REAL_ATLAS3_JOINS_LABEL_RESUME_COMPATIBILITY_V1','passed':False,'planned_real_joins':3,'completed_real_joins':0,
        'native_encoder_model_feature_tensor_or_training_calls':0,'new_GETs':0,'READY':False,'phase':'original-binding'}
    try:
        b=binding();native=load_native_contract_code();joins={}
        for group,purpose in [('WaterBottle','train'),('Corset','train'),('BoomBox','eval-dev')]:
            target,identity=targets_for_group(group,purpose=purpose)
            if list(target['albedo'].shape)!=[3] or type(target['roughness']) is not float or type(target['metallic']) is not float:raise ValueError('Original target interface differs')
            joins[group]={'identity':identity,'native_target_interface':{'albedo':'float32[3]','roughness':'float','metallic':'float'}};r['completed_real_joins']+=1
        heldout_rejected=False
        try:targets_for_group('BoomBox',purpose='train')
        except ValueError as e:heldout_rejected=str(e)=='Development group cannot enter training'
        if not heldout_rejected:raise ValueError('Development leaked into training')
        train_paths=[local_path(row['input']) for row in b['train']]
        if len(train_paths)<2 or any(not p.is_file() for p in train_paths):raise ValueError('Real native >=2 input premise still missing')
        r['phase']='metadata-only-native-compatibility-tests'
        # Explicit fixture values for fields which cannot be actual without a
        # model/runtime. They are never exported as actual identity/qualification.
        fixture_identity={key:hashlib.sha256(('CPU_METADATA_FIXTURE_NOT_ACTUAL_'+key).encode()).hexdigest() for key in ['backend','weights','sources','runtime','imports']}
        config={'seed':20260901,'warmup_steps':5,'max_steps':20,'query_points':2048}
        contract,identity=domain_contract(native,b,config,actual_native_identity=fixture_identity,actual_device_ids=[0])
        metadata=native['_sf3d_checkpoint_metadata'](contract,20)
        check_resume_metadata(native,metadata,contract)
        rejects={}
        legacy=copy.deepcopy(contract);legacy['config_sha256']=identity['native_base_config_SHA256']
        legacy_metadata=native['_sf3d_checkpoint_metadata'](legacy,20)
        changed=copy.deepcopy(b);changed['train'][1]['target_pin']['SHA256']='0'*64
        changed_contract,_=domain_contract(native,changed,config,actual_native_identity=fixture_identity,actual_device_ids=[0])
        swapped=copy.deepcopy(b);swapped['train'].reverse()
        swapped_contract,_=domain_contract(native,swapped,config,actual_native_identity=fixture_identity,actual_device_ids=[0])
        bad_cursor=copy.deepcopy(metadata);bad_cursor['next_data_cursor']=1-bad_cursor['next_data_cursor']
        for name,saved,expected in [('legacy_cache_without_label_identity',legacy_metadata,contract),('changed_training_label_bundle',metadata,changed_contract),
            ('changed_training_order',metadata,swapped_contract),('changed_resume_data_cursor',bad_cursor,contract)]:
            try:check_resume_metadata(native,saved,expected)
            except native['SF3DCheckpointValidationError'] as e:rejects[name]={'rejected':True,'type':type(e).__name__,'reason':str(e)}
            else:raise ValueError('Required metadata-only case accepted: '+name)
        dev_change=copy.deepcopy(b);dev_change['development'][0]['target_pin']['SHA256']='f'*64
        if training_label_identity(dev_change)[0]!=identity['training_label_input_SHA256']:raise ValueError('Development label affects training identity')
        moved=copy.deepcopy(b)
        for row in moved['train']:row['input']='different/physical/root.png';row['target']='different/physical/root.npz'
        if training_label_identity(moved)[0]!=identity['training_label_input_SHA256']:raise ValueError('Path-only relocation alters data identity')
        preserved=G/'results/bw1000/2026-11-21/training-sf3d-public-atlas-target-feature-wiring-native-gap-local.json'
        checked(preserved,{'bytes':3891,'SHA256':'d5996f661afd40cc990f470cec8e572cd85abb25cb057eafbc57125e29bd5d17'})
        private={'schema':'SF3D_REAL_THREE_INPUT_TARGET_PLAN_NO_FEATURES_V1','joins':joins,'training_label_identity':identity['label_document'],
            'training_label_identity_SHA256':identity['training_label_input_SHA256'],'resume_hash_composition':{'native_base_config_SHA256':identity['native_base_config_SHA256'],'domain_config_SHA256':identity['domain_config_SHA256'],
            'native_input_SHA256':identity['native_ordered_input_SHA256']},'model_runtime_identity':'UNKNOWN; requires actual native loader','feature_widths':'UNKNOWN','native_features':False}
        raw=(json.dumps(private,sort_keys=True,indent=2)+'\n').encode();(OUT/'three-input-target-resume-plan-private.json').write_bytes(raw)
        r.update(passed=True,state='REAL_TRAIN2_DEV1_TARGETS_AND_LABEL_BOUND_METADATA_COMPATIBILITY_PASS_FEATURES_ABSENT',phase='complete',
            actual_groups={'train':2,'dev':1,'blind':0},real_input_minimum_two_passed=True,heldout_to_train_rejected=True,
            actual_original_T11_checkpoint_or8RGBA_read=False,native_math_or_original_backend_modified=False,original_two_object_negative_preserved=True,
            independent_training_label_identity_SHA256=identity['training_label_input_SHA256'],native_ordered_input_SHA256=identity['native_ordered_input_SHA256'],
            metadata_contract_checks={'fixture_only_model_runtime_device_identity':True,'actual_model_identity_qualified':False,
                'normal_label_bound_metadata_passes':1,'negative_cases':rejects,'dev_labels_do_not_affect_training_identity':True,
                'physical_path_relocation_does_not_affect_training_identity':True,'full_checkpoint_tensor_integrity_optimizer_or_rng_validated':False,
                'actual_old_checkpoint_loaded':False,'synthetic_feature_or_state_tensors_created':False},
            frozen_original_metadata_functions=native['__proof__'],private_plan={'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()},
            actual_wall_seconds=time.monotonic()-began,training_GO=False,GPU_GO=False,Model_GO=False,formal17_GO=False)
    except Exception as e:
        import traceback
        raw=(json.dumps({'phase':r['phase'],'type':type(e).__name__,'reason':str(e),'trace':traceback.format_exc()},indent=2)+'\n').encode();(OUT/'failure-private.json').write_bytes(raw)
        r.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,failure={'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()})
    (OUT/'result.json').write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:r[k] for k in ['passed','state','planned_real_joins','completed_real_joins','native_encoder_model_feature_tensor_or_training_calls','READY']},sort_keys=True))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
