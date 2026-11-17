"""Synthetic metadata boundaries only; never read actual data or run native code."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from read_original import COLUMNS,LABEL,DATA_SHA,PUBLICATION,CLAIM_SHA,audit,resources,native_record,epoch_records


def check():
    failures=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):failures.append(name)
        else:raise AssertionError('accepted '+name)
    capture=dict(accounting_columns=COLUMNS,parents={'1103606':[],'1087656':[]})
    missing=audit(Path('/deliberately-unreadable-original-root'),capture)
    assert missing['state']=='ROOT_READER_NOT_TERMINAL_NO_ORIGINAL_READ' and missing['original_file_pins']=={}
    assert missing['planned_examples']==432 and missing['planned_native_encode_calls']==576
    row=['1103606.999',LABEL,'RUNNING','0:0','1','f13r1n11','cpu=8,mem=16G,node=1']
    capture['parents']['1103606']=[row]
    assert audit(Path('/deliberately-unreadable-original-root'),capture)['original_file_pins']=={}
    goodrow=copy.deepcopy(row);goodrow[2]='COMPLETED'
    assert resources(goodrow)['GRES']==0
    for name,index,value in [('GPU',6,'cpu=8,mem=16G,node=1,gres/dcu=1'),('CPU',6,'cpu=4,mem=16G,node=1'),
                             ('memory',6,'cpu=8,mem=8G,node=1'),('node',5,'a01r4n14'),('failed',3,'1:0')]:
        bad=copy.deepcopy(goodrow);bad[index]=value;reject('resources_'+name,lambda bad=bad:resources(bad))
    owner=dict(parent='1103606',node='f13r1n11',step='999',deadline=1600.0,work_deadline=1450.0,
               fd_before=dict(full_fd_inventory=True,fd_count=8,KFD_DRM_holders=[]))
    receipt=dict(passed=True,deadline=1600.0,work_deadline=1450.0,planned_rows={s:144 for s in ('train','dev','blind')},
        planned_native_encode_calls=576,actual_GPU_cards=0,actual_updates=0,weight_open_attempt_hashes=[],
        model_weight_values_read=False,failure_type=None,fd_before=owner['fd_before'],
        fd_after=dict(full_fd_inventory=True,fd_count=8,KFD_DRM_holders=[]),full_fd_inventory_performed=True)
    state=dict(publication=PUBLICATION,claim_SHA256=CLAIM_SHA,epoch=1000.0,deadline=1600.0,
               maximum_invocations=1,replay_allowed=False)
    intent=dict(**state,budget=dict(slurm_seconds=480,inside_seconds=450,client_seconds=470,
        work_deadline=1450.0,slurm_time='00:08:00'),command=['srun','--jobid=1103606','--nodelist=f13r1n11',
        '--gres=none','--cpus-per-task=8','--mem=16384M','--time=00:08:00',
        'timeout --kill-after=10s 450 python inside.py --deadline 1600.0 --work-deadline 1450.0'])
    outcome=dict(epoch=1000.0,deadline=1600.0,accounting_columns=COLUMNS,
        accounting=dict(terminal_visible=True,row=goodrow),current_step_outcome='TERMINAL',rc=0,
        client_timeout=False,owned_client_cleanup_signals=[],receipt_errors=[],passed=True,replay_allowed=False)
    assert epoch_records(state,intent,outcome,owner,receipt,goodrow)['Slurm_seconds']==480
    bad=copy.deepcopy(outcome);bad['deadline']=1700
    reject('epoch_reset',lambda:epoch_records(state,intent,bad,owner,receipt,goodrow))
    bad=copy.deepcopy(intent);bad['budget']['slurm_seconds']=540
    reject('Slurm_cap',lambda:epoch_records(state,bad,outcome,owner,receipt,goodrow))
    bad=copy.deepcopy(owner);bad['parent']='1087656'
    reject('owner_parent',lambda:epoch_records(state,intent,outcome,bad,receipt,goodrow))
    native=dict(state='NATIVE_CPU_INPUT_AND_TRAIN_COLLATOR_PASS',planned_rows={s:144 for s in ('train','dev','blind')},
        completed_rows={s:144 for s in ('train','dev','blind')},actual_native_encode_calls=576,native_rows=432,
        media_SHA_verified=48,max_length=4096,max_pixels=262144,truncation='raise',model_weights_loaded=False,
        rope_dummy_model_on_meta=True,collated_train_rows=2,actual_GPU_cards=0,actual_updates=0,
        versions={'ms-swift':'4.4.2','transformers':'5.12.1','qwen-vl-utils':'0.0.14'},manifest_SHA256=DATA_SHA,
        dev_blind_label_content_opened=False,splits={},label_identity_checks={},processed_media={})
    for split in ('train','dev','blind'):
        native['splits'][split]=dict(rows=144,input_tokens_range=[250,300],image_tokens_range=[150,160],
            supervised_tokens=576 if split=='train' else 0,counts_SHA256='a'*64)
        native['label_identity_checks'][split]=dict(actual_bytes_SHA256_verified=split=='train',content_parsed=split=='train')
    sizes={}
    for i in range(48):
        name='synthetic-image-'+str(i)
        sizes[name]=dict(original_size=[1440,360],bytes=100,SHA256=hashlib.sha256(str(i).encode()).hexdigest())
        native['processed_media'][name]=dict(**sizes[name],grid=[1,32,20],spatial_size=[320,512],image_tokens=160,pixel_values_shape=[640,1536])
    assert native_record(native,receipt,owner,sizes)['examples']==432
    mutations=[('partial_rows',lambda x:x['completed_rows'].__setitem__('blind',143)),
        ('encode_missing',lambda x:x.__setitem__('actual_native_encode_calls',575)),
        ('over4096',lambda x:x['splits']['train'].__setitem__('input_tokens_range',[250,4097])),
        ('blind_supervision',lambda x:x['splits']['blind'].__setitem__('supervised_tokens',1)),
        ('blind_gold_opened',lambda x:x['label_identity_checks']['blind'].__setitem__('content_parsed',True)),
        ('model_materialized',lambda x:x.__setitem__('model_weights_loaded',True)),
        ('missing_meta',lambda x:x.__setitem__('rope_dummy_model_on_meta',False)),
        ('media_missing',lambda x:x['processed_media'].pop('synthetic-image-0'))]
    for name,mutate in mutations:
        bad=copy.deepcopy(native);mutate(bad)
        reject(name,lambda bad=bad:native_record(bad,receipt,owner,sizes))
    bad=copy.deepcopy(receipt);bad['weight_open_attempt_hashes']=['a'*64]
    reject('weight_audit',lambda:native_record(native,bad,owner,sizes))
    bad=copy.deepcopy(receipt);bad['fd_after']['KFD_DRM_holders']=[dict(fd=9,target='/dev/kfd')]
    reject('KFD_post',lambda:native_record(native,bad,owner,sizes))
    return dict(state='LOCAL_READER_SYNTHETIC_BOUNDARIES_PASS',valid_component_cases=5,
        rejected_cases=failures,rejected_count=len(failures),nonterminal_no_original_read_cases=2,
        actual_original_consumed=False,actual_native_calls=0,actual_GPU_calls=0,actual_Slurm_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=check()
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],rejected_count=result['rejected_count'],actual_original_consumed=False)))
