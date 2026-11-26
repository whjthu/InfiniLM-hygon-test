"""Reconcile the original Gemma domain adapter and recovery using bounded metadata only."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re


HERE=Path(__file__).resolve().parent
REPO=HERE.parent
CORE='code/bw1000/gemma-usgs-confirmatory96-native-owner-v2/core/'
CP25={'bytes':2977,'SHA256':'897b5aadef83c6c4cdab76d2a3d9a24c83e3eb47c6e97a48e7f71e8e24b3515c'}
SCOPE='247fa3682a114542ac3640fbc726aaae'
INITIAL_NAMED='e6b624e3342df4d84751be5b65c2722137ff17109f2dfb18db8c5fd4c916b98c'
ADAPTER_CFG='3453c728728e4eefb7704b62a4e727278f4ebdf9a94f98b67ba146c8be9a16bc'
MEMBER_PINS={
    'adapter_config.json':{'bytes':1177,'SHA256':ADAPTER_CFG},
    'adapter_model.safetensors':{'bytes':119280712,'SHA256':'5c130885be6edaa0541d337d5e0e166d5cdb1b53d72eae1180009f2791e7ee05'},
}
SHA=re.compile(r'[0-9a-f]{64}')
FILES={
    'adapter_readback':('local-only/gemma-cp25-original851-adapter-member-pins-readonly-v1.json',856,'e5d1f57714ac326cec89f20c97f1352953a68bde4dc2ea81624a98467e8719ea'),
    'training':('results/bw1000/2026-11-22/training-gemma-usgs72-v2-original851-eight-rank-25-updates-terminal.json',2889,'53e393df525443d7c5c44b1dee8633f30a389610d9a72dcf988d094be6d260d2'),
    'seal':('results/bw1000/2026-11-26/training-gemma-3682-confirmatory96-288-independent-seal-score-pending.json',1926,'a6f0f53d78c29301088d399fc546c3474148678a6180278af82e30ae22176dbd'),
    'authenticated_seal':('local-only/gemma96-3682-3696-once-actual-seal-metadata-c1662f41c38d420683d5e4fcbcfc967a/receipt.json',1577,'279b2fa0e0e67bc888a72d061f900d8e8a03ca9265b296643cc869981de57c6f'),
    'effect':('results/bw1000/2026-11-27/training-gemma-usgs-confirmatory96-original3682-first-offline-score.json',7722,'d36d05412bd96f5437d079fde7c43bf2cf858a8aff69c3b5cbcc1851a8f21fd0'),
    'statistics':('tools/gemma_confirmatory96_statistics.py',4988,'c29b4e98e85efc37d9a891ea2f4fc9c60623a15e9371ce499736735560fe7f89'),
    'core_manifest':(CORE+'SOURCE_FILES.json',1778,'0f69af4e74ca193d5acf517b9f252b0957b7bc1dfe4e143d469a94745f6f0d49'),
    'infer':(CORE+'infer_blind36.py',19367,'9136cd0aae6278d5eef512aecf720b680ffba5c93e71293d3f2320b7c0b5acda'),
    'slots':(CORE+'slots.py',3246,'018e05d8157b0191448500fcbc785f7f97cb386e814608d6b91bdc4782c20d69'),
    'independent_reader':(CORE+'read_terminal.py',5323,'a79a3bfc7a0b8397d6cc8d8c11a921a4b5f8f54d4bbfc3c9c3ba2e44943252c9'),
    'base16':(CORE+'MODEL_BINDING.json',5096,'46428e251757e386835e2832a32308a6a07178d9ef776a3ea1b70062c5cf77b3'),
    'thresholds':('code/bw1000/gemma-usgs-confirmatory96-native-owner-v2/EVALUATION_THRESHOLDS.json',1968,'f0c33a24feb9d48ff84be0c504ed10427ce8ca137e28d752f21f2db2f009abd9'),
    'resume':('results/bw1000/2026-11-22/training-gemma-usgs72-resume20to25-original860-eight-rank-terminal-numerical-fail.json',2749,'e385031e45863ce7ae1849150f47edd523181099544891017f4f0050f363fc67'),
    'boundary':('results/bw1000/2026-11-22/training-gemma-original860-first-forward-boundary-CPU890.json',5048,'08b1711f38d94a42f5a952f42917922b4bd88f909e80d6e1e696675f5bc38cfe'),
    'boundary_analysis':('analysis/training-gemma-original860-first-forward-boundary-v1.json',5597,'cef059eba0e6f8f85a4c2c262030cac28cfb539bb8d267752c5d3d617c06321b'),
}


def need(ok,code):
    if not ok:raise ValueError(code)


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def read(path,expected):
    need(path.is_file() and not path.is_symlink(),'REGULAR_METADATA_ONLY')
    with path.open('rb') as f:raw=f.read(expected['bytes']+1)
    need(pin(raw)==expected,'EXACT_METADATA_DIGEST_REQUIRED')
    return raw


def read_sources():
    values={}
    for key,(name,size,digest) in FILES.items():
        raw=read(REPO/name,dict(bytes=size,SHA256=digest))
        values[key]=json.loads(raw) if name.endswith('.json') else raw
    core=values['core_manifest']['files']
    need(all(core[Path(FILES[k][0]).name]==pin(values[k]) for k in ('infer','slots','independent_reader')),
         'EXACT_PINNED_3682_ADAPTER_AND_SEAL_SOURCE_LINEAGE')
    return values


def checkpoint_files(data):
    need(pin(data)==CP25,'EXACT_ORIGINAL_851_CHECKPOINT25_MANIFEST')
    value=json.loads(data)
    names={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json',
           'optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))}
    need(value['request_id']==SCOPE and value['step']==25 and value['required_members']==14 and
         set(value['files'])==names,'EXACT_ORIGINAL_851_FOURTEEN_CHECKPOINT_MEMBERS')
    files=value['files']
    need(files['adapter_config.json']['SHA256']==ADAPTER_CFG,'ORIGINAL_NATIVE_ADAPTER_CONFIG_PIN')
    need(all(type(row['bytes']) is int and row['bytes']>0 and SHA.fullmatch(row['SHA256']) for row in files.values()),
         'CHECKPOINT_MEMBER_EXTENT_AND_SHA')
    return {key:{k:files[key][k] for k in ('bytes','SHA256')} for key in ('adapter_config.json','adapter_model.safetensors')}


def adapter_members(envelope):
    value=envelope['response']
    need(envelope['rc']==0 and value['request']==SCOPE and value['step']==25 and
         value['checkpoint_metadata_bytes_read']==CP25['bytes'] and value['original_manifest']==CP25 and
         value['manifest_members']==14 and value['weight_bytes_read']==value['adapter_config_bytes_read']==
         value['new_GPU_steps']==0 and set(value['adapter_members'])==set(MEMBER_PINS),
         'EXACT_851_READONLY_MEMBER_STAT_RECEIPT_REQUIRED')
    for name,row in value['adapter_members'].items():
        need({field:row[field] for field in ('bytes','SHA256')}==MEMBER_PINS[name] and
             row['current_inode_size_mtime_ctime_match'] is True and row['local_mount_device_rebased'] is True,
             'ORIGINAL_ADAPTER_MEMBER_STAT_LINEAGE_REQUIRED')
    return {name:dict(row) for name,row in MEMBER_PINS.items()}


def recover(observed,fail,analysis):
    ranks=observed['observations']['checked_ranks']
    pairs=[(r['rank'],x) for r in ranks for x in r['first_four']]
    unequal=[(rank,p) for rank,p in pairs if p['exact_loss'] is False]
    need([r['rank'] for r in ranks]==[0,1] and len(pairs)==8 and all(p['exact_input'] for _,p in pairs)
         and [(rank,p['tail_ordinal']) for rank,p in unequal]==[(0,1)]
         and all(r['checkpoint_rng_load_saved_equal'] is True and r['restored_state_equal']==['adapter','optimizer','scheduler']
                 and r['state_differences_after_update_21']==['adapter','optimizer'] for r in ranks)
         and fail['numeric_pass'] is fail['independent_summary']['exact_paired_recovery'] is False
         and fail['independent_summary']['compared_tail_microbatch_pairs']==160
         and fail['independent_summary']['actual_updates']==5
         and observed['observations']['forward_entry_rng_observed'] is False and analysis['causal_diagnosis']=='UNKNOWN',
         'ACTUAL_860_FAILURE_AND_FIRST_FORWARD_BOUNDARY_REQUIRED')
    first=unequal[0][1]
    return dict(step='1103606.860',numerically_exact=False,actual_resumed_updates=5,compared_tail_inputs=160,
                all_compared_inputs_equal=True,first_observed_divergence=dict(rank=0,callback_global_step=20,
                    next_update=21,tail_ordinal=1,continuous_loss=first['continuous_loss'],resumed_loss=first['resumed_loss'],
                    loss_abs_delta=first['absolute_loss_gap']),local_window_exact_inputs=8,local_window_exact_losses=7,
                at_checkpoint_rng_load_equal=True,forward_entry_RNG=None,cause=None,
                post_update21_diff=['adapter','optimizer'])


def project(values,member_pins=None):
    proven=adapter_members(values['adapter_readback'])
    if member_pins is not None:
        need(member_pins==proven,'ORIGINAL_MANIFEST_AND_READBACK_DISAGREE')
    t,s,sc=values['training'],values['seal'],values['effect']
    bound=values['authenticated_seal']; v=bound['response']
    thresholds=values['thresholds']
    need(t['passed'] and t['actual_steps'][1]=='1103606.851' and t['actual_updates']==25 and t['ranks_completed']==8
         and t['training_request']==SCOPE and t['checkpoint_manifests']['25']==CP25
         and t['fresh_initial_named_SHA256']==INITIAL_NAMED and thresholds['arms']['domain25']['manifest_SHA256']==CP25['SHA256'],
         'EXACT_DOMAIN_CONTINUOUS_851_SOURCE_REQUIRED')
    need(bound['rc']==0 and bound['authority_verified'] is bound['remote_received'] is bound['payload_flushed'] is True
         and v['all288sealed_before_gold'] is v['independent_passed'] is v['passed'] is True
         and v['blind_gold_read'] is False and v['actual_steps']==s['actual_pre_gpu_post']
         and v['completed']==s['actual_prediction_slots']==sc['predictions_scored']==288
         and s['natural_EOS']==288 and s['caps']==s['errors']==s['missing']==0
         and v['core']==s['remote_inner_seal']==sc['original_inner_seal']
         and v['independent']==s['remote_outer_seal']==sc['original_outer_seal']
         and sc['original_GPU_step']==s['actual_GPU_step']=='1103606.3682'
         and sc['original_independent_read_step']==s['independent_Slurm_read_step']=='1103606.3696',
         'EXACT_3682_NEW_PROCESS_AND_INDEPENDENT_288_SEAL')
    need(b"entry['actual476_pre_forward_equal'] is True" in values['slots'] and
         b'torch.equal(actual[target].detach().cpu(),value)' in values['infer'] and
         b"final['model_loads']==1" in values['independent_reader'],'ORIGINAL_ADAPTER_ACTUAL_PRECISION_GATE_NOT_PRESENT')
    metrics={arm:dict(correct=val['correct'],denominator=val['denominator']) for arm,val in sc['arms'].items()}
    need(metrics=={'base':{'correct':40,'denominator':96},'domain25':{'correct':52,'denominator':96},
                   'question25':{'correct':23,'denominator':96}}, 'EXACT_CONF96_THREE_ARMS')
    # Recompute original group bootstrap from published aggregate wins/losses only.
    namespace={'__name__':'_pinned_conf96_statistics'}
    exec(compile(values['statistics'],'gemma_confirmatory96_statistics.py','exec'),namespace)
    comparisons={}
    for other in ('base','question25'):
        record=sc['comparisons']['domain25_vs_'+other]
        groups=record['group_differences']
        need(len(groups)==8 and all(g['wins']+g['losses']+g['ties']==g['denominator']==12 for g in groups),
             'EXACT_PUBLIC_GROUP_DENOMINATORS')
        delta=[g['wins']-g['losses'] for g in groups]
        need(sum(delta)==metrics['domain25']['correct']-metrics[other]['correct'],'SCORE_AND_GROUP_AGGREGATES_DISAGREE')
        gain=float(Fraction(sum(delta)*100,96))
        confidence=[float(x*100) for x in namespace['interval'](delta)]
        need(gain==record['gain_percentage_points'] and confidence==record['paired_group_percentile_95_percentage_points'],
             'PUBLISHED_PAIRS_OR_INTERVAL_DIFFERS')
        comparisons[other]=dict(gain_pp=gain,CI95_pp=confidence)
    base=values['base16']
    need(base['files_count']==len(base['files'])==16 and base['actual_full_target_weight_SHA_verified'] is True,
         'ORIGINAL_FULL_GEMMA_BASE16_IDENTITY_REQUIRED')
    return dict(schema='GEMMA_USGS_DOMAIN_CP25_RELEASE_EVIDENCE_V1',source_training_step='1103606.851',
         original_checkpoint_manifest=CP25,adapter_config_SHA256=ADAPTER_CFG,adapter_release_members=proven,
         adapter_member_evidence=dict(original_manifest_digests=True,current_inode_size_mtime_ctime_match=True,
                                      current_payload_SHA_recomputed=False,release_copy_verified=False),
         model=dict(alias='Gemma-3-4B-IT',base_manifest_SHA256=FILES['base16'][2],pinned_base_files=16),
         native_reload=dict(step='1103606.3682',independent_step='1103606.3696',model_process_fresh=True,
             actual_named_LoRA_values_checked=476,completions=288,all_sealed_before_gold=True),
         inference_recipe=dict(model_type='gemma3_vision',max_input=512,max_pixels=262144,greedy_output_cap=256,
             truncation='raise',padding_side='right',padding_free=False,precision='BF16',attn_impl='sdpa',
             one_image_per_request=True,response_contract='FINAL: <answer>',backend='Swift TransformersEngine + AdapterRequest'),
         confirmed96=dict(arms=metrics,independent_groups=8,compared=comparisons,gate_passed=sc['domain_effect_gate']['passed']),
         resume=recover(values['boundary'],values['resume'],values['boundary_analysis']),
         missing=['ADAPTER_COPY_ACTUAL_BYTES_AND_SHA_BEFORE_RELEASE','INDEPENDENT_FORMAL_DELIVERY_QUALIFICATION'],
         formal_delivery_ready=False,raw_weights_read=False,raw_predictions_read=False,raw_answers_read=False,
         attribution_credit_added=0,source_pins={k:dict(path=p,bytes=n,SHA256=h) for k,(p,n,h) in FILES.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-metadata',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    files=None
    if args.checkpoint_metadata:
        need(args.checkpoint_metadata.name=='checkpoint-25-files.json','ONLY_THE_NAMED_ORIGINAL_METADATA_MAY_BE_READ')
        files=checkpoint_files(read(args.checkpoint_metadata,CP25))
    result=project(read_sources(),files)
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if args.output:
        with args.output.open('xb') as f:f.write(raw)
        print(json.dumps(dict(output=pin(raw),missing=result['missing'],new_readiness=False)))
    else:
        print(raw.decode(),end='')


if __name__=='__main__':main()
