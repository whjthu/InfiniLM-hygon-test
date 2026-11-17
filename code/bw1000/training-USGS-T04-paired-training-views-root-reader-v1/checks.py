"""Offline metadata/index checks only; never parse caption/view body on PC."""
from collections import Counter
import hashlib
import json
import random


def need(ok,why):
    if not ok:raise ValueError(why)

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()

def expected_indices():
    rng=random.Random(20261118);out=[]
    while len(out)<800:
        values=list(range(45));rng.shuffle(values);out.extend(values)
    return out[:800]

def check_producer(receipt,producer,binding):
    for x in [receipt,producer]:
        need(x['request_id']==binding['request_id'] and x['publication']==binding['publication'] and x['epoch']==binding['epoch'],'Original context differs')
    need(receipt['rc']==0 and receipt['passed'] is True and receipt['exec_request_attempted'] is True and producer['passed'] is True,'Original process not known PASS')
    need(producer['deadline']==binding['epoch']+120 and producer['work_deadline']==binding['epoch']+100 and producer['final_deadline']==binding['epoch']+110,'Original epoch reset or changed')
    need(producer['CPU_affinity']==producer['CPU_affinity_actual'] and len(producer['CPU_affinity'])==2 and not set(producer['CPU_affinity'])&{0,1}
        and producer['self_device_FDs']==[] and producer['spawned_processes']==producer['owned_descendants_spawned']==0,'Actual CPU/device/process context differs')
    need(producer['GPU']==producer['new_encodes']==producer['new_encoded_rows']==producer['actual_training_updates']==producer['new_network_calls']==0
        and producer['actual_unique_source_rows']==90 and producer['actual_new_view_rows']==1600,'Views are not training steps/native calls')
    need(producer['caption_payload_exported'] is False and producer['dev_or_blind_input_gold_read'] is False and producer['CPU_reference_hashes_identity_candidates_only'] is True
        and producer['actual_GPU_rank_cursor_padding_or_sampler_qualification'] is False,'Data scope/hash interpretation differs')
    need(set(producer['arms'])=={'domain','control'},'Exact two arms required')
    for storage,arm in [('domain','domain'),('control','generic')]:
        a=producer['arms'][storage];need(a['passed'] is True and a['arm']==arm and a['storage_arm']==storage
            and a['unique_source_rows']==45 and a['physical_view_rows']==a['planned_view_rows']==800 and a['counts_per_unique_min']==17 and a['counts_per_unique_max']==18,'Arm full45/800 denominator differs')
        need(a['planned_world_size']==8 and a['planned_per_rank_batch']==1 and a['planned_accumulation']==4 and a['planned_updates']==25
            and a['actual_training_updates']==a['new_encodes']==0 and a['native_preprocessing_replayed'] is False,'Static view cannot substitute for actual training')
    need(producer['arms']['domain']['CPU_reference_effective_supervised_tokens_unique45'] is None
        and producer['arms']['domain']['CPU_reference_effective_supervised_tokens_weighted800'] is None,'Domain uninstrumented token counts must remainNone')
    need(producer['arms']['control']['CPU_reference_effective_supervised_tokens_unique45']==694,'Actual control45 CPU reference total differs')
    need(producer['arms']['domain']['original_unique_index_sequence_SHA256']==producer['arms']['control']['original_unique_index_sequence_SHA256']==sha(canonical(expected_indices())),'Same fixed800 schedule differs')
    return True

def check_primary_ledger(value,producer_arm,storage):
    need(set(value)=={'arm','view_SHA256','original_index_by_physical_row'},'Observer ledger exact dict schema differs')
    expected_arm='domain' if storage=='domain' else 'generic';indices=value['original_index_by_physical_row']
    need(value['arm']==expected_arm and value['view_SHA256']==producer_arm['view']['SHA256'] and indices==expected_indices(),'Primary ledger arm/view/index mapping differs')
    counts=Counter(indices);need(len(indices)==800 and set(counts)==set(range(45)) and set(counts.values())=={17,18},'Exposure denominator differs')
    raw=canonical(value);pin=producer_arm['occurrences'];need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'Primary ledger original byte identity differs')
    return dict(rows=800,unique_source_indices=45,minimum=17,maximum=18,SHA256=sha(raw))
