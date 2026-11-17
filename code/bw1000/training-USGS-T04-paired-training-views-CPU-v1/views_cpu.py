"""Read qualified45 and native reference metadata, emit800 without encode or training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from exposures import exposure_indices


def need(ok,why):
    if not ok:raise ValueError(why)

def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def pinned(pin):
    p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and 0<pin['bytes']<=256*1024,'Bounded exact prior artifact required')
    a=p.stat();raw=p.read_bytes();b=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
        and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'Prior artifact changed/unstable');return raw

def references(rows,native,binding):
    train=[r for r in native if r['split']=='train'];need(len(rows)==len(train)==45 and [r['ordinal'] for r in train]==list(range(45)),'Full original45 source/reference order required')
    need(len({r['id'] for r in rows})==45,'Repeated unique source ID')
    refs=[]
    for i,(row,r) in enumerate(zip(rows,train)):
        need(set(row)=={'id','messages','images'} and len(row['messages'])==2 and row['messages'][0]['role']=='user' and row['messages'][1]['role']=='assistant'
            and isinstance(row['messages'][1]['content'],str) and row['messages'][1]['content'].startswith('FINAL: ') and len(row['images'])==1,'Unchanged canonical45 FINAL schema differs')
        sid=sha(row['id'].encode());need(sid==r['source_id_SHA256'] and r['full_input_retained'] is True and r['labels_None'] is False and r['supervised_EOS_count']>0,'Canonical/native identity or supervision mismatch')
        p=Path(row['images'][0]);root=Path(binding['target_data']);need(p.is_relative_to(root),'Image path escaped prior qualified root');rel=p.relative_to(root).as_posix()
        need(rel.startswith('train/media/') and rel in binding['media'],'Dev/blind/undeclared image in training')
        for name in ['input_ids','labels','attention_mask','pixel_values']:
            field=r['collated_fields'][name];need(field['finite'] is True and field['elements']>0 and len(field['SHA256'])==64,'Native reference field absent')
        need(r['collated_fields']['input_ids']['elements']==r['collated_fields']['labels']['elements']==r['full_length'],'Actual CPU full input/labels length differs')
        refs.append(dict(original_unique_index=i,source_id_SHA256=sid,source_image_SHA256=binding['media'][rel]['SHA256'],source_image_path=row['images'][0],
            native_record_SHA256=sha(canonical(r)),CPU_reference_full_length=r['full_length'],CPU_reference_first_supervised_index=r['first_supervised_index'],
            effective_supervised_tokens_actual_CPU_reference=r.get('effective_supervised_tokens'),native_supervised_EOS_count=r['supervised_EOS_count'],native_EOS_identity_SHA256=sha(canonical(r['native_EOS_ids'])),
            encoded_input_ids_SHA256=r['encoded_fields']['input_ids']['SHA256'],encoded_labels_SHA256=r['encoded_fields']['labels']['SHA256'],
            collated_input_ids_SHA256=r['collated_fields']['input_ids']['SHA256'],collated_labels_SHA256=r['collated_fields']['labels']['SHA256'],
            collated_attention_mask_SHA256=r['collated_fields']['attention_mask']['SHA256'],collated_vision_fields_SHA256=sha(canonical({k:v for k,v in r['collated_fields'].items() if k not in {'input_ids','labels','attention_mask'}})),
            CPU_reference_batch_size=1,identity_candidate_only=True,actual_GPU_rank_cursor_or_padding_match_qualified=False))
    return refs

def build_arm(arm,config,source,output,check,progress):
    check();qraw=(source/config['qualification']['file']).read_bytes();need(len(qraw)==config['qualification']['bytes'] and sha(qraw)==config['qualification']['SHA256'],'Prior independent qualification differs');q=json.loads(qraw)
    if arm=='domain':need(q['native_preprocessing_qualified'] is True and q['completed_train']==45 and q['completed_dev']==10 and q['actual_updates']==0,'Domain45 prior native not qualified')
    else:need(q['native_preprocessing_qualified'] is True and q['new_native_completed']==45 and q['new_dev_completed']==0 and q['actual_updates']==0,'Control45 prior native not qualified')
    raw=pinned(config['canonical45']);braw=pinned(config['data_binding']);nraw=pinned(config['native_identities']);binding=json.loads(braw);native=json.loads(nraw)
    need(binding['passed'] is True and binding['files']['train-swift.jsonl']['SHA256']==sha(raw) and binding['files']['train-swift.jsonl']['bytes']==len(raw)
        and binding['files']['train-swift.jsonl']['rows']==45 and len(native)==config['native_file_total_records'],'Native stage/file denominator differs')
    if arm=='domain':need(q['adapter_files']['train']['SHA256']==sha(raw) and q['data_binding_SHA256']==sha(braw),'Domain canonical45 public identity differs')
    else:need(q['canonical45_input']['SHA256']==sha(raw) and q['data_binding_SHA256']==sha(braw) and q['native_identities_SHA256']==sha(nraw),'Control canonical45 public identity differs')
    need(binding['dev_gold_read'] is False and binding['blind_inputs_gold_or_media_read'] is False,'Prior source scope violated split boundary')
    lines=raw.splitlines(keepends=True);need(len(lines)==45 and all(line.endswith(b'\n') for line in lines),'Exact45 complete JSONL lines required')
    rows=[json.loads(line) for line in lines];refs=references(rows,native,binding);indices=exposure_indices();need(len(indices)==800 and set(indices)==set(range(45)),'Frozen exposure function changed')
    output.mkdir(mode=0o700);refraw=canonical(refs);(output/'source45-identities-private.json').write_bytes(refraw)
    view=output/'train-800.jsonl';occ=output/'occurrence800-private.jsonl';viewsha=hashlib.sha256();occsha=hashlib.sha256();viewbytes=0;occbytes=0;counts=Counter()
    with view.open('xb') as vf,occ.open('xb') as of:
        for physical,i in enumerate(indices):
            check();line=lines[i];vf.write(line);viewsha.update(line);viewbytes+=len(line);counts[i]+=1
            ref=refs[i];entry=dict(physical_row=physical,original_unique_index=i,source_id_SHA256=ref['source_id_SHA256'],source45_line_SHA256=sha(line),
                CPU_native_record_SHA256=ref['native_record_SHA256'],encoded_input_ids_SHA256=ref['encoded_input_ids_SHA256'],encoded_labels_SHA256=ref['encoded_labels_SHA256'],
                collated_input_ids_SHA256=ref['collated_input_ids_SHA256'],collated_labels_SHA256=ref['collated_labels_SHA256'],collated_attention_mask_SHA256=ref['collated_attention_mask_SHA256'],
                collated_vision_fields_SHA256=ref['collated_vision_fields_SHA256'],identity_candidate_only=True,actual_GPU_padding_or_rank_mapping_qualified=False)
            encoded=canonical(entry);of.write(encoded);occsha.update(encoded);occbytes+=len(encoded)
            if physical%100==99:progress(arm,dict(source45_verified=45,planned_view_rows=800,completed_view_rows=physical+1))
    need(sum(counts.values())==800 and len(counts)==45 and set(counts.values())=={17,18},'Exposure denominator differs')
    known=all(r['effective_supervised_tokens_actual_CPU_reference'] is not None for r in refs)
    primary=canonical(dict(arm='domain' if arm=='domain' else 'generic',view_SHA256=viewsha.hexdigest(),original_index_by_physical_row=indices))
    (output/'exposure-ledger-private.json').write_bytes(primary)
    result=dict(passed=True,arm='domain' if arm=='domain' else 'generic',storage_arm=arm,unique_source_rows=45,physical_view_rows=800,planned_view_rows=800,planned_updates=25,planned_world_size=8,planned_per_rank_batch=1,planned_accumulation=4,
        counts_per_unique_min=min(counts.values()),counts_per_unique_max=max(counts.values()),source45_SHA256=sha(raw),source_binding_SHA256=sha(braw),native_reference_SHA256=sha(nraw),
        source_reference45_SHA256=sha(refraw),view=dict(name='train-800.jsonl',bytes=viewbytes,SHA256=viewsha.hexdigest(),rows=800),
        occurrences=dict(name='exposure-ledger-private.json',bytes=len(primary),SHA256=sha(primary),rows=800),
        occurrence_ledger=dict(name='occurrence800-private.jsonl',bytes=occbytes,SHA256=occsha.hexdigest(),rows=800),
        original_unique_index_sequence_SHA256=sha(canonical(indices)),source_record_ids_ordered_SHA256=sha(canonical([r['source_id_SHA256'] for r in refs])),
        CPU_reference_input_tokens_unique45=sum(r['CPU_reference_full_length'] for r in refs),CPU_reference_input_tokens_weighted800=sum(counts[i]*r['CPU_reference_full_length'] for i,r in enumerate(refs)),
        CPU_reference_effective_supervised_tokens_unique45=sum(r['effective_supervised_tokens_actual_CPU_reference'] for r in refs) if known else None,
        CPU_reference_effective_supervised_tokens_weighted800=sum(counts[i]*r['effective_supervised_tokens_actual_CPU_reference'] for i,r in enumerate(refs)) if known else None,
        domain_effective_supervision_not_instrumented_in_original_native55=not known,new_encodes=0,caption_payload_exported=False,dev_or_blind_input_or_gold_read=False,
        images_or_weights_read=False,CPU_batch1_reference_hashes_only=True,actual_GPU_input_hash_match_or_rank_cursor_qualification=False,
        actual_training_updates=0,actual_training_token_exposures='NOT_RUN',native_preprocessing_replayed=False,training_GO=False,GPU_GO=False)
    (output/'view-binding-private.json').write_bytes(canonical(result));return result
