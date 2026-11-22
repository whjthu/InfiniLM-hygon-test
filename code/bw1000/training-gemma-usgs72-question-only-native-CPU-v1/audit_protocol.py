"""Pure new no-image audit; check_sample copied exactly from qualified native72 source."""
import copy

def plain(value):return value.tolist() if hasattr(value,'tolist') else value

def no_visual_fields(value):
    if not isinstance(value,dict):raise ValueError('native encode/collator must return dict')
    forbidden={'image_grid_thw','video_grid_thw','image_sizes','image_bound','image_embeds','vision_features','videos','audios'}
    if any('pixel' in str(key).lower() or key in forbidden for key in value):
        raise ValueError('unexpected visual/pixel field in question-only native ABI')

def audit_rows(rows,full,capped,fingerprint,suffix,end_ids,check,event):
    records=[];failures=[];attempted=0
    for ordinal,row in enumerate(rows):
        check();attempted+=1;event('row_attempt',ordinal)
        try:
            if 'images' in row or any(m['role'] not in ('user','assistant') for m in row['messages']):
                raise ValueError('question-only native input contains unexpected media or role')
            event('full_encode_attempt',ordinal);original=full.encode(copy.deepcopy(row))
            event('capped_encode_attempt',ordinal);value=capped.encode(copy.deepcopy(row))
            no_visual_fields(original);no_visual_fields(value)
            original_pin=fingerprint(original);current_pin=fingerprint(value)
            if original_pin['SHA256']!=current_pin['SHA256'] or current_pin.get('finite') is not True:
                raise ValueError('question-only full-vs512 native input differs or nonfinite')
            ids,labels=plain(value['input_ids']),plain(value['labels'])
            supervised=check_sample(ids,labels,suffix,end_ids)
            if supervised<=0:raise ValueError('empty effective supervised target')
            event('collator_attempt',ordinal);batch=capped.data_collator([value]);no_visual_fields(batch)
            if plain(batch['input_ids'])!=[ids] or plain(batch['labels'])!=[labels]:
                raise ValueError('native collator changed actual input/label mask')
            if 'attention_mask' in batch and plain(batch['attention_mask'])!=[[1]*len(ids)]:
                raise ValueError('collator unexpectedly masked non-padding text')
            batch_pin=fingerprint(batch)
            if batch_pin.get('finite') is not True:raise ValueError('nonfinite collated field')
            record=dict(ordinal=ordinal,input_tokens=len(ids),shifted_supervised_tokens=supervised,
                complete_native_suffix_supervised=True,pixel_fields_absent=True,encoded_fields=sorted(value),
                collated_fields=sorted(batch),encoded=current_pin,collated=batch_pin)
            records.append(record);event('row_pass',record)
        except (TimeoutError,MemoryError):raise
        except Exception as error:
            failure=dict(ordinal=ordinal,type=type(error).__name__,private_reason=str(error)[:256])
            failures.append(failure);event('row_failure',failure)
    return dict(planned=72,attempted=attempted,completed=len(records),failed=len(failures),missing=72-attempted,
        passed=attempted==len(records)==72 and not failures,records=records,failures=failures)

def check_sample(ids, labels, suffix, end_ids):
    if not suffix or not set(suffix).intersection(end_ids) or not 1 < len(ids) == len(labels) <= 512:
        raise ValueError("invalid sequence length or native end-of-turn metadata")
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("invalid token identity")
    if labels[0] != -100 or any(type(label) is not int or label not in (-100, token)
                                 for token, label in zip(ids, labels)):
        raise ValueError("invalid causal label mask")
    if ids[-len(suffix):] != suffix or labels[-len(suffix):] != suffix:
        raise ValueError("complete native end-of-turn suffix is not supervised")
    return sum(label != -100 for label in labels[1:])
