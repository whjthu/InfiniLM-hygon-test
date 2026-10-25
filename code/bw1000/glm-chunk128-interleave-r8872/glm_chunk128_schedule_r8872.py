"""GLM-specific prompt packing; no tensor, device, or Engine access."""
def prompt_calls(batch, chunk):
    assert batch == 16 and chunk in (128,512)
    return [(list(range(a,a+8)),p,min(chunk,512-p))
            for a in (0,8) for p in range(0,512,chunk)]

def pack_prompt(prompts, rows, past, count, stage):
    assert len(rows)==8 and all(len(prompts[r])==512 for r in rows)
    assert past in (0,128,256,384) and count in (128,512)
    assert past+count<=512
    desc=[];flat=[];positions=[]
    for row in rows:
        desc.extend((row,past,count))
        flat.extend(prompts[row][past:past+count] if stage==0 else [0]*count)
        positions.extend(range(past,past+count))
    return desc,flat,positions

def pack_decode(prompts,generated,rows,output_index,stage):
    assert 1<=output_index<128
    desc=[];flat=[];positions=[]
    for row in rows:
        assert len(generated[row])==output_index
        past=len(prompts[row])+output_index-1
        desc.extend((row,past,1));flat.append(generated[row][-1] if stage==0 else 0)
        positions.append(past)
    return desc,flat,positions

def validate_events(events, generated, chunk):
    """Audit actual saved CAPI descriptors and generation boundaries."""
    assert len(generated)==16 and chunk in(128,512)
    lengths=[0]*16;outputs=[[]for _ in range(16)];pre=[];last_prefill=-1
    for index,e in enumerate(events):
        requests=e['requests'];assert [q[0]for q in requests]==e['rows']
        assert sum(q[2]for q in requests)==e['flat_tokens']
        for row,past,count in requests:
            assert past==lengths[row] and count>0 and past+count<=640
            if e['round']==0:
                assert not outputs[row] and past<512 and past+count<=512
                assert count==chunk and e['prompt_offset']==past and e['prompt_count']==count
                if past+count==512:outputs[row].append(0)
            else:
                assert count==1 and past>=512 and e['round']==len(outputs[row])
                assert past==512+e['round']-1
                outputs[row].append(e['round'])
            lengths[row]=past+count
        if e['round']==0:
            pre.append((e['rows'],requests[0][1],requests[0][2]));last_prefill=index
    assert pre==prompt_calls(16,chunk)
    assert outputs==[list(range(len(g)))for g in generated]
    assert lengths==[512+len(g)-1 for g in generated]
    early=[e for e in events[:last_prefill]if e['round']>0]
    assert len(early)==(3 if chunk==128 else 0)
    return {'prefill_calls':len(pre),'decode_calls':sum(e['round']>0 for e in events),
            'early_decode_calls_before_last_prefill':len(early),
            'early_decode_outputs_before_last_prefill':sum(len(e['rows'])for e in early),
            'final_lengths':lengths,'all_position_continuations_valid':True}
