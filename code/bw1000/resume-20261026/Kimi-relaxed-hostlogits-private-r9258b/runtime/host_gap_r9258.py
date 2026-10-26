"""Gather only paired scalar BF16 values from the already-paid HOST logits.
No full-vocabulary float conversion, no new native work, no GPU API.
"""
import time,math,hashlib,json
import numpy as np

def scalar_gaps(logits, selected, token_rows):
    t=time.perf_counter()
    if logits.dtype!=np.uint16 or logits.ndim!=2:raise ValueError('HOST BF16 matrix required')
    widths=[len(x)for x in token_rows];T=sum(widths)
    if any(not 1<=w<=8 for w in widths)or len(selected)!=T or logits.shape[0]!=T:raise ValueError('causal packed rows')
    rows=[];candidates=[];top=[];offset=0
    for values in token_rows:
        for j,p in enumerate(values[1:]):
            row=offset+j;s=selected[row]
            if not 0<=int(p)<logits.shape[1]or not 0<=int(s)<logits.shape[1]:raise ValueError('vocabulary ID')
            rows.append(row);candidates.append(int(p));top.append(int(s))
        offset+=len(values)
    # Advanced-index copies are length K_total, not a V-sized conversion.
    pair=np.stack([logits[rows,top],logits[rows,candidates]],axis=1)if rows else np.empty((0,2),np.uint16)
    values=(pair.astype(np.uint32)<<16).view(np.float32)
    gap=values[:,0]-values[:,1];result=[None]*T;invalid=negative=0
    for row,v in zip(rows,gap):
        if not math.isfinite(float(v)):invalid+=1
        elif v<0:negative+=1
        else:result[row]=float(v)
    return {'gaps':result,'candidate_rows':len(rows),'gathered_BF16_scalars':2*len(rows),
            'compact_F32_conversion_scalars':2*len(rows),'nonfinite_pairs':invalid,'negative_gaps':negative,
            'gather_wall_s':time.perf_counter()-t,'scope':'HOST scalar gathers from existing flags3 output; no new GPU/fullvocab RPC'}

def special_provenance(tok,eos,vocab_size=None):
    from_added=sorted(int(k)for k,v in tok.added_tokens_decoder.items()if bool(getattr(v,'special',False)))
    listed=sorted(set(int(v)for v in tok.all_special_ids));eos=sorted(set(map(int,eos)))
    native_special=getattr(tok,'special_tokens',{})
    native_ids=sorted(set(map(int,native_special.values())))if isinstance(native_special,dict)else[]
    protected=sorted(set(listed)|set(from_added)|set(eos)|set(native_ids))
    bound=len(tok)if vocab_size is None else int(vocab_size)
    if bound<1 or any(v<0 for v in protected):raise ValueError('invalid special metadata')
    outside=[{'id':v,'token':str(tok.convert_ids_to_tokens(v))}for v in protected if v>=bound]
    out={'protected_ids':protected,'model_vocab_size':bound,'native_tiktoken_special_ids':native_ids,'outside_MODEL_declared_specials':outside,'in_MODEL_protected_ids':[v for v in protected if v<bound],'tokenizer_length':len(tok),'protected_outside_tokenizer_length':[v for v in protected if v>=len(tok)],'tokenizer_all_special_ids':listed,'added_tokens_decoder_special_ids':from_added,
         'frozen_EOS_ids':eos,'special_tokens_map':{str(k):str(v)for k,v in tok.special_tokens_map.items()},
         'origin':'loaded checkpoint AutoTokenizer; all_special_ids UNION added_tokens_decoder.special UNION frozenEOS UNION native tiktoken.special_tokens; out-of-model declarations kept protected but unreachable'}
    out['provenance_SHA256']=hashlib.sha256(json.dumps(out,sort_keys=True).encode()).hexdigest()
    return out
