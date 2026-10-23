"""CPU verifier for5858 monotone source-level counters; not a GPU timer."""
def validate(before,after,*,tokens,moe_layers,policy,enabled):
    assert len(before)==len(after)==8
    result=[]
    for a,b in zip(before,after):
        assert len(a)==len(b)==7 and all(type(v)is int and v>=0 for v in a+b)
        assert all(y>=x for x,y in zip(a,b))
        d=[y-x for x,y in zip(a,b)]
        calls,rows,plans,h2d,payload,unused,enabled_calls=d
        assert calls in [0,moe_layers]
        if tokens>=512:assert calls==moe_layers
        else:assert calls==0
        if policy and 8<=tokens<=64:assert calls==0
        assert rows==calls*tokens and plans>=calls
        packed=bool(enabled and tokens>128)
        assert h2d==3*(calls if packed else plans)
        assert enabled_calls==(calls if packed else 0)
        assert b[5]==(max(a[5],tokens)if calls else a[5])
        assert payload>=calls*(16*tokens*16+4)
        if not calls:assert d[:5]==[0]*5 and d[6]==0
        result.append({'calls':calls,'plans':plans,'metadata_H2D_calls':h2d,'payload_bytes':payload,'enabled_calls':enabled_calls,'absolute_max_rows':b[5]})
    return result
