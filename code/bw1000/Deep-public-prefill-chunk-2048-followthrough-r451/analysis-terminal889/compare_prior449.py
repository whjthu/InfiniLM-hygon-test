"""Compare two real first-OFF requests. This does not locate a numeric root cause."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
H=lambda raw:hashlib.sha256(raw).hexdigest()
PINS=['1fc2580cf4d3aa743ebc06f686aade6d2f8fb442edefd22d27c2961457f1a503',
      '8ddb9aa83b3f3e0ef4ac051e36e32f82fb183624cbc40fff8af74244efd4f3db']
def load(path,pin):
    raw=Path(path).read_bytes();assert H(raw)==pin
    q=json.loads(gzip.decompress(raw));values=q['files']
    items=values.items() if isinstance(values,dict) else ((v['path'],v)for v in values)
    out={}
    for name,v in items:
        b=base64.b64decode(v['base64'],validate=True)
        assert len(b)==v['bytes'] and H(b)==v['SHA256'];out[name]=b
    return out
def one(files,suffix):
    found=[(name,raw)for name,raw in files.items()if name.endswith(suffix)]
    assert len(found)==1,suffix
    return found[0]
def compare(left,right):
    sources=Path(__file__).resolve().parents[2]
    l=load(left,PINS[0]);r=load(right,PINS[1])
    lo=json.loads(one(l,'/owner.json')[1]);ro=json.loads(one(r,'/owner.json')[1])
    a=lo['requests'][0];b=ro['requests'][0]
    lb=one(l,'/request-0.request.json')[1];rb=one(r,'/request-0.request.json')[1]
    assert lb==rb
    assert a['r449_mode']==b['r451_mode']==0
    assert a['r449_r366_fixed']==b['r451_r366_fixed']==b['r451_fullpre_fixed']==0
    assert lo['Native_SHA256']==ro['Native_SHA256']
    assert (lo['TP'],lo['PP'],lo['HC_mode'])==(ro['TP'],ro['PP'],ro['HC_mode'])==(8,1,0)
    assert a['prompt_tokens']==b['prompt_tokens']==8489
    assert a['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']==b['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']
    assert a['Native_continuation']==b['Native_continuation']
    assert a['actual_forward_completed']==b['actual_forward_completed']==106
    assert a['normal_reset_before'] and b['normal_reset_before'] and a['normal_reset_after'] and b['normal_reset_after']
    trace=a['Native_continuation'];pre=[x for x in trace if x['kind']=='prefill']
    assert [x['tokens']for x in pre]==[1024]*8+[297]
    pairs=[dict(index=i,prior_ID=x,current_ID=y)for i,(x,y)in enumerate(zip(a['generated_token_ids'],b['generated_token_ids']))if x!=y]
    assert len(a['generated_token_ids'])==len(b['generated_token_ids'])==98 and pairs
    old=sources/'Deep-HC-fullpre-public-pair-r449/source-v1/public_deep_model'
    new=sources/'Deep-public-prefill-chunk-2048-followthrough-r451/source-v1/public_deep_model'
    same=[];changed=[]
    for p in sorted(new.glob('*.py')):
        q=old/p.name
        if q.exists():
            item=dict(name=p.name,prior_SHA256=H(q.read_bytes()),current_SHA256=H(p.read_bytes()))
            (same if item['prior_SHA256']==item['current_SHA256'] else changed).append(item)
    assert {v['name']for v in changed}=={'worker.py','request_policy.py','native_binding.py'}
    priorclient=json.loads(one(l,'-client/result.json')[1])
    assert priorclient['rows'][0]['semantic_gold'] is True
    return dict(scope='Actual449 firstOFF correct versus actual451 firstOFF wrong; no replay, teacher trace or causal kernel proof',
        original_archives_SHA256=PINS,prior='910374.1392/request0',current='910376.889/request0',
        equal_request_body_SHA256=H(lb),equal_Native_SHA256=lo['Native_SHA256'],same_model_parallelism=[8,1,0],
        same_full_prompt8489=True,same_firstOFF_HC0_r3660_fullpre0=True,same_prefill_chunk1024_tail297=True,
        same_logical_Native_continuation=True,same106_completed_forwards=True,same_normal_request_resets=True,
        prior_model_load_s=lo['loading_seconds'],current_model_load_s=ro['loading_seconds'],
        prior_gold=True,current_gold=False,prior_and_current_natural_EOS=True,generated_lengths=[98,98],
        differing_generated_IDs=pairs,identical_runtime_source_members=same,changed_runtime_members=changed,
        new_chunk_contract_SHA256=H((new/'chunk_contract.py').read_bytes()),
        current_first_chunk_is1024=True,actual2048executed=False,
        not_controlled_same_residency=True,hardware_nodes_differ=True,
        limitations=['Different model loads and nodes; source equality and logical calls do not establish identical intermediate activation/cache state.',
                    'No teacher-forced shared-prefix logits or current/previous KV comparison was recorded.',
                    'Do not blame2048, fullpre candidate, one kernel, nondeterminism or prefill mathematics from this cross-run contrast.',
                    'The recorded healthy old449 scoped gain and new451 baseline semanticfailure are both retained.'],
        next='Consume Deep450 actual fullpaired quality first. If another Model load is needed, use one bounded same-residency first-OFF shared-prefix diagnostic with finite/logit margin and cache-state evidence, rather than repeating six-request performance load.',
        new_requests=0,new_GPU=0,new_performance_round=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('current');a=p.parse_args()
    print(json.dumps(compare(a.prior,a.current),ensure_ascii=False,indent=2))
