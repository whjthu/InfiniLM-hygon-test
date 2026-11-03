"""Read-only CPU851 artifact verification. The original producer remains FAILED1."""
from pathlib import Path
import argparse, base64, hashlib, json, math, re, struct
EXPECTED_NATIVE='/refactor/candidates/Deep-HC-T1-Model-r366-cpu-v1/libDeep_HC_T1_Model_r366.so'
RECEIPT_SHA='662820da875ade0aa58b6e2145ec95611f73e096dbfbcf239691aff9f2a19d9a'
LIB_SHA='17153310c14d065a877ea047394bed049eb619c24b4f755fc4655fa01aa66348'
sha=lambda b:hashlib.sha256(b).hexdigest()
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]

def load_files(archive):
    q=json.loads(Path(archive).read_bytes());files={}
    for item in q['files']:
        b=base64.b64decode(item['base64'])if 'base64'in item else item['text'].encode()
        assert len(b)==item['bytes']and sha(b)==item['SHA256'],item['path']
        files[item['path'].split('/candidates/',1)[-1]]=b
    return q,files

def verify(observation,artifacts):
    old,oldfiles=load_files(observation);raw=next(v for k,v in oldfiles.items()if k.endswith('/CPU-build.json'));assert sha(raw)==RECEIPT_SHA
    receipt=json.loads(raw);assert receipt['passed']is False and receipt['completed']is True and receipt['actual_CPU']=='910407.851'
    assert 'DSOneeded'in receipt['traceback']and receipt['commands'][-1][0]=='readelf'
    assert old['accounting']==['910407.851|Deep419-router-CPU-v6|FAILED|1:0|00:00:07|a14r2n20']
    a,files=load_files(artifacts)
    def one(name):return next(v for k,v in files.items()if k.endswith('/'+name))
    partial=json.loads(one('partial-cases.json'));assert partial['completed']and partial['error']is None and len(partial['cases'])==14
    assert sum(x['original_passed']for x in partial['cases'])==13
    classified=[x for x in partial['cases']if not x['original_passed']];assert len(classified)==1 and classified[0]['case']=='negative_underflow'
    assert classified[0]['classification']=='KNOWN_CPU_BACKEND_UNDERFLOW_DIFFERENCE'
    rows=[]
    for row in partial['cases']:
        name=row['case'];inp=one(name+'.input');out=one(name+'.output');assert len(inp)==2056 and len(out)==2120
        assert sha(inp[:2048])==row['input_SHA256']
        ids=struct.unpack('<6q',out[:48]);weights=struct.unpack('<6f',out[48:72]);scores=struct.unpack('<256f',out[72:1096]);selection=struct.unpack('<256f',out[1096:]);epsilon,factor=struct.unpack('<2f',inp[2048:])
        assert len(set(ids))==6 and all(0<=i<256 for i in ids)
        selected=[scores[i]for i in ids]
        total=f32(f32(f32(selected[0]+selected[1])+f32(selected[2]+selected[3]))+f32(selected[4]+selected[5]));denom=f32(total+epsilon)
        # Exact disclosed C++ F32 tree; independent of the producer's Torch check.
        expected=[f32(f32(v/denom)*factor)if denom else math.nan for v in selected]
        for x,y in zip(weights,expected):
            if math.isnan(y):assert math.isnan(x)
            elif math.isinf(y):assert x==y
            else:assert math.isfinite(x)and abs(x-y)<2e-6
        if not row['original_passed']:
            assert list(ids)==[50,89,155,60,220,195]
            assert sha(out[1096:])=='e3892fba455cd4aff8840cc4a98b8eb4dc0ce88643e280aade25c5e4a16d4035'
            assert row['route_validation']['margin_separated']and row['route_validation']['same_expert_set']
            assert row['own_score_normalization']['maxabs']<2e-6 and row['own_score_normalization']['NRMSE']<2e-6
        rows.append({'case':name,'original_passed':row['original_passed'],'classification':row['classification'],'input_SHA256':sha(inp),'output_SHA256':sha(out)})
    contract=json.loads(one('pure-math-8thread-contract.log'));assert contract['passed']and contract['actual_owner_threads']==8 and not contract['SDK_loaded']and contract['GPU_calls']==0
    consumer=json.loads(one('CPU-whole-consumer.log'));assert consumer['passed']and consumer['owner_records']==8 and consumer['planned_timed_query_records']==80 and consumer['SDK_device_leaves_fake']and consumer['nonzero_forward_not_counted_completed']
    pure_needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',one('pure-math-dynamic.log').decode());assert set(pure_needed)<={'libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6','ld-linux-x86-64.so.2'}
    needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',one('private-DSO-dynamic.log').decode());assert needed.count(EXPECTED_NATIVE)==1
    ldd=one('private-DSO-ldd-r-NO-initializers.log').decode();assert EXPECTED_NATIVE in ldd and'not found'not in ldd and'undefined symbol'not in ldd
    nm=one('private-DSO-nm.log').decode();names=[x.split()[-1]for x in nm.splitlines()]
    required=['error','id_size','id','create','keys','load','finish','forward','project','same_logits','proof_info','proof_read','stats','dtype','destroy','shutdown'];assert all(names.count('deep_router_'+n+'_r419')==1 for n in required)
    assert a['library']['SHA256']==LIB_SHA and a['library']['bytes']==676368
    return dict(id='Deep419-CPU851-existing-artifact-closure',read_only_verifier_passed=True,original_CPU='910407.851 FAILED1/7s',original_CPU_producer_passed=False,original_CPU_receipt_SHA256=RECEIPT_SHA,producer_failure='Last basename-only DT_NEEDED assertion; actual absolute dependency equals the exact pinned Native path.',existing_artifact_CPU_math_and_link_checks_passed=True,original_validate_pass_count=13,known_CPU_backend_classified_cases=['negative_underflow'],original_predicates_all_passed=False,cases=rows,pure_math_contract=contract,whole_consumer=consumer,pure_math_DT_NEEDED=pure_needed,private_DSO_DT_NEEDED=needed,library=a['library'],math_executable=a['math_executable'],source_math_not_changed=True,new_job_or_compile=False,SDK_default_active_indices='UNKNOWN; deferred to same future real-device component, not a separate provider-only run',GPU_qualification=False,Model_qualification=False,scope_limit='CPU shared mathematics/driver fake leaves and static link/ELF only; no private Native/SDK DSO execution. Raw synthetic inputs/outputs remain L/R.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('observation');p.add_argument('artifacts');p.add_argument('--out');args=p.parse_args()
    text=json.dumps(verify(args.observation,args.artifacts),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if args.out:Path(args.out).write_text(text,encoding='utf-8')
    else:print(text,end='')
