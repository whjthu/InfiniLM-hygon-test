"""Finite immutable baseline job trace/replay contract. No scheduling rewrite.

Full-logit reference provider alone writes baseline IDs. Candidate never creates
or updates them. Replay is untimed numerical evidence, NOT live service TPS.
Reset all owners with existing barriers before each provider phase externally.
"""
import hashlib,json,pathlib,os


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_records(header,records):
    assert header['scope']=='MATCHED_SHAPE_JOBTRACE_ONLY' and 1<=header['rows']<=32
    assert 0<len(records)<=4096 and header['reference_flags']==67 and header['candidate_flags']==71
    pasts={};last={};row_keys={};done=set();seen=set();outputs={}
    eos=set(header['eos_token_ids'])
    for seq,record in enumerate(records):
        job=record['job'];assert job['seq']==seq and record['job_SHA256']==digest(job)
        assert len(job['rows'])==len(job['pasts'])==len(job['keys'])==len(job['inputs'])==len(record['reference_ids'])>0
        assert len(set(job['rows']))==len(job['rows']) and 0<sum(map(len,job['inputs']))<=8192
        assert record['oracle']['finite'] and record['oracle']['native_equals_CPU']
        assert record['reference_ids']==record['oracle']['cpu_argmax_ids']==record['oracle']['native_ids']
        for key,row,past,ins,token in zip(job['keys'],job['rows'],job['pasts'],job['inputs'],record['reference_ids']):
            assert type(key)is int and type(row)is int and 1<=row<=header['rows']
            assert key not in done and len(ins)>0 and past+len(ins)<=1024
            assert all(type(v)is int and 0<=v<163840 for v in ins)
            if row not in row_keys:
                assert key not in seen and past==0 and job['kind']=='prefill'
                row_keys[row]=key;seen.add(key);outputs[key]=[]
            else:
                assert row_keys[row]==key and past==pasts[row] and job['kind']=='decode'
                assert ins==[last[row]],'baseline trace changed autoregressive prefix'
            assert type(token)is int and 0<=token<163840
            outputs[key].append(token);pasts[row]=past+len(ins);last[row]=token
            if token in eos or len(outputs[key])==header['output_limits'][str(key)]:done.add(key)
    assert seen==set(map(int,header['output_limits'])) and done==seen,'incomplete/censored baseline cannot seal'
    return outputs


def seal_reference(path,header,records):
    outputs=validate_records(header,records)
    payload=dict(header=header,records=records,complete_outputs=outputs)
    p=pathlib.Path(path)
    with p.open('x',encoding='utf8')as f:
        json.dump(payload,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_reference(path,expected_SHA256,identity):
    p=pathlib.Path(path);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_SHA256
    d=json.loads(raw);assert d['header']['identity']==identity,'DSO/math/cache/inputs identity differs'
    assert validate_records(d['header'],d['records'])=={int(k):v for k,v in d['complete_outputs'].items()}
    return d


def replay(reference,adapter,collect,persist):
    """Call all4owners in same order using existing unanimous CPU control.

    Every returned ID and firstdiff is persisted BEFORE a hard mismatch error.
    Stops atfirstdiff; no teacherforced continuation masquerading as generation.
    """
    actual={};records=[]
    for expected in reference['records']:
        job=expected['job'];ids=adapter.forward(job)
        packet=collect('matched-replay-'+str(job['seq']),dict(ids=ids,job_SHA256=digest(job)))
        assert all(v['job_SHA256']==expected['job_SHA256'] for v in packet)
        ids=packet[3]['ids'];assert ids is not None and len(ids)==len(job['rows'])
        diff=next((i for i,(a,b)in enumerate(zip(ids,expected['reference_ids'])) if a!=b),None)
        row=dict(seq=job['seq'],rows=job['rows'],pasts=job['pasts'],actual_ids=ids,
                 expected_ids=expected['reference_ids'],first_difference_row_index=diff,
                 qualification='MATCHED_TRACE_ONLY' if diff is None else 'REJECTED_FIRST_DIFF_DIAGNOSTIC_ONLY')
        for key,token in zip(job['keys'],ids):actual.setdefault(key,[]).append(token)
        records.append(row);persist(dict(records=records,actual_outputs=actual,firstdiff=row if diff is not None else None))
        assert diff is None,'matched provider token mismatch; all actuals persisted'
    assert actual=={int(k):v for k,v in reference['complete_outputs'].items()}
    return dict(qualified=True,scope='MATCHED_SHAPE_JOBTRACE_ONLY',replay_service_TPS='NOT_A_PERFORMANCE_MEASUREMENT')
