"""CPU only, runs after normal Model terminal; raw arrays remain on server."""
import pathlib,json,hashlib
from kimi_native_boundary_reader_r8531 import read_boundary_records
from kimi_native_state_consumer_r8519 import summarize_native_target

def consume(r,run):
    def mapped(value):
        value=str(value)
        return r/value[len('/refactor/'):] if value.startswith('/refactor/') else pathlib.Path(value)
    report=json.loads((r/'results'/('kimi-missing-detail-r8533-'+run+'-node0.json')).read_text())
    capture=report['native_capture'];assert capture['status']=='CAPTURE_COMPLETE_OFFLINE_PENDING'
    cases=[]
    for allowners in capture['cases']:
        owner=allowners[0];saved=json.loads(mapped(owner['outputs_path']).read_text());jobs=[]
        mismatches=[dict(seq=v['seq'],firstdiff=v['firstdiff'],actual=v['actual_ids'],frozen=v['frozen_ids'],native_equals_CPU=v['emitter_observation']['native_equals_CPU'])for v in saved if v['firstdiff']is not None or not v['emitter_observation']['native_equals_CPU']]
        for record in saved:
            if record['packed_target']is None:continue
            directory=mapped(record['perstage_trace_dirs'][0]);job=record['job']
            boundaries=read_boundary_records(directory)
            native=summarize_native_target(directory,job,record['packed_target'])
            assert len(native)==8
            for value in native:
                value['before_pool_read_applicability']='UNKNOWN: actual initflag/callee read semantics not captured. Never classify before-pool hash difference as causal read difference.'
            jobs.append(dict(kind=job['kind'],seq=job['seq'],pasts=job['pasts'],
                packed_offsets=record['original_token_offsets'],target=record['packed_target'],
                boundaries=boundaries,native=native,emitter_observation=record['emitter_observation']))
        cases.append(dict(label=owner['label'],source=owner['source'],jobs=jobs,prefix_mismatch_observations=mismatches))
    assert [v['label']for v in cases]==['Oconfirm','Wmissing']
    def index(case):
        out={}
        for j in case['jobs']:
            for native in j['native']:
                for category in ('states','intermediates'):
                    for point,value in native[category].items():
                        out[(j['kind'],native['rank'],native['layer'],category,point)]={k:value[k]for k in ('shape','storage_dtype','bytes','SHA256')}
        return out
    a,b=map(index,cases);assert a.keys()==b.keys()
    changed=[dict(key=list(key),O=a[key],W=b[key],read_applicability='UNKNOWN'if key[-1].endswith('_before')else'OBSERVED_OUTPUT_OR_INTERMEDIATE')for key in sorted(a)if a[key]!=b[key]]
    # The unchanged parent Oa captured in599 provides a cross-session repeat.
    previous=json.loads((r/'incoming/Kimi599-actual-raw-CPU-r8532.json').read_text())
    previous_case=dict(jobs=previous['results']);prior=index(previous_case)
    assert a.keys()==prior.keys()
    cross_session=[dict(key=list(key),previous_Oa=prior[key],Oconfirm=a[key])for key in sorted(a)if a[key]!=prior[key]]
    return dict(scope='OBSERVATIONAL_NATIVE_FIRSTDECODE_ONLY',
        primary_acceptance='FAILED_OBSERVED historical finaltokens; no performance/modelpromotion',
        cases=cases,Oconfirm_vs_previous_Oa_differences=cross_session,
        O_vs_W_native_target_differences=changed,
        before_state_policy='UNKNOWN read applicability unless separately proven from actual kernel source/initflags; past/offsets retained',
        intermediate_bit_differences_reject_model=False,
        raw_storage='REMOTE_ONLY',new_GPU=False,new_Model=False)
