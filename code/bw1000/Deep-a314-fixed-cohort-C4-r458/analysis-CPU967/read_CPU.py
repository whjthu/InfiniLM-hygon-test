"""Offline original CPU967 receipt/log/accounting audit; no source imports."""
import hashlib,json,sys
from pathlib import Path

def audit(path):
    raw=Path(path).read_bytes();outer=json.loads(raw)
    b=outer['receipt_rawtext'].encode();h=hashlib.sha256(b).hexdigest()
    assert len(b)==outer['receipt_bytes'] and h==outer['receipt_SHA256']=='937199ee08d4a94f13286b3adf851d4e70bc1ff5dbd5775706c9dcbd41f3d047'
    q=json.loads(b)
    assert q['passed'] and q['completed'] and q['actual_CPU']=='910407.967'
    assert q['source_manifest_SHA256']=='4c782ade4e6f28c71e210de3344bf1630533dfdb39481b12a144464dd9c4d474'
    assert not any(q[k]for k in ['Native_dlopen','SDK_execution','Runtime_execution','weights','GPU','Model_READY','new_compiles','repeated_CPU945_encoding','old_suites_replayed'])
    assert outer['sacct']['rc']==0 and '910407.967|Deep458-fixed-C4-CPU-v1|COMPLETED|0:0|3|a14r2n20'in outer['sacct']['stdout']
    assert outer['squeue']['rc']==0 and not outer['squeue']['stdout'].strip()
    for p,log in outer['logs'].items():
        data=log['text'].encode();assert len(data)==log['bytes']and hashlib.sha256(data).hexdigest()==log['SHA256'],p
    a=q['actual_packet_handoff'];assert a['input_ordinals']==[0,1,128,129]and len(a['cases'])==4
    assert [x['prompt_tokens']for x in a['cases']]==[1242,1305,538,1050]
    assert all(x['output_budget']==4096 for x in a['cases'])
    assert len({x['case_id']for x in a['cases']})==4
    native_derivation=q['actual_a314_metadata_derivation'];assert native_derivation['objects']==50 and native_derivation['descriptor_rows']==4 and native_derivation['capacity_per_row']==32768 and not native_derivation['Native_dlopen']
    assert native_derivation['library_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
    assert q['actual_four_CPU945_inputs'] and q['C1_descriptor_rejected_before_binding']and q['foreign_encoding_rejected_before_records']and not q['project_or_SDK_maps']
    c=q['changed_consumer'];assert c['passed'] and c['actual_new_caller'] and c['actual_original_forward_wrapper'] and c['actual_context_manager']
    assert {v['case']for v in c['cases']}=={'cohort','serial','canceled','failed_forward_drains'}
    for v in c['cases']:
        assert v['passed'] and v['actual_libffi']and v['Native_leaves_recorded']and not v['actual_Native']
        assert v['high_context_pointer_preserved']and v['wrong_owner_thread_rejected']and v['packet_file_roundtrip']
        assert v['reset_all_calls']==2 and v['normal_destroy_callback_calls']==1
    active=next(v['decode_rows']for v in c['cases']if v['case']=='cohort')
    assert active==[[0,1,2,3],[0,1,2,3],[1,2,3],[2,3],[3]]
    return dict(passed=True,scope='CPU967 engineering only; Native/SDK/memory/weights recording leaves',
        actual_CPU=q['actual_CPU'],accounting='COMPLETED0/3s',receipt_SHA256=h,
        original_observation_SHA256=hashlib.sha256(raw).hexdigest(),source_manifest_SHA256=q['source_manifest_SHA256'],
        actual_four_CPU945_inputs=True,original_prompt_tokens=[1242,1305,538,1050],budgets=[4096]*4,
        packet=a,Native_derivation=native_derivation,changed_cases=c['cases'],GPU_GO=False,Model_READY=False,
        full32K_rows4_memory_floor='UNKNOWN',actual_Native=False)

if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2)+'\n',end='')
