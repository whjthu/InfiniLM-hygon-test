"""Consume the real independent511 metadata; test byte-copy only with explicit fixtures."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from handoff import (REPORT,WORKER,CAPTURE,validated_report,plan_from_report,validate_worker,
                     exclusive_copy,stable_read,pin)

D=Path(__file__).resolve().parent;G=D.parents[2]
REPORT_PATH=G/'local-only/training80-user-all-nodes-live-first/gemma-render72-independent-20261122-v1.json'
OUT=G/'local-only/training-gemma-native511-handoff-v1-CPU-original'


def main():
    start=time.monotonic();OUT.mkdir();checks=[]
    actual_raw=stable_read(REPORT_PATH,REPORT);actual=validated_report(actual_raw);plan=plan_from_report(actual_raw)
    assert plan['source']['SHA256']==plan['destination']['SHA256']==WORKER['SHA256']
    assert plan['source']['bytes']==plan['destination']['bytes']==344
    assert not plan['byte_handoff_executed'] and not plan['actual_rank_loader_observed'] and not plan['training_READY']
    assert plan['actual_800_exposure_supervised_tokens'] is None
    checks.append('actual1218B_original511_report_consumed_and_exact344B_target_copy_plan_bound')
    (OUT/'actual-original511-handoff-plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n',encoding='utf8')
    try:validated_report(actual_raw+b' ')
    except ValueError:checks.append('altered_independent_report_rejected')
    else:raise AssertionError('Different report bytes accepted')
    fixture={'passed':True,'unique_rows':72,'encode_calls':144,'collator_calls':72,'private_capture_SHA256':CAPTURE,
       'model_loaded':False,'optimizer_updates':0,'actual_rank_loader_observed':False,'training_ready':False,'epoch':1000.,'deadline':1390.}
    raw=(json.dumps(fixture,sort_keys=False)+'\n').encode();report=copy.deepcopy(actual);report['worker_receipt']=pin(raw)
    assert pin(raw)!=WORKER,'Fixture must never impersonate the actual unavailable producer bytes'
    validate_worker(raw,report);checks.append('explicit_synthetic_producer_scope_schema_accepted_not_actual_bytes')
    for field,value in [('model_loaded',True),('optimizer_updates',1),('actual_rank_loader_observed',True),
                        ('unique_rows',True),('collator_calls',71),('deadline',1391.)]:
        bad=dict(fixture);bad[field]=value;b=json.dumps(bad).encode();br=copy.deepcopy(report);br['worker_receipt']=pin(b)
        try:validate_worker(b,br)
        except ValueError:checks.append('reject_synthetic_'+field)
        else:raise AssertionError('Invalid producer scope accepted')
    with tempfile.TemporaryDirectory(prefix='gemma-handoff-fixture-') as temp:
        root=Path(temp)/'normal';root.mkdir();receipt=exclusive_copy(raw,root,report)
        assert (root/'native-input-cpu-result.json').read_bytes()==raw
        assert receipt['source_producer_bytes_reserialized'] is False and receipt['training_READY'] is False
        checks.append('actual_local_filesystem_copy_keeps_synthetic_bytes_unsorted_unmodified')
        before=(root/'native-input-cpu-result.json').read_bytes()
        try:exclusive_copy(raw,root,report)
        except ValueError:checks.append('existing_metadata_not_replayed_or_overwritten')
        else:raise AssertionError('Existing target copied twice')
        assert (root/'native-input-cpu-result.json').read_bytes()==before
        root=Path(temp)/'intent';root.mkdir();(root/'intent.json').write_text('fixture UNKNOWN original intent')
        try:exclusive_copy(raw,root,report)
        except ValueError:checks.append('preexisting_original_GPU_intent_blocks_metadata_restage')
        else:raise AssertionError('Original request marker ignored')
        assert not (root/'native-input-cpu-result.json').exists()
    for path in D.glob('*.py'):ast.parse(path.read_bytes(),feature_version=(3,10))
    result={'schema':'GEMMA_ORIGINAL511_METADATA_BOUND_HANDOFF_LOCAL_V1','passed':True,'actual_independent_report':REPORT,
       'actual_original_CPU_step':'1103606.511','actual_verified_source_rows':72,'actual_consumer_expected_worker':WORKER,
       'actual_target_producer_bytes_observed':False,'actual_target_handoff_executed':False,'actual_new_native_encodes_or_GPU_or_optimizer_updates':0,
       'local_file_copy_tests_are_synthetic_only':True,'synthetic_producer_differs_from_original_pin':True,
       'tests_passed':len(checks),'checks':checks,'plan_file':pin((OUT/'actual-original511-handoff-plan.json').read_bytes()),
       'original_initial_trainable_approval_or_payload_untouched':True,'actual_rank_sampler_qualification':False,
       'training_READY':False,'wall_seconds':time.monotonic()-start}
    p=OUT/'result.json';p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'checks':len(checks),'actual_source_report_bytes':1218,'actual_target_producer_bytes':0,
                     'receipt':{'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}}))


if __name__=='__main__':main()
