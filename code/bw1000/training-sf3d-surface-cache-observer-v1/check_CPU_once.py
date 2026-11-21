"""Only new metadata failure/atomicity fixtures and original-AST proof; no Tensor fake."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from cache_observer import SurfaceCacheObserver
from render_observed import render,D,G

OUT=G/'local-only/training-sf3d-surface-cache-observer-v1-CPU-original'


def main():
    start=time.monotonic();OUT.mkdir();checks=[]
    overlay,proof=render()
    for name,raw in overlay.items():ast.parse(raw,feature_version=(3,10))
    checks.append('six_generated_source_files_Python310_AST')
    with tempfile.TemporaryDirectory(prefix='sf3d-cache-metadata-only-') as temp:
        root=Path(temp);o=SurfaceCacheObserver(root,labels_sha256='a'*64,input_sha256='b'*64,planned_objects=2)
        with o.phase('prepare_batch',0):
            active=json.loads((o.root/'progress.json').read_bytes())
            assert active['phases'][-1]['entered'] and active['phases'][-1]['completed'] is False
        saved=json.loads((o.root/'progress.json').read_bytes());assert saved['phases'][-1]['completed']
        assert not saved['cache_fields_and_labels_validated'] and not saved['training_READY']
        checks.append('actual_atomic_metadata_before_and_after_native_call_boundary_without_any_native_call')
        error=RuntimeError('METADATA_FIXTURE_NOT_A_NATIVE_ERROR_OR_TENSOR')
        try:
            with o.phase('query_triplane',1):raise error
        except RuntimeError as actual:assert actual is error
        failure=json.loads((o.root/'progress.json').read_bytes());last=failure['phases'][-1]
        assert not last['completed'] and last['object_index']==1 and last['error_type']=='RuntimeError'
        assert last['message_SHA256']==hashlib.sha256(str(error).encode()).hexdigest()
        assert str(error) not in (o.root/'progress.json').read_text()
        checks.append('failure_keeps_actual_phase_and_primary_exception_without_raw_tensor_or_message_export')
        before=(o.root/'progress.json').read_bytes()
        try:SurfaceCacheObserver(root,labels_sha256='a'*64,input_sha256='b'*64,planned_objects=2)
        except FileExistsError:pass
        else:raise AssertionError('Existing observer namespace was overwritten')
        assert (o.root/'progress.json').read_bytes()==before
        checks.append('existing_progress_namespace_not_replayed_or_overwritten')
        for name,index in [('not_a_native_phase',0),('prepare_batch',2),('prepare_batch',True)]:
            try:
                with o.phase(name,index):pass
            except ValueError:pass
            else:raise AssertionError('Invalid native phase/index accepted')
        checks.append('unregistered_phase_or_wrong_object_index_refused')
        o.state['fixture_oversized']='x'*(65*1024)
        try:o.write()
        except ValueError:pass
        else:raise AssertionError('Metadata cap failed')
        del o.state['fixture_oversized'];assert (o.root/'progress.json').read_bytes()==before
        checks.append('metadata64KiB_bound_preserves_last_valid_snapshot')
        with patch.object(o,'write',side_effect=[None,OSError('fixture diagnostic disk failure')]):
            try:
                with o.phase('image_features',1):raise error
            except RuntimeError as actual:assert actual is error
            else:raise AssertionError('Diagnostic storage masked native failure')
        checks.append('failed_diagnostic_write_cannot_replace_primary_native_exception')
    result={'schema':'SF3D_SURFACE_CACHE_OBSERVER_NEW_LOCAL_BOUNDARIES_V1','passed':True,
        'checks':checks,'checks_passed':len(checks),'generated_overlay':{k:{'bytes':len(v),'SHA256':hashlib.sha256(v).hexdigest()} for k,v in overlay.items()},
        'original_AST_reuse':proof,'fixture_only':True,'metadata_only_no_fake_feature_tensors':True,
        'real_cache_field_or_label_validation_executed':False,'actual_native_CUDA_or_model_or_feature_or_optimizer_calls':0,
        'old_surface_CLI_or_label_tests_repeated':False,'actual_training_READY':False,'wall_seconds':time.monotonic()-start}
    p=OUT/'result.json';p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'new_metadata_checks':len(checks),'real_cache_calls':0,'receipt':{'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}}))


if __name__=='__main__':main()
