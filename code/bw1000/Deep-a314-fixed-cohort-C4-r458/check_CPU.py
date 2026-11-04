"""New fixed-cohort caller only. SDK/Native/memory/weight leaves recorded."""
import copy
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch
from fixed_cohort import native_context as owner
from fixed_cohort.continuation import run
from fixed_cohort.profile import config as profile_config
from fixed_cohort.recording import Recording, RecordingTorch


def checks(prepared, output):
    output = Path(output); output.mkdir()
    results = []
    for label, mode, fail, cancel in [('cohort', 'cohort', False, False),
            ('serial', 'serial', False, False), ('canceled', 'cohort', False, True),
            ('failed_forward_drains', 'cohort', True, False)]:
        rec = Recording(prepared['cfg']['vocab_size'], fail_decode=fail)
        report = {}; saved = []
        def save(phase, value):
            # Actual file handoff of complete private scalar/token report. No
            # stdout/Git raw inputs; the file roundtrip catches serialization.
            p = output/(label+'.json')
            p.write_text(json.dumps(value, allow_nan=False)+'\n')
            q = json.loads(p.read_bytes()); assert q['profile'] == prepared['config']['profile']
            saved.append((phase, q))
        def weight_leaf(binding, ctx, **kwargs):
            assert ctx == rec.ctx
            rec.calls.append(dict(kind='recording_weight_load_not_executed'))
        affinity = {'actual_owner_report': {'owners': []}, 'recording_only': True}
        policy = dict(failures=0, owners=[dict(tid=i, caller='libinfiniops.so', created=1, verified=1) for i in range(8)])
        failed = False
        with patch.object(owner, 'providers', lambda *a: {'recording_only': True}), \
             patch.object(owner, 'verify_affinity', lambda *a: affinity), \
             patch.object(owner, 'load_checkpoint', weight_leaf), \
             patch.object(owner, 'observe_threads', lambda *a: {'status': 'RECORDING', 'owners': []}), \
             patch.object(owner, 'blas_snapshot', lambda: policy):
            try:
                with owner.FixedCohortContext(rec.binding, prepared['receipt'],
                        config=prepared['config'], cfg=prepared['cfg'], mapping=prepared['mapping'],
                        torch=RecordingTorch(), safe_open=None, report=report,
                        deadline_check=lambda *a, **kw: None) as context:
                    wrong_thread = []
                    def check_other_thread():
                        try: context.check_thread()
                        except AssertionError: wrong_thread.append(True)
                    t = threading.Thread(target=check_other_thread); t.start(); t.join(2)
                    assert wrong_thread == [True] and not t.is_alive()
                    trial = run(context, prepared['records'], mode=mode,
                        deadline=time.time()+60, save=save,
                        canceled=lambda: cancel and rec.decode_calls >= 1)
                    if not cancel:
                        assert [s['finish'] for s in trial['states']] == ['EOS']*4
                    else:
                        assert any(s['finish'] == 'canceled' for s in trial['states'])
                    if label == 'cohort':
                        active = [v['rows'] for v in trial['calls'] if v['kind'] == 'decode']
                        assert active == [[0,1,2,3],[0,1,2,3],[1,2,3],[2,3],[3]], active
                        for row in range(3):
                            first = trial['states'][row]['token_events'][0]['monotonic']
                            next_prefill = next(v['start_monotonic'] for v in trial['calls'] if v['kind']=='prefill' and v['rows']==[row+1])
                            assert first <= next_prefill, 'First token must not wait for other prefill'
            except RuntimeError:
                failed = True
        assert failed == fail
        assert rec.reset_calls == 2 and rec.destroy_calls == 1
        assert not rec.callback_errors, rec.callback_errors
        assert sum(v['kind']=='create' for v in rec.calls) == 1
        final = saved[-1][1]
        assert final['Native_reset_before'] and final['Native_reset_after']
        if fail:
            assert any(phase=='failed_before_drain' for phase,_ in saved)
            assert final['failed'] and not final['normal']
        decode_rows = [v['rows'] for v in rec.calls if v['kind']=='forward' and v['decode']]
        if label == 'serial': assert all(len(v)==1 for v in decode_rows)
        results.append(dict(case=label, passed=True, actual_libffi=True,
            actual_Native=False, Native_leaves_recorded=True, high_context_pointer_preserved=True,
            reset_all_calls=2, normal_destroy_callback_calls=1,
            forward_count=sum(v['kind']=='forward' for v in rec.calls), decode_rows=decode_rows,
            wrong_owner_thread_rejected=True, packet_file_roundtrip=True,
            sampled_synthetic=True, no_actual_weights_or_memory=True))
    return dict(passed=True, cases=results, actual_new_caller=True,
        actual_original_forward_wrapper=True, actual_context_manager=True,
        Native_dlopen=False, Runtime_execution=False, weights=False,
        GPU=False, Model_READY=False, old_suites_replayed=False)


if __name__ == '__main__':
    # Local smoke uses no external artifacts. Remote CPU passes actual CPU945
    # metadata through entry.prepare and calls this identical checks function.
    from fixed_cohort.profile import config
    p=dict(config=config('recording-only'), cfg={'vocab_size':129280,'compress_ratios':[4]*43,'num_hidden_layers':43},
           mapping={}, receipt={}, records=[dict(row=i,case_id='recording-'+str(i),prompt_token_ids=tuple(range(1050+i)),max_new_tokens=4096) for i in range(4)])
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(checks(p, Path(d)/'private'), indent=2))
