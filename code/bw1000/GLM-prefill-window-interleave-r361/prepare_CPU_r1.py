"""Extend the successful688 whole-owner harness, not a standalone new helper."""
from pathlib import Path
import ast
D=Path(__file__).resolve().parent;BASE=D.parent/'GLM-two-cohort-Model-r355';OUT=D/'runtime-r1'
s=(BASE/'CPU-repair-r2/CPU_whole.py').read_text()
s=s.replace("HERE=pathlib.Path('/refactor/incoming/GLM-two-cohort-Model-r355')", "HERE=pathlib.Path(__file__).resolve().parent")
s=s.replace('GLM355','GLM361').replace('CPU355','CPU361').replace('GLM_R355','GLM_R361').replace('GLM-r355','GLM-r361')
s=s.replace("result=dict(completed=False,passed=False,GPU=0", "result=dict(completed=False,passed=False,actual_CPU=step,CPU_only=True,CPU_source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),GPU=0")
s=s.replace("token=154820 if l.generated[row]>=(3 if row==0 else 5)","token=154820 if l.generated[row]>=(9 if row==0 else 11)")
s=s.replace("    class Function:","    fault={'kind':None}\n    class Function:")
s=s.replace("a[6].zero_();a[6][0,token]=1;l.trace.append(('F',row,past,T))", "a[6].zero_();a[6][0,token]=1;l.trace.append(('F',row,past,T))\n                if fault['kind']=='cancel'and len(l.trace)==5:pathlib.Path(os.environ['GLM_R361_CANCEL_FILE']).write_text('{}')\n                if fault['kind']=='partial_error'and len(l.trace)==6:return 1")
needle="        result['cases'].append(dict(name='actual_owner_client_prepared_pair_admissions',passed=True,requests=12,CPU_semantic_gold_wire_mock=True,synthetic_logits=True))"
assert needle in s
s=s.replace(needle,needle+'''
        for i,pair in enumerate(emitter['pairs']):
            hit=pair['prefill_consumption'];assert hit['qualified'] and pair['decode_defer_fixed_both_arms']
            assert hit['enabled']==PLAN[i]
            assert hit['complete_long_W4_groups']==(5 if PLAN[i]else 0)
            assert hit['short_advances']==(5 if PLAN[i]else 0)
            assert [r['tokens']for r in pair['requests']]==[9,11]
            assert pair['requests'][1]['generated_ids'][0]==101,'Long retained logits overwritten by intervening short forward'
            events=[e for e in pair['feedback_events']if e['phase']=='prefill']
            assert [e['row']for e in events]==([0,0,0,0,0,0,1]if PLAN[i]else[0,1])
        result['cases'].append(dict(name='actual_owner_boundary_hit_past_feedback_and_retained_logits',passed=True,ON_groups=5,ON_short_advances=5,OFF_advances=0,defer_fixed_ON=True,full_timeline_scope=True))
''')
needle="        result['cases'].append(dict(name='all32_factory_argv_actual_owner_and_cleanup',passed=True,owners=owners))"
assert needle in s
negative='''
        for kind,expected_calls in [('cancel',5),('partial_error',6)]:
            prepared=prepare_app('GLM','/old/models/ZhipuAI-GLM-5.3');engine=prepared['engine']
            tag_negative='GLM361-CPU-'+step+'-'+kind
            os.environ.update(GLM_RUN_R5040=tag_negative,GLM_R361_CANCEL_FILE=str(R/'control'/('GLM-r361-cancel-'+tag_negative+'.json')))
            fault['kind']=kind;first[0]=True;tape.clear();cursor[0]=0;errors.clear();processes.clear()
            failure_report={'library_sha256':emitter['library_sha256']};failure_library=Library();failure_path=out/(kind+'.json')
            with patch.object(owner,'PLAN',(True,)),patch.object(client,'PLAN',(True,)):
                try:owner.run(failure_library,object(),3,0,payload,record_control,check,lambda x:x,lambda:0,154880,failure_report,failure_path,lambda *a:None,prepared)
                except (InterruptedError,RuntimeError)as error:failure_type=type(error).__name__
                else:raise AssertionError('Partial/cancel Native failure was silently continued')
            assert len(failure_library.trace)==expected_calls
            assert failure_library.counts['modern_glm_model_reset']==2,'No reset/replay after failed partial work'
            assert not failure_report['normal_shutdown'] and not failure_report['normal_reset_close'] and not failure_report['pairs']
            assert failure_report.get('service_traceback') and all(not p.thread.is_alive()for p in processes)
            if kind=='partial_error':assert failure_library.past[0]==480 and failure_library.generated[0]==2
            result['cases'].append(dict(name='actual_owner_'+kind+'_failstop',passed=True,Native_forward_calls=expected_calls,no_forward_or_reset_after_failure=True,frontend_not_replayed=True,error_type=failure_type,Native_is_recording=True))
        fault['kind']=None
        # Execute the actual new worker catch branch and original guarded exit2.
        owner_try=next(n for n in worker.body if isinstance(n,ast.Try)and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='_preloaded_probe_r3091'for x in ast.walk(n)))
        failure_lib=Library();failure_path=out/'actual-worker-failstop.json';failure_report={}
        def partial(*a):raise RuntimeError('CPU recording Native already advanced; forbidden ledger replay')
        env=dict(_preloaded_probe_r3091=partial,lib=failure_lib,ctx=object(),stage=3,rank=0,payload={},control=lambda *a:None,check=check,ptr=lambda x:x,stream=lambda:0,vocab=154880,report=failure_report,report_path=failure_path,phase_marker=lambda *a:None,_protocol_preweight={'prepared':prepared},traceback=traceback,json=json)
        try:exec(compile(ast.Module(body=[owner_try],type_ignores=[]),'actual361-worker-catch','exec'),env)
        except RuntimeError:pass
        else:raise AssertionError('Actual worker swallowed Native failure')
        assert failure_report['failed_owner_failstop'] and not failure_lib.counts
        guarded=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='guarded')
        class ExpectedExit(BaseException):pass
        def exit2(code):raise ExpectedExit(code)
        env=dict(worker=partial,traceback=traceback,r=R,tag='GLM361-CPU-'+step+'-guarded',stage=3,json=json,os=os)
        exec(compile(ast.Module(body=[guarded],type_ignores=[]),'actual361-guarded-exit','exec'),env)
        with patch.object(os,'_exit',side_effect=exit2):
            try:env['guarded'](0,{})
            except ExpectedExit as error:assert error.args==(2,)
            else:raise AssertionError('Original guarded failstop exit2 missing')
        result['cases'].append(dict(name='actual_worker_partial_error_branch_and_guarded_exit2',passed=True,no_Native_reset_destroy_or_shutdown_on_error=True,original_whole_set_controller_cleanup_required=True))
'''
s=s.replace(needle,needle+negative)
s=s.replace("'actual355-worker-normal-cleanup'","'actual361-worker-normal-cleanup'").replace("'actual355-create-call'","'actual361-create-call'")
s=s.replace("    assert not torch.cuda.is_initialized()", "    from deadline_r290 import check as original_check\n    with patch.dict(os.environ,{'GLM_R290_GPUSTART_UNIX':str(time.time()-6121)}):\n        try:original_check('CPU expired original residency')\n        except TimeoutError:pass\n        else:raise AssertionError('Original deadline silently reset')\n    result['cases'].append(dict(name='original_absolute_deadline',passed=True,no_Native=True))\n    assert not torch.cuda.is_initialized()")
s=s.replace("    (out/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))", "    raw=json.dumps(result,indent=2).encode()+bytes([10]);(out/'CPU-result.json').write_bytes(raw)\n    if result['passed']:\n        candidate=R/'candidates/GLM-prefill-window-interleave-r361/runtime-r1';candidate.mkdir(parents=True,exist_ok=False);(candidate/'CPU-result.json').write_bytes(raw)\n    print(json.dumps(result))")
ast.parse(s);(OUT/'CPU_whole.py').write_bytes(s.encode())
check=(BASE/'check_controller_CPU.py').read_text().replace('r355','r361')
check=check.replace("    q['no_WORLD_in_F0_F1_callbacks']=True", "    q['no_WORLD_in_F0_F1_callbacks']=True\n    assert len([n for n in ast.walk(t)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='prefill_two_rows'])==1\n    assert any(isinstance(n,ast.keyword)and n.arg=='enabled'and isinstance(n.value,ast.Constant)and n.value.value is True for n in ast.walk(t))\n    q['actual_interleave_consumer_and_fixed_defer_source']=True")
ast.parse(check);(OUT/'check_controller_CPU.py').write_bytes(check.encode())
shell=(BASE/'CPU-repair-r2/CPU_inside.sh').read_bytes().replace(b'GLM-two-cohort-Model-r355',b'GLM-prefill-window-interleave-r361/runtime-r1').replace(b'CPU-repair-r2/CPU_whole.py',b'CPU_whole.py')
assert b'\r'not in shell;(OUT/'CPU_inside.sh').write_bytes(shell)
print('361 CPU harness AST/source prepared; no GPU')
