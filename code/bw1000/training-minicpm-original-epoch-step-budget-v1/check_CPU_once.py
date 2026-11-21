"""New real-argv/timeout fixtures against an assembled original controller."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from assemble_candidate import assemble,G,ORIGINAL

OUT=G/'local-only/training-minicpm-original-epoch-step-budget-v1-CPU-corrected-original'


def load(source):
    module=types.ModuleType('minicpm_budget_candidate');module.__file__=str(ORIGINAL)
    sys.path.insert(0,str(G/'tools'))
    exec(compile(source,str(ORIGINAL)+':two-function-unpublished-budget-candidate','exec'),module.__dict__)
    return module


def main():
    started=time.monotonic();OUT.mkdir()
    source,proof=assemble();driver=load(source);checks=[];captures=[];original_bytes=ORIGINAL.read_bytes()
    original=load(original_bytes)
    old_gpu=original.step_command('gpu',['NO_REAL_COMMAND'])
    counterexample={'fixture_elapsed':1910,'old_GPU_time_minutes':int(old_gpu[old_gpu.index('-t')+1]),
       'old_planned_step_end_offset_without_termination':1910+2100,'original_whole_offset':2400,
       'old_controller_actual_source_queried_or_launched':False}
    def exercise(phase,used,timeout=False):
        rows=[]
        with tempfile.TemporaryDirectory(prefix='minicpm-budget-fixture-') as t:
            def run(argv,**kw):
                rows.append({'argv':argv,'timeout':kw['timeout']})
                if timeout:raise subprocess.TimeoutExpired(argv,kw['timeout'])
                return types.SimpleNamespace(returncode=0)
            fake_sub=types.SimpleNamespace(run=run,STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired,SubprocessError=subprocess.SubprocessError)
            def terminal(query,deadline,max_observations):
                return dict(terminal_visible=True,row=[driver.PARENT+'.123','fixture','CANCELLED' if timeout else 'COMPLETED',
                            '1:0' if timeout else '0:0','1',driver.NODE,'gres/dcu='+str(driver.PHASES[phase][0])],observations=[])
            with patch.object(driver,'RUN',Path(t)),patch.object(driver,'accounting',return_value=None),patch.object(driver,'subprocess',fake_sub),patch.dict(sys.modules,{'slurm_accounting':types.SimpleNamespace(wait_terminal=terminal)}):
                result=driver.execute_step(phase,['NO_REAL_COMMAND'],lambda:used,1000.)
                assert dict(epoch=1000.,**result)==json.loads((Path(t)/(phase+'-accounting.json')).read_bytes())
            assert len(rows)==1
            row=rows[0];minutes=int(row['argv'][row['argv'].index('-t')+1])
            assert minutes>=1 and row['timeout']>0 and minutes*60+5<=row['timeout']
            if phase=='gpu':assert used+row['timeout']<=2100 and used+minutes*60+5<=2100
            else:assert minutes*60<=driver.PHASES[phase][3] and used+row['timeout']+26<2400
            if timeout:assert result['client_timeout'] and not result['passed'] and result['row'][2]=='CANCELLED'
            else:assert result['passed']
            captures.append({'phase':phase,'fixture_elapsed':used,'actual_captured_subprocess_argv':row['argv'],
                             'actual_captured_timeout':row['timeout'],'fixture_only':True,'timeout_branch':timeout,'Slurm_time_minutes':minutes})
            return minutes
    assert exercise('gpu',30)==34;checks.append('normal_GPU_actual_argv34_client2070_original2100end')
    assert exercise('gpu',1910)==3;checks.append('late_GPU_actual_argv3_client190_original2100end')
    assert exercise('pre',0)==1;checks.append('pre_actual_one_minute_bounded_CPU')
    assert exercise('post',2100)==1;checks.append('post_actual60_not_ceil120_exceeds90cap')
    assert exercise('gpu',1910,True)==3;checks.append('timeout_branch_uses_same_shrunken_argv_and_preserves_cancelled_terminal')
    for used in [1920,2400,float('nan'),float('inf'),-1,True]:
        with patch.object(driver,'accounting',return_value=None),patch.dict(sys.modules,{'slurm_accounting':types.SimpleNamespace(wait_terminal=lambda *a,**k:None)}):
            try:driver.execute_step('gpu',['NO_REAL_COMMAND'],lambda:used,1000.)
            except ValueError:checks.append('reject_original_clock_'+repr(used))
            else:raise AssertionError('Invalid/expired phase clock was not rejected')
    for seconds in [0,59,2101,True,float('nan')]:
        try:driver.step_command('gpu',['NO_REAL_COMMAND'],seconds=seconds)
        except ValueError:checks.append('reject_unlimited_short_or_oversize_time_'+repr(seconds))
        else:raise AssertionError('Unsafe time value was not rejected')
    # Existing five controller tests run against the NEW candidate, not old
    # worker fixtures or any original job/model/native processor.
    spec=importlib.util.spec_from_file_location('budget_regression_cases',G/'tools/test_control_minicpm_usgs_pair_target.py')
    tests=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'control_minicpm_usgs_pair_target':driver}):spec.loader.exec_module(tests)
    stream=io.StringIO();regression=unittest.TextTestRunner(stream=stream,verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert regression.wasSuccessful() and regression.testsRun==5,stream.getvalue()
    assert ORIGINAL.read_bytes()==original_bytes,'Frozen controller was modified'
    result={'schema':'MINICPM_ORIGINAL_EPOCH_STEP_BUDGET_LOCAL_FIXTURES_V1','passed':True,'candidate':proof,
       'old_source_counterexample':counterexample,'new_checks_passed':len(checks),'new_checks':checks,'captured_actual_call_boundaries':captures,
       'candidate_existing_controller_regressions':{'passed':True,'run':regression.testsRun,'failures':0,'errors':0},
       'original_controller_unchanged':True,'fixture_only':True,'subprocess_and_accounting_doubles':True,
       'actual_Slurm_or_GPU_or_native_calls':0,'actual_optimizer_updates':0,'actually_installed_or_adopted':False,
       'native_worker_core_math_or2400_2100_1980_caps_changed':False,'READY':False,'wall_seconds':time.monotonic()-started}
    p=OUT/'result.json';p.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'new_checks':len(checks),'regression_tests':5,'actual_Slurm_calls':0,
           'result':{'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}}))


if __name__=='__main__':main()
