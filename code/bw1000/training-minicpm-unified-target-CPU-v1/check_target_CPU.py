"""Check the installed unified entry and real target CPU initialization path."""
import hashlib,importlib.metadata as meta,importlib.util,json,os,runpy,signal,socket,sys,time,unittest
from pathlib import Path
epoch=float(os.environ['TARGET_CPU_EPOCH']);deadline=float(os.environ['TARGET_CPU_DEADLINE'])
assert deadline==epoch+150 and epoch<=time.time()<deadline
signal.alarm(max(1,int(deadline-time.time())))
assert socket.gethostname().split('.')[0]=='f11r4n09' and os.environ['SLURM_JOB_ID']=='1087656'
assert all(os.environ[x]=='-1' for x in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
out=Path('/evidence');start=time.monotonic()
owner=dict(node=socket.gethostname().split('.')[0],step='1087656.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),
           startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
           boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest())
with (out/'owned-node.json').open('x') as f:json.dump(owner,f)
result=None
try:
    # The original normal-entry check runs before importing Torch.
    runpy.run_path('/stage-tools/verify_entry.py',run_name='verify_shared_entry')
    import torch
    assert not torch.cuda.is_initialized() and torch.version.hip and str(torch.__version__).startswith('2.10.0')
    torch.set_num_threads(2);sys.path.insert(0,'/shared-training')
    methods=[];tests=[]
    for name,cls in [('test_initial_trainables.py','InitialTrainablesTest'),('test_hygon_vlm_recipe.py','RecipeDiagnosticTest')]:
        path=Path('/shared-training/vlm/tests')/name
        spec=importlib.util.spec_from_file_location(name[:-3],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module,cls))
        observed=unittest.TextTestRunner(verbosity=2).run(suite)
        tests.append(dict(name=name,tests=observed.testsRun,failures=len(observed.failures),errors=len(observed.errors)))
        assert observed.wasSuccessful()
    assert [x['tests'] for x in tests]==[4,6]
    runpy.run_path('/shared-training/vlm/plugins/native_dropout_alignment.py',run_name='register_native_initialization')
    from swift.callbacks import callbacks_map
    assert 'native_lora_dropout_alignment' in callbacks_map and not torch.cuda.is_initialized()
    versions={n:meta.version(n) for n in ('torch','transformers','ms-swift','peft')}
    assert versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1'
    result=dict(passed=True,owner=owner,tests=tests,versions=versions,HIP=torch.version.hip,
                callback_registry_imported=True,CUDA_initialized=False,original_model_loaded=False,
                fixture_only=True,wall_seconds=time.monotonic()-start,Model_GO=False)
except BaseException as e:
    result=dict(passed=False,owner=owner,error_type=type(e).__name__,message=str(e)[:300],
                wall_seconds=time.monotonic()-start,Model_GO=False)
finally:
    with (out/'target-CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
