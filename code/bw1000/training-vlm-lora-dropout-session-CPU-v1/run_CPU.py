"""Exercise real PEFT and AdamW on CPU; never initialize CUDA or load a checkpoint."""
import importlib.metadata as metadata,importlib.util,json,os,runpy,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;epoch=float(os.environ['SESSION_CPU_EPOCH']);deadline=float(os.environ['SESSION_CPU_DEADLINE'])
assert deadline-epoch==180 and epoch<=time.time()<deadline
signal.alarm(max(1,min(150,int(deadline-time.time()-15))))
started=time.monotonic();result={}
try:
 assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='-1'
 import torch,unittest
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 assert str(torch.__version__)=='2.8.0a0+5228986c39.nv25.06'
 sys.path.insert(0,str(ROOT/'framework'))
 path=ROOT/'framework/vlm/tests/test_lora_dropout_session.py';spec=importlib.util.spec_from_file_location('test_native_session',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.LoRADropoutSessionTest)
 observed=unittest.TextTestRunner(verbosity=2).run(suite)
 runpy.run_path(str(ROOT/'framework/vlm/plugins/native_dropout_alignment.py'),run_name='registered_native_dropout_alignment')
 from swift.callbacks import callbacks_map
 assert 'native_lora_dropout_alignment' in callbacks_map
 assert not torch.cuda.is_initialized()
 result={'state':'REAL_PEFT_ADAMW_NATIVE_DROPOUT_SESSION_CPU_PASS' if observed.wasSuccessful() else 'REAL_PEFT_ADAMW_NATIVE_DROPOUT_SESSION_CPU_FAIL','tests_run':observed.testsRun,'failures':len(observed.failures),'errors':len(observed.errors),'torch':str(torch.__version__),'peft':metadata.version('peft'),'swift':metadata.version('ms-swift'),'callback_registry_imported':True,'GPU_initialized':False,'checkpoint_loaded':False,'model_qualified':False,'formal_performance':False,'wall_seconds':time.monotonic()-started}
except BaseException as error:
 result={'state':'CPU_SESSION_EXECUTION_FAILED','error_type':type(error).__name__,'message':str(error)[:400],'wall_seconds':time.monotonic()-started}
finally:
 with (ROOT/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if result['state']!='REAL_PEFT_ADAMW_NATIVE_DROPOUT_SESSION_CPU_PASS':raise SystemExit(1)
