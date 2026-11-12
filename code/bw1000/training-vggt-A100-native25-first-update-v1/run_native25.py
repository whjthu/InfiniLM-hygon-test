"""Original online VGGT camera-head training, with private first-update capture."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent;S=W/'src';F=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-native-observer-CPU-v1/framework')
assert sys.argv[1:]==[str(W/'train')]
epoch=float(os.environ['VGGT_FIRST_UPDATE_EPOCH']);deadline=float(os.environ['VGGT_FIRST_UPDATE_DEADLINE']);assert epoch<=time.time()<deadline and deadline-epoch==820
def expired(*_):raise TimeoutError('original native epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
sys.path.insert(0,str(F/'generation/src'))
import torch
assert torch.cuda.device_count()==1 and torch.version.cuda=='12.9' and not torch.version.hip
torch.set_num_threads(16);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from gencuda.backends.vggt_training import run_vggt_training
config=json.loads((S/'original-config.json').read_bytes());out=W/'train';out.mkdir(mode=0o700,exist_ok=False)
(out/'config.json').write_bytes((S/'original-config.json').read_bytes());start=time.monotonic()
result=run_vggt_training(Path('/mnt/data/nvidia-baseline-20261010/inputs/models/vggt-1b'),config,out,Path('/home/whj/nvidia-baseline/sources/vggt'))
m=result['metrics'];assert m['global_optimizer_step']==m['executed_optimizer_steps']==25 and m['measurement_steps']==20 and not m['frozen_aggregator_feature_cache']
p=json.loads((W/'capture/progress.json').read_bytes());assert p['phase']=='NATIVE25_RETURNED' and p['completed_global_steps']==25 and p['first_update_complete'] and not p['failed'] and len(p['files'])==351
result['diagnostic_environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,formal_performance=False,checkpoint_IO_and_original_timers_preserved=True,first_update_capture_includes_sync_D2H_serialization=True)
(out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(dict(native_global_steps=25,measured_steps=20,capture_files=351,formal_performance=False)))
