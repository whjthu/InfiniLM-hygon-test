"""Fresh process restores original checkpoint20 and runs original updates21..25."""
from pathlib import Path
import hashlib,json,math,os,signal,sys,time
W=Path(__file__).resolve().parent.parent;S=W/'src';F=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-native-observer-CPU-v1/framework');assert sys.argv[1:]==[str(W/'train')]
epoch=float(os.environ['VGGT_NATIVE_EPOCH']);deadline=float(os.environ['VGGT_NATIVE_DEADLINE']);assert epoch<=time.time()<deadline and deadline-epoch==820
assert not any(k.startswith('VGGT_FIRST_UPDATE_') for k in os.environ)
def expired(*_):raise TimeoutError('original resume epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
prior=json.loads((S/'PRIOR_REFERENCE.json').read_bytes());cp=Path(prior['checkpoint20']['path']);assert cp.is_file() and not cp.is_symlink() and cp.stat().st_size==prior['checkpoint20']['bytes'] and sha(cp)==prior['checkpoint20']['SHA256']
sys.path.insert(0,str(F/'generation/src'))
import torch
assert torch.cuda.device_count()==1 and torch.version.cuda=='12.9' and not torch.version.hip
torch.set_num_threads(16);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from gencuda.backends.vggt_training import run_vggt_training
config=json.loads((S/'original-config.json').read_bytes());config['resume_from_checkpoint']=str(cp);out=W/'train';out.mkdir(mode=0o700,exist_ok=False);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n');started=time.monotonic()
result=run_vggt_training(Path('/mnt/data/nvidia-baseline-20261010/inputs/models/vggt-1b'),config,out,Path('/home/whj/nvidia-baseline/sources/vggt'))
m=result['metrics'];assert m['completed_global_step_before_resume']==20 and m['global_optimizer_step']==25 and m['executed_optimizer_steps']==m['measurement_steps']==5 and m['resume_source_sha256']==prior['checkpoint20']['SHA256'] and result['checks']['resume_boundary'] and result['checks']['frozen_aggregator_unchanged']
original=Path(prior['root']);old=json.loads((original/'train/result.json').read_bytes());assert sha(original/'train/result.json')==prior['result_SHA256']
refsteps=[json.loads(line) for line in (original/'train/steps.ndjson').read_bytes().splitlines() if line];newsteps=[json.loads(line) for line in (out/'steps.ndjson').read_bytes().splitlines() if line];refsteps=[v for v in refsteps if v['global_optimizer_step']>=21]
assert len(refsteps)==len(newsteps)==5 and [v['global_optimizer_step'] for v in newsteps]==list(range(21,26))
pairs=[dict(step=b['global_optimizer_step'],continuous_batch_cursor=a['batch_cursor'],resumed_batch_cursor=b['batch_cursor'],continuous_loss=a['loss'],resumed_loss=b['loss'],absolute_loss_error=abs(a['loss']-b['loss']),loss_exact=a['loss']==b['loss']) for a,b in zip(refsteps,newsteps)]
comparison=dict(original_checkpoint20_SHA256=prior['checkpoint20']['SHA256'],original_continuous_result_SHA256=prior['result_SHA256'],same_native_identity=m['checkpoint_identity']==old['metrics']['checkpoint_identity'],same_final_boundary_SHA256=m['final_state_sha256']==old['metrics']['final_state_sha256'],continuous_final_boundary_SHA256=old['metrics']['final_state_sha256'],resumed_final_boundary_SHA256=m['final_state_sha256'],pairs=pairs,all_batch_cursors_equal=all(v['continuous_batch_cursor']==v['resumed_batch_cursor'] for v in pairs),all_losses_exact=all(v['loss_exact'] for v in pairs),fresh_process=True,formal_performance=False)
comparison['passed']=comparison['same_native_identity'] and comparison['same_final_boundary_SHA256'] and comparison['all_batch_cursors_equal'] and comparison['all_losses_exact'];result['fresh_process_comparison']=comparison
result['diagnostic_environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-started,formal_performance=False,no_first_update_capture=True,original25_window_preserved=True)
(out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(dict(native_global_step=25,resumed_updates=5,comparison_passed=comparison['passed'],formal_performance=False)))
