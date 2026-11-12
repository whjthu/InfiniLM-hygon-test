"""Bounded independent native fusion terminal and owned-device post consumer."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-native-fusion-A100-v1')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native fusion terminal reader expired')))
signal.alarm(80);start=time.monotonic();artifacts={};failure=None;post=None
def sha(raw):return hashlib.sha256(raw).hexdigest()
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()
  return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():
  p=W/n;assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p.read_bytes())==pin['SHA256']
 total=0
 for n in ('intent.json','GPU-owner.json','GPU-child-owner.json','GPU-result.json','FUSION_IDENTITY.json','TRAINABLE_METADATA.json','IMPORTED_SOURCE_FILES.json','fusion-progress.private.jsonl','progress.json','outcome.json','GPU.private.log','failure-private.json'):
  p=W/n
  if not p.exists():continue
  assert not p.is_symlink() and p.stat().st_size<=8*1024**2
  raw=p.read_bytes();total+=len(raw);assert total<=16*1024**2
  artifacts[n]=dict(bytes=len(raw),SHA256=sha(raw))
  if n.endswith('.json'):artifacts[n]['value']=json.loads(raw)
  elif n.endswith('.jsonl'):artifacts[n]['value']=[json.loads(line) for line in raw.splitlines() if line]
 i=artifacts['intent.json']['value'];o=artifacts['outcome.json']['value'];r=artifacts['GPU-result.json']['value']
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+1200 and r['deadline']==i['epoch']+1100
 assert o['returncode']==0 and o['passed'] and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['GPU_result']==r and artifacts['GPU-result.json']['SHA256']==o['GPU_result_SHA256']
 assert o['wall_seconds']<1200 and o['peak_group_RSS_bytes']<=64*1024**3 and len(o['GPU_affinity'])==4
 for owner in (o['owner'],o['child'],r['owner']):
  current=proc(owner['PID']);assert current is None or current['startticks']!=owner['startticks'] or current['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   row=proc(int(n));assert not(row and row['PGID']==o['child']['PGID'] and row['state']!='Z')
 assert r['passed'] and r['GPU_cards']==1 and r['full_native_pipeline_loaded'] and r['actual_preset_fusion_executed']
 assert r['base_tensors']==521 and r['preset_targets']==240 and r['actual_trainable_tensors']==476 and r['actual_trainable_elements']==9922560
 assert r['frozen_DiT_tensors']==521 and r['frozen_DiT_elements']==6154908736 and r['allocated_after']==r['reserved_after']==0
 assert r['actual_forward_calls']==r['actual_backward_calls']==r['actual_optimizer_updates']==0
 fusion=artifacts['FUSION_IDENTITY.json']['value'];assert fusion['passed'] and len(fusion['parameters'])==521 and len(fusion['preset_factor_SHA256'])==480
 assert artifacts['FUSION_IDENTITY.json']['SHA256']==r['FUSION_IDENTITY_SHA256']
 target_count=0;changed=0
 for n,p in fusion['parameters'].items():
  assert p['after_SHA256']==p['expected_SHA256'];target_count+=p['preset_target']
  if not p['preset_target']:assert p['after_SHA256']==p['before_SHA256']
  changed+=p['after_SHA256']!=p['before_SHA256']
 assert target_count==240 and 0<changed<=240 and changed==fusion['changed_target_tensors']==r['changed_target_tensors']
 assert fusion['preset_input_factors_unchanged'] and fusion['RNG_unchanged'] and not fusion['independent_backend']
 tm=artifacts['TRAINABLE_METADATA.json']['value'];expected=json.loads((W/'src/TRAINABLE_METADATA.json').read_bytes())
 assert set(tm)==set(expected) and len(tm)==476 and sum(p['elements'] for p in tm.values())==9922560
 assert all(p['shape']==expected[n]['shape'] and p['dtype']=='torch.bfloat16' and p['requires_grad'] for n,p in tm.items())
 providers=artifacts['IMPORTED_SOURCE_FILES.json']['value'];binding=json.loads((W/'src/NATIVE_BINDING.json').read_bytes());source=Path(binding['source_root'])
 assert artifacts['IMPORTED_SOURCE_FILES.json']['SHA256']==r['source_provider_manifest_SHA256'] and len(providers)==r['source_provider_files']
 for pin in providers.values():
  p=source/pin['relative_path'];assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p.read_bytes())==pin['SHA256']
 for n in ('trainable-initial.private.safetensors','post-constructor-RNG.private.pt'):
  p=W/n;assert not p.is_symlink() and p.stat().st_size<=24*1024**2
  h=hashlib.sha256()
  with p.open('rb') as stream:
   while True:
    raw=stream.read(1024**2)
    if not raw:break
    h.update(raw)
  artifacts[n]=dict(bytes=p.stat().st_size,SHA256=h.hexdigest(),values_transferred=False)
 assert artifacts['trainable-initial.private.safetensors']['SHA256']==r['initial_trainable_SHA256']
 assert artifacts['trainable-initial.private.safetensors']['bytes']==r['initial_trainable_bytes']
 assert artifacts['post-constructor-RNG.private.pt']['SHA256']==r['RNG_SHA256']
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 selected=[[v.strip() for v in line.split(',')] for line in cards if GPU_UUID in line]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 post=dict(selected=selected,selected_compute_apps=[line for line in apps if GPU_UUID in line],owned_GPU_UUID=GPU_UUID,owned_GPU_PCI=GPU_PCI)
 assert len(selected)==1 and selected[0][1]==GPU_UUID and selected[0][2]==GPU_PCI and int(selected[0][3])<=16 and not post['selected_compute_apps']
 assert r['independent_numerical_backend'] is False and not r['numerical_training_qualified'] and not r['formal_performance'] and not r['Model_GO']
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-7000:])
print(json.dumps(dict(passed=failure is None,failure=failure,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,
 artifacts=artifacts,post=post,source_files=len(EXPECTED_FILES),reader_original_weight_bytes_read=0,reader_private_initial_values_exported=False,
 new_CPU_or_GPU_launches=0,inside_seconds=time.monotonic()-start,Model_GO=False)))
if failure is not None:raise SystemExit(1)
