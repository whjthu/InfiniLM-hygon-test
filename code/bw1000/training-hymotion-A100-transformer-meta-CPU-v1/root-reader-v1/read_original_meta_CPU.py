from pathlib import Path
import hashlib,json,math,os,socket,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hymotion-transformer-meta-CPU-v1');started=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p):
 try:rest=Path('/proc',str(p['PID']),'stat').read_text().split(') ',1)[1].split()
 except FileNotFoundError:return dict(original_alive=False,state='MISSING')
 return dict(original_alive=int(rest[19])==p['startticks'],state='MATCH' if int(rest[19])==p['startticks'] else 'PID_REUSED')
intent=json.loads((W/'intent.json').read_bytes());out=json.loads((W/'outcome.json').read_bytes());owner=json.loads((W/'owned-child.json').read_bytes());result=json.loads((W/'CPU-result.json').read_bytes());assert intent['claim']==out['claim']==CLAIM and intent['request_id']==out['request_id']==REQUEST and intent['host']==owner['host']==socket.gethostname()
assert out['passed'] and out['returncode']==0 and out['owned_child_reaped'] and not out['forced_cleanup'] and out['error_type'] is None
assert intent['epoch']==out['epoch']==result['epoch'] and out['deadline']==intent['deadline']==intent['epoch']+240 and result['deadline']==intent['epoch']+180 and out['wall_seconds']<240 and result['wall_seconds']<180
assert sha(W/'CPU-result.json')==out['CPU-result.json']['SHA256']==EXPECTED_RESULT_SHA and (W/'CPU-result.json').stat().st_size==out['CPU-result.json']['bytes'];assert sha(W/'CPU.log')==out['CPU_log_SHA256'];log=[json.loads(x) for x in (W/'CPU.log').read_bytes().splitlines()];assert len(log)==1 and log[0]['passed'] and log[0]['trainable_parameters']==1638400
published={}
for n,pin in SOURCES.items():
 p=W/'source'/n;assert sha(p)==pin and intent['source_files'][n]==pin;published[n]=pin
up=json.loads((W/'source/UPSTREAM_FILES.json').read_bytes());U=Path(up['root']);upstream={}
for n,pin in up['files'].items():
 p=U/n;assert not p.is_symlink() and p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];upstream[n]=pin
assert len(upstream)==16 and sum(x['bytes'] for x in upstream.values())==148217
modules=result['runtime_project_modules'];assert len(modules)==10
for n,pin in modules.items():assert pin['SHA256']==upstream[pin['path']]['SHA256']
assert result['base_parameters_meta']==1042868681 and result['trainable_tensors']==160 and result['trainable_parameters']==1638400 and len(result['lora_layers'])==80
train=result['trainable_manifest'];assert len(train)==160 and sum(v['numel'] for v in train.values())==1638400 and all(v['dtype']=='torch.float32' and v['meta'] and math.prod(v['shape'])==v['numel'] for v in train.values());assert all(n.endswith(('.lora_A','.lora_B')) for n in train)
assert result['GC_canonical_manifest_unchanged'] and result['base_frozen'] and result['optimizer_empty_and_exact_binding'];assert result['GPU_cards']==result['weights_or_cache_tensor_bytes_read']==result['forward_backward_updates']==0 and not result['CUDA_initialized'] and not result['Model_GO'] and not result['full_pipeline_constructed'] and not result['frame_length_qualified'];assert result['peak_RSS_bytes']<4294967296 and len(result['owner']['affinity'])==2
pids=dict(timeout_child=identity(owner),native_model_child=identity(result['owner']));assert not any(x['original_alive'] for x in pids.values())
files={n:dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n)) for n in ['intent.json','owned-child.json','outcome.json','CPU-result.json','CPU.log','model-construction.private.txt']}
answer=dict(passed=True,claim=CLAIM,request_id=REQUEST,actual_result=result,original_artifacts=files,verified_published_files=published,upstream_files=upstream,original_processes=pids,actual_epoch=intent['epoch'],whole_deadline=intent['deadline'],owned_child_reaped=True,forced_cleanup=False,independent_reader_imported_Torch=False,independent_reader_GPU_cards=0,independent_reader_weights_or_cache_bytes=0,reader_wall_seconds=time.monotonic()-started)
print(json.dumps(answer,allow_nan=False))
