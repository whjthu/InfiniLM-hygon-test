from pathlib import Path
import base64,hashlib,json,socket,subprocess,time
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hymotion-target-meta-CPU-v2');started=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
intent=json.loads((W/'intent.json').read_bytes());out=json.loads((W/'outcome.json').read_bytes());owned=json.loads((W/'owned-srun.json').read_bytes());assert intent['claim']==out['claim']==CLAIM and intent['request_id']==out['request_id']==REQUEST and intent['epoch']==out['epoch'] and out['deadline']==intent['deadline']==intent['epoch']+300;assert out['owned_srun_reaped'] and not out['forced_cleanup'];assert sha(W/'outcome.json')==EXPECTED_OUTCOME_SHA if EXPECTED_OUTCOME_SHA else True
pins={}
for n in REMOTE_FILES:
 p=W/'source'/n;assert p.is_file() and not p.is_symlink() and sha(p)==SOURCES[n]==intent['source_files'][n];pins[n]=sha(p)
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];assert out['accounting_columns']==columns and len(out['accounting'])==1;row=out['accounting'][0];assert row[1]=='training-hymotion-target-meta-CPU-v2' and row[5]=='f11r4n15';assert row[2] in ['COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED']
fresh=[line.split('|') for line in subprocess.check_output(['sacct','-n','-P','-j',row[0],'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if line.split('|')[0]==row[0]];assert fresh==[row]
p=W/'CPU.log';assert p.stat().st_size<262144 and sha(p)==out['log_SHA256'];raw=p.read_bytes();original_alive=None
if owned['host']==socket.gethostname():
 try:stat=Path('/proc',str(owned['PID']),'stat').read_text().split(') ',1)[1].split();original_alive=int(stat[19])==owned['startticks']
 except FileNotFoundError:original_alive=False
 assert original_alive is False
qpath=W/'evidence/CPU-result.json';result=json.loads(qpath.read_bytes()) if qpath.is_file() else None
if result:
 assert sha(qpath)==out['CPU-result.json']['SHA256'] and result==out['CPU-result.json']['value'];assert result['owner']['step']==row[0]
assert out['passed'] and row[2:4]==['COMPLETED','0:0'] and result is not None and result['passed'] and not result['CUDA_initialized'] and not result['Model_GO']
meta=result['native_meta_result']['value'];mp=W/'evidence/native-meta-result.json';assert sha(mp)==result['native_meta_result']['SHA256'] and mp.stat().st_size==result['native_meta_result']['bytes'] and json.loads(mp.read_bytes())==meta
spec=json.loads((W/'source/META_SPEC.json').read_bytes());assert meta['trainable_manifest']==spec['reference_trainable_manifest'] and meta['lora_layers']==spec['reference_lora_layers'] and meta['base_parameters_meta']==spec['reference_base_parameters_meta']==1042868681;assert meta['trainable_tensors']==160 and meta['trainable_parameters']==1638400 and len(meta['lora_layers'])==80
assert meta['optional_GPU_extensions_excluded']==['flash_attn','flash_attn_2_cuda'] and not meta['production_attention_qualified'] and meta['GC_canonical_manifest_unchanged'] and meta['base_frozen'] and meta['optimizer_empty_and_exact_binding'];assert meta['GPU_cards']==meta['weights_or_cache_tensor_bytes_read']==meta['forward_backward_updates']==0 and not meta['CUDA_initialized'] and not meta['Model_GO'] and not meta['frame_length_qualified'] and meta['owner']['PID']==result['owner']['PID']
assert meta['epoch']==result['epoch']==out['epoch'] and meta['deadline']==result['deadline']==out['epoch']+220 and meta['wall_seconds']<220 and meta['peak_RSS_bytes']<4294967296
fm=json.loads((W/'source/FRAMEWORK_SUBSET.json').read_bytes());assert fm['total_framework_files']==333 and len(fm['files'])==10 and len(result['runtime_unified_modules'])==9
for n,pin in result['runtime_unified_modules'].items():assert pin['SHA256']==fm['files'][pin['path']]['SHA256']
assert len(meta['runtime_project_modules'])==10
for n,pin in meta['runtime_project_modules'].items():assert sha(W/'source/upstream'/pin['path'])==pin['SHA256']==spec['upstream_files'][pin['path']]['SHA256']
provider_maps={}
for n,pin in result['providers'].items():
 native=Path(pin['path']);assert native.is_absolute()
 if str(native).startswith('/training-runtime/'):
  p=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-t06-sdxl-runtime-v1')/native.relative_to('/training-runtime')
 else:
  assert str(native).startswith('/usr/local/');p=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1')/str(native).lstrip('/')
 assert p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];provider_maps[n]=dict(container_path=str(native),HOST_path=str(p),SHA256=sha(p))
answer=dict(provider_maps=provider_maps,evidence_verified=True,qualification_passed=bool(out['passed']),original_outcome=out,actual_CPU_result=result,fresh_accounting=fresh,accounting_columns=columns,source_files_verified=pins,original_intent=intent,original_owner=owned,original_owner_alive=original_alive,reader_host=socket.gethostname(),original_artifacts={n:dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n)) for n in ['intent.json','owned-srun.json','outcome.json','CPU.log']},CPU_log_base64=base64.b64encode(raw).decode(),reader_imported_Torch=False,reader_GPU_cards=0,weights_or_cache_bytes_read=0,reader_wall_seconds=time.monotonic()-started)
print(json.dumps(answer,allow_nan=False))
