"""Independent stdlib reader of the single original CPU process and its providers."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
transport=json.loads((D/'transport-outcome.json').read_bytes());assert transport['transport_rc']==0 and transport['value']['passed']
claim=transport['claim'];request=transport['request_id'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-llava-A100-native-contract-CPU-v1.json']))
remote=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nPINS='+repr({n:p for n,p in task['source_SHA256'].items() if n=='run_checks.py' or n.startswith('framework/')})+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,os,sys
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-llava-native-update-contract-CPU-v1');artifacts={}
def read(name,cap=2*1024**2):
 p=W/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<=cap
 raw=p.read_bytes();artifacts[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return raw
out=json.loads(read('outcome.json'));cpu=json.loads(read('CPU-result.json'));intent=json.loads(read('intent.json'))
assert out['claim']==intent['claim']==CLAIM and out['request_id']==intent['request_id']==REQUEST
assert out['passed'] and out['returncode']==0 and out['stop_reason'] is None and out['owned_group_reaped'] and out['CPU_result']==cpu and out['CPU_result_SHA256']==artifacts['CPU-result.json']['SHA256']
assert out['epoch']==cpu['epoch']==intent['epoch'] and out['deadline']==intent['deadline']==out['epoch']+300 and cpu['deadline']==out['epoch']+260
assert 0<cpu['wall_seconds']<260 and cpu['wall_seconds']<=out['wall_seconds']<300 and out['peak_group_RSS_bytes']<=8*1024**3 and len(out['CPU_affinity'])==4
assert cpu['passed'] and cpu['shared_tests']==24 and len(cpu['rejected_boundaries'])==21 and len(set(cpu['rejected_boundaries']))==21
assert cpu['actual_model_class']=='LLaVAOneVision1_5_ForConditionalGeneration' and cpu['actual_language_class']=='LLaVAOneVision1_5_TextModel' and cpu['actual_vision_class']=='RiceTransformerPretrainedModel'
assert cpu['resolved_template_type']=='llava_onevision1_5' and cpu['all_native_eager_attention'] and cpu['language_layers']==36 and cpu['vision_layers']==24 and cpu['language_GC'] is cpu['vision_GC'] is False
assert cpu['native_trainable_tensors']==504 and cpu['native_trainable_parameters']==43646976 and cpu['real_Swift_prepare_adapter'] and cpu['real_PEFT'] and cpu['empty_AdamW_bound'] and cpu['callback_registered'] and cpu['metadata_files_verified']==14
assert cpu['GPU_initialized'] is False and cpu['weight_file_open_attempts']==[] and cpu['model_weight_values_read'] is False and not cpu['initial_values_qualified'] and not cpu['materialized_RoPE_repair_qualified'] and not cpu['training_forward_backward']
for name,pin in PINS.items():
 path=name if name.startswith('framework/') else 'src/'+name
 assert hashlib.sha256(read(path)).hexdigest()==pin
binding=json.loads(read('LLAVA_NATIVE_BINDING.json'));providers=json.loads(read('providers.json'));owners=[json.loads(read(n)) for n in ('CPU-owner.json','CPU-child-owner.json')]
assert cpu['binding_SHA256']==artifacts['LLAVA_NATIVE_BINDING.json']['SHA256'] and cpu['providers_SHA256']==artifacts['providers.json']['SHA256']
assert owners[0]['PID']==owners[1]['PID']==out['child']['PID'] and owners[0]['startticks']==owners[1]['startticks']==out['child']['startticks']
def proc(pid):
 try:
  f=Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split();return dict(state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
for original in (out['owner'],out['child'],cpu['owner']):
 row=proc(original['PID']);assert row is None or row['startticks']!=original['startticks'] or row['state']=='Z'
live=[p for p in os.listdir('/proc') if p.isdigit() and (r:=proc(int(p))) and r['PGID']==out['child']['PGID'] and r['state']!='Z'];assert not live
manifest=binding['trainable_manifest'];assert len(manifest)==504 and len({v['name'] for v in manifest})==504 and sum(v['numel'] for v in manifest)==43646976
assert all(v['dtype']=='torch.float32' and ('.lora_A.' in v['name'] or '.lora_B.' in v['name']) and '.model.language_model.' in v['name'] for v in manifest)
assert binding['spec']['case']=='T04.sft' and binding['spec']['attention']=='eager' and binding['spec']['training']['gradient_checkpointing'] is False and binding['resolved_template_type']=='llava_onevision1_5' and binding['model_source_SHA256']=='bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198'
assert providers['HIP'] is None and providers['CUDA'] and providers['classes'] and all(p['SHA256']==binding['model_source_SHA256'] for p in providers['files'].values())
for name,pin in cpu['actual_sources'].items():
 p=Path(name);assert p.is_file() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
read('tests.txt');read('CPU.private.log')
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,artifacts=artifacts,original_outcome=out,binding=binding,providers=providers,owners_absent=True,owned_group_empty=True,reader_Torch_imported=False,GPU_GO=False,Model_GO=False)))
'''
with (D/'root-reader-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(remote).hexdigest(),read_only=True),f)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=90)
for n,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,claim=claim,request_id=request,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});c=v.get('original_outcome',{}).get('CPU_result',{});print(json.dumps(dict(transport_rc=r.returncode,independent_reader_passed=v.get('passed'),owners_absent=v.get('owners_absent'),trainable_tensors=c.get('native_trainable_tensors'),trainable_parameters=c.get('native_trainable_parameters'),inside_seconds=c.get('wall_seconds'),whole_seconds=v.get('original_outcome',{}).get('wall_seconds'),GPU_cards=0)))
