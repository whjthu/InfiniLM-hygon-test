from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-gemma-A100-first-updates-v1';assert not (D/'failure-terminal-read-intent.json').exists();t=json.loads((D/'transport-outcome.json').read_bytes());(D/'failure-terminal-read-intent.json').write_bytes((json.dumps(dict(claim=t['claim'],request_id=t['request_id'],scope='Original failed native reference and independent post; no model/reader replay'))+'\n').encode())
remote=r'''from pathlib import Path
import datetime,hashlib,json,os,re,socket,subprocess,sys
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-A100-first-updates-v1');out={}
for name in ['installed.json','supervisor-intent.json','gpu-intent.json','child-owner.json','outcome.json','outer-result.json']:
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=300000;out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
assert out['outcome.json']['value']['claim']==CLAIM and out['outcome.json']['value']['request_id']==REQUEST
p=W/'native.private.log';size=p.stat().st_size
with p.open('rb') as f:f.seek(max(0,size-250000));text=f.read().decode(errors='replace')
known=re.findall(r'(?:ValueError|RuntimeError|AssertionError|AttributeError): ((?:native Trainer argument differs|native PEFT configuration differs): [A-Za-z0-9_]+|complete native trainable names,shapes,dtypes or counts differ|expanded native LoRA targets differ|native effective warmup differs|native frozen boundary differs: [A-Za-z0-9_.]+)',text)
errors=re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)):',text);frames=[dict(file=Path(p).name,line=int(n),function=fn) for p,n,fn in re.findall(r'File "([^"]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)',text)]
ranks=[]
for rank in range(8):
 folder=W/'capture'/('rank-%02d'%rank);r=dict(rank=rank,planned_updates=2,planned_micro_forwards=8,planned_artifacts=25,planned_native_global_step=25,capture_directory_exists=folder.exists())
 for name in ['owner.json','progress.json','failure.json','receipt.json','native25-complete.json']:
  p=folder/name
  if p.is_file():
   raw=p.read_bytes();assert len(raw)<200000;r[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 ranks.append(r)
args={};p=W/'train/args.json'
if p.is_file():
 raw=p.read_bytes();value=json.loads(raw);args={k:value.get(k) for k in ['warmup_ratio','warmup_steps','max_steps','learning_rate','gradient_accumulation_steps','max_grad_norm','gradient_checkpointing','bf16','torch_dtype','lora_rank','lora_alpha','lora_dropout']}
reg=out['outcome.json']['value']['owners'];helper=W/'src/owned_processes.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER;sys.path.insert(0,str(W/'src'));from owned_processes import refresh
remaining=refresh({int(k):v for k,v in reg.items()},W);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=12).strip().splitlines()
try:
 with socket.socket() as s:s.bind(('127.0.0.1',29891))
 port=True
except OSError:port=False
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_files=out,native_log_bytes=size,native_log_known_errors=sorted(set(known)),native_log_error_types=sorted(set(errors)),last_error_frames=frames[-16:],rank_denominators=ranks,original_args_whitelist=args,independent_post=dict(owned_processes_remaining=len(remaining),compute_apps=apps,master_port_bindable=port,GPU_context_created_by_reader=False))))
'''
helper=hashlib.sha256((D/'package/owned_processes.py').read_bytes()).hexdigest();code='CLAIM='+repr(t['claim'])+'\nREQUEST='+repr(t['request_id'])+'\nHELPER='+repr(helper)+'\n'+remote;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];r=subprocess.run(ssh,input=code.encode(),capture_output=True,timeout=35);(D/'failure-terminal-stdout.private.json').write_bytes(r.stdout);(D/'failure-terminal-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0;result=json.loads(r.stdout);assert result['original_files']['outcome.json']['value']==t['value']['outcome.json']['value'];(D/'independent-original-failed-terminal.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());o=result['original_files']['outcome.json']['value'];print(json.dumps(dict(native_returncode=o['native_returncode'],keeper_passed=o['keeper_passed'],wall_seconds=o['wall_seconds'],cleanup=o['cleanup'],known_errors=result['native_log_known_errors'],error_types=result['native_log_error_types'],args=result['original_args_whitelist'],rank_capture_exists=[x['capture_directory_exists'] for x in result['rank_denominators']],post=result['independent_post'],frames=result['last_error_frames'])))
