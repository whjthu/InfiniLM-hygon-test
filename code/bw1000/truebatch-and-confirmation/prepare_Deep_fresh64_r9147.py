from pathlib import Path
import json,hashlib,random,datetime,ast,base64,subprocess
r=Path.cwd();d=r/'Deep-B32-confirmation-private-r9147';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=r/'gsm8k-test-source-r9100.jsonl';assert sha(raw)=='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14';rows=[json.loads(x)for x in raw.read_text().splitlines()if x];old={x['source_index']for x in json.loads((r/'quality-bench-gsm8k-r9100.json').read_bytes())['cases']};order=list(range(len(rows)));random.Random(20261027).shuffle(order);selected=[i for i in order if i not in old][:64];assert len(selected)==64 and not set(selected)&old
q={'suite':'gsm8k-official-test-fresh64-fixed-random-pilot','source_SHA256':sha(raw),'source_revision':'3101c7d5072418e28b9008a6636bde82a006892c','source_count':len(rows),'seed':20261027,'selection':'Shuffle allindices with seed; exclude prior fixed32; takefirst64, noanswer orlengthfilter','pilot_not_noninferiority_proof':True,'cases':[{'id':'gsm8k-test-'+str(i),'source_index':i,'question':rows[i]['question'],'gold_answer':rows[i]['answer'],'gold_numeric':rows[i]['answer'].rsplit('####',1)[1].strip().replace(',','')}for i in selected]};data=d/'quality-bench-gsm8k-fresh64-r9147.json';data.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode())
s=(r/'Deep-quality-private-r9101/prepare_Deep_quality_tokenizer_r9101c.py').read_text().replace('quality-bench-gsm8k-r9100.json',data.name).replace('Deep-quality-tokenized-r9101c.json','Deep-quality-tokenized-r9147.json').replace('acc36a387858099f277d9440dcd48fdb4140127f509f0329279b2e7bd3ab3b6b',sha(data)).replace("len(q['cases'])==32","len(q['cases'])==64")
(d/'prepare_Deep_quality_tokenizer_r9147.py').write_bytes(s.encode());ast.parse(s)
s=(r/'Deep-quality-private-r9101/Deep_quality_tokenizer_CPU_r9101c_inside.sh').read_text().replace('r9101c','r9147');(d/'Deep_quality_tokenizer_CPU_r9147_inside.sh').write_bytes(s.encode());assert b'\r'not in(d/'Deep_quality_tokenizer_CPU_r9147_inside.sh').read_bytes()
files={p.name:base64.b64encode(p.read_bytes()).decode()for p in d.iterdir()if p.is_file()};remote='''import pathlib,json,base64,hashlib,subprocess,os,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming';payload=PAYLOAD;staged=[]
for n,v in payload.items():
 p=inc/n;b=base64.b64decode(v);h=hashlib.sha256(b).hexdigest()
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 else:p.write_bytes(b)
 staged.append({'name':n,'SHA256':h,'bytes':len(b)})
subprocess.run(['bash','-n',str(inc/'Deep_quality_tokenizer_CPU_r9147_inside.sh')],check=True)
parent='910406';node='a14r2n19';state=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=20);assert 'JobState=RUNNING'in state
record=r/'results/Deep-tokenizer-r9147-launch.json';assert not record.exists();args=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=8192M','-t','00:03:00','--job-name=Deep-tokenizer-r9147','--output='+str(r/'logs/Deep-tokenizer-r9147-%J.log'),'env','WAVE_INSIDE_R2323=Deep_quality_tokenizer_CPU_r9147_inside.sh','bash',str(inc/'resource_wave_local_r3097_outer.sh'),'probe']
env={k:v for k,v in os.environ.items()if not(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ('SLURM_CONF','SLURM_CONF_SERVER'))and k not in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}
with record.open('x')as f:json.dump({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'staged':staged,'args':args,'CPU_only':True,'GPU_initialized':False},f,indent=2)
with(r/'logs/Deep-tokenizer-r9147-controller.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
print(json.dumps({'controller_PID':p.pid,'record':str(record),'parent':parent,'node':node,'CPU_only':True,'staged':staged}))
'''.replace('PAYLOAD',repr(files))
(r/'stage_Deep_tokenizer_r9147_remote.py').write_bytes(remote.encode());print(json.dumps({'files':len(files),'fresh64dataset_SHA256':sha(data),'old32excluded':True,'GPU':False}))
