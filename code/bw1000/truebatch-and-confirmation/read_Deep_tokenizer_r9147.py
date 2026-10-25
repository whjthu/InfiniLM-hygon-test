import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/Deep-quality-tokenized-r9147.json';out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sacct':[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910406','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%80'],text=True,timeout=30).splitlines()if 'Deep-tokenizer-r9147'in x]}
if p.exists():
 d=json.loads(p.read_bytes());out['receipt']={k:d.get(k)for k in ['producer','accepted_count','skipped','input_lengths','shot_count','CPU_seconds','source_SHA256','template_provenance','all_unique_IDs','no_GPU_native_weights']};out['tokenized_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
else:
 files=list((r/'logs').glob('Deep-tokenizer-r9147-*.log'));out['tail']=[{'file':p.name,'tail':p.read_text(errors='replace')[-2500:]}for p in files]
print(json.dumps(out))
