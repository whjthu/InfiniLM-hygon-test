from pathlib import Path
import hashlib,json,subprocess,datetime
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='longpair-20261102-d';files=[]
paths=[R/'results'/('Kimi-long-pair-r347-'+run+'-watch.json')]+[R/'results'/('Kimi-long-pair-r346-'+run+'-node%d.json'%n)for n in range(4)]+[R/'results'/('Kimi-long-draft-r334-'+run+'.json')]
for p in paths:
 row=dict(path=str(p),exists=p.exists())
 if p.exists():
  raw=p.read_bytes();r=json.loads(raw);row.update(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
  if 'watch'in p.name:row['report']=r
  elif 'draft'in p.name:row['report']=r
  else:
   keep=['source_manifest_SHA256','completed','phase','node','actual_step','Native_SHA256','capture_port_reservation','create_seconds','loaded_keys','loading_elapsed_seconds','loading_seconds','error','traceback','first_capture_validation','draft_wire_source_binding','all8_owned_workers_shutdown_completed','cleanup_errors']
   row['compact_report']={k:r[k]for k in keep if k in r}
   if 'pair_report'in r:
    pair=r['pair_report'];row['compact_report']['pair_report']={k:v for k,v in pair.items()if k!='requests'}
    row['compact_report']['requests']=[{k:v for k,v in x.items()if k not in ['generated_ids','emitter_events','Native_calls','cycles']}for x in pair.get('requests',[])]
 files.append(row)
raw=subprocess.check_output(['sacct','-S','2026-11-01','-j','910380,910371,910407','-n','-P','-o','JobIDRaw,JobName%140,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30)
logs=[]
w=files[0].get('report',{})
logpaths=[]
for kind in ['target','draft','keeper']:
 label='Kimi-long-'+kind+'-r347-'+run
 logpaths.append(R/'logs'/(label+'-srun-client.log'))
 step=w.get('actual_'+kind)
 if kind=='keeper':step='910407.719'
 if step:
  names=[label+'-'+step+'-%d.log'%rank for rank in range(4)]if kind=='target'else[label+'-'+step+('-0.log'if kind=='draft'else'.log')]
  logpaths.extend(R/'logs'/n for n in names)
for p in logpaths:
 if not p.exists():continue
 size=p.stat().st_size
 with p.open('rb')as f:f.seek(max(0,size-5000));tail=f.read().decode('utf8','replace')
 logs.append(dict(path=str(p),bytes=size,tail=tail,tail_only=True,full_SHA256_not_computed=True))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,files=files,accounting='\n'.join(x for x in raw.splitlines()if run in x),logs=logs)))
