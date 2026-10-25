import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/Kimi-93layer-static-budget-r9152.json';o={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910406','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30).splitlines()if 'Kimi-93layer-budget-r9152'in x]}
if p.exists():
 d=json.loads(p.read_bytes());o['report_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();o['report']={k:d.get(k)for k in ['completed','served_layers','config_SHA256','index_SHA256','partition_header_SHA256','PP_ranges_by_size','header_only_not_GPU_capacity','model_host_view','header_bytes_read','metadata_keys','topologies','method']}
else:
 o['logtails']=[{'file':p.name,'tail':p.read_text(errors='replace')[-2200:]}for p in(r/'logs').glob('Kimi-93layer-budget-r9152-*.log')]
print(json.dumps(o))
