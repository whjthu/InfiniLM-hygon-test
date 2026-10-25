import pathlib,json,hashlib,subprocess,shlex,ast
r=pathlib.Path.cwd()/'Deep-packed-batch-private-r8985';names=['Deep_latent_maps_r8946.py', 'Deep_packed_batch_component_r8985.py', 'Deep_packed_batch_component_r8985_inside.sh', 'launch_Deep_packed_batch_component_r8985.py', 'verify_Deep_packed_batch_component_r8985.py', 'Deep-packed-batch-manifest-r8985.json'];h={n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in names}
for n in names:
 if n.endswith('.py'):ast.parse((r/n).read_text())
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r.parent/'known_hosts')];host='aclsylqidf@zzeshell.scnet.cn';incoming='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
def ssh(c):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],text=True,timeout=160)
c='import pathlib,json,hashlib;r=pathlib.Path('+repr(incoming)+');h='+repr(h)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))';old=json.loads(ssh(c));assert all(v is None or v==h[n]for n,v in old.items());missing=[n for n in names if old[n]is None]
if missing:subprocess.run(['scp','-q','-P','65032',*opts,*[str(r/n)for n in missing],host+':'+incoming],check=True,timeout=150)
assert json.loads(ssh(c))==h
accounting=ssh("import subprocess,json;print(json.dumps({'CPU227':subprocess.check_output(['sacct','-S','2026-10-24','-j','910407.227','-n','-P','-o','JobID,State,ExitCode,Elapsed'],text=True,timeout=30)}))")
(r/'Deep-packed-batch-staged-r8985.json').write_text(json.dumps({'source_SHA256':h,'GPU':'NOT_LAUNCHED_ROOT_CONDITIONAL_GO','CPU_accounting':json.loads(accounting)},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps({'staged':names,'GPU':'NOT_LAUNCHED','CPU_accounting':json.loads(accounting)}))
