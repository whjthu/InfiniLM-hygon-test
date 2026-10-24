import pathlib,json,hashlib,subprocess,shlex,ast
r=pathlib.Path.cwd();d=r/'Deep-activation-private-r8840c';sources=[p for p in d.iterdir()if p.is_file()];m={'source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},'scope':'CPU2hostTU+1privatekernelTU,noGPU/dlopen/SDK/weights'};(r/'Deep-activation-source-r8841c.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf8',newline='\n');files=sources+[r/n for n in ['Deep-activation-source-r8841c.json','build_Deep_activation_r8841c.py','build_Deep_activation_r8841c_inside.sh','launch_Deep_activation_CPU_r8841c.py']]
for p in files:
 if p.suffix in ['.py','.sh']:p.write_text(p.read_text(encoding='utf-8-sig'),encoding='utf8',newline='\n')
 if p.suffix=='.py':ast.parse(p.read_text())
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts')];host='aclsylqidf@zzeshell.scnet.cn';incoming='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/';names={p:('deep-activation-r8840c/'+p.name if p in sources else p.name)for p in files};h={names[p]:hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
def ssh(c):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],text=True,timeout=160)
c='import pathlib,json,hashlib;r=pathlib.Path('+repr(incoming)+');h='+repr(h)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))';old=json.loads(ssh(c));assert all(v is None or v==h[n]for n,v in old.items());missing=[p for p in files if old[names[p]]is None]
ssh('import pathlib;pathlib.Path('+repr(incoming+'deep-activation-r8840c')+').mkdir(exist_ok=True)')
for group in [sources,[p for p in files if p not in sources]]:
 selected=[p for p in group if p in missing]
 if selected:subprocess.run(['scp','-q','-P','65032',*opts,*map(str,selected),host+':'+incoming+('deep-activation-r8840c/' if group is sources else '')],check=True,timeout=150)
assert json.loads(ssh(c))==h
(r/'Deep-activation-CPU-local-intent-r8841c.json').write_text(json.dumps({'source_SHA256':h,'scope':'2host1kernelCPU/noGPU/SDK/dlopen/weights'},indent=2)+'\n',encoding='utf8',newline='\n');print(ssh('import runpy;runpy.run_path('+repr(incoming+'launch_Deep_activation_CPU_r8841c.py')+',run_name="__main__")'))
