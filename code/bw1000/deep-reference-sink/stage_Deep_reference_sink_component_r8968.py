import pathlib,json,hashlib,subprocess,shlex,ast
r=pathlib.Path.cwd();names=['Deep_latent_maps_r8946.py', 'Deep_reference_sink_component_r8968.py', 'Deep_reference_sink_component_r8968_inside.sh', 'launch_Deep_reference_sink_component_r8968.py', 'verify_Deep_reference_sink_component_r8968.py', 'Deep-reference-sink-manifest-r8968.json'];h={n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in names}
for n in names:
 if n.endswith('.py'):ast.parse((r/n).read_text())
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts')];host='aclsylqidf@zzeshell.scnet.cn';incoming='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
def ssh(c):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],text=True,timeout=160)
c='import pathlib,json,hashlib;r=pathlib.Path('+repr(incoming)+');h='+repr(h)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))';old=json.loads(ssh(c));assert all(v is None or v==h[n]for n,v in old.items());missing=[n for n in names if old[n]is None]
if missing:subprocess.run(['scp','-q','-P','65032',*opts,*[str(r/n)for n in missing],host+':'+incoming],check=True,timeout=150)
assert json.loads(ssh(c))==h
(r/'Deep-reference-sink-staged-r8968.json').write_bytes((json.dumps({'source_SHA256':h,'GPU':'NOT_LAUNCHED_PENDING_ROOT_GO'},indent=2)+'\n').encode());print(json.dumps({'staged':names,'GPU':'NOT_LAUNCHED'}))
