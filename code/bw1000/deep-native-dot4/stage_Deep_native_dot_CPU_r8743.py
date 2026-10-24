import pathlib,json,hashlib,subprocess,shlex,ast
r=pathlib.Path(__file__).resolve().parent;files=['deep_native_dot_adapter_r8742.cpp', 'deep_routed_sdot4_forward_r8742.hpp', 'Deep-native-dot-source-r8743.json', 'build_Deep_native_dot_r8743.py', 'build_Deep_native_dot_r8743_inside.sh', 'launch_Deep_native_dot_CPU_r8743.py'];host='aclsylqidf@zzeshell.scnet.cn';incoming='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
for n in files:
 p=r/n
 if p.suffix in['.py','.sh']:p.write_text(p.read_text(encoding='utf-8-sig'),encoding='utf-8',newline='\n')
 if p.suffix=='.py':ast.parse(p.read_text())
h={n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in files};opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts')]
def ssh(c):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],text=True,timeout=160)
c='import pathlib,hashlib,json;r=pathlib.Path('+repr(incoming)+');h='+repr(h)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))'
old=json.loads(ssh(c));assert all(v is None or v==h[n]for n,v in old.items())
missing=[n for n,v in old.items()if v is None]
if missing:subprocess.run(['scp','-q','-P','65032',*opts,*[str(r/n)for n in missing],host+':'+incoming],check=True,timeout=150)
assert json.loads(ssh(c))==h
p=r/'Deep-native-dot-CPU-local-intent-r8743.json'
with p.open('x')as f:json.dump(dict(source_SHA256=h,scope='CPU-only one authorized entry'),f)
print(ssh('import runpy;runpy.run_path('+repr(incoming+'launch_Deep_native_dot_CPU_r8743.py')+',run_name="__main__")'))
