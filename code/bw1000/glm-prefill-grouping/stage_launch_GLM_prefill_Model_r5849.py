"""One batch SCP, exact SHA/lease, then sole already-authorized entry once."""
import hashlib,json,pathlib,shlex,subprocess
r=pathlib.Path(__file__).resolve().parent;out=r/'r5849-GLM-prefill-Model-launch.json';assert not out.exists()
ready=json.loads((r/'GLM-prefill-source-ready-r5849.json').read_text());files=dict(ready['files_SHA256']);files['GLM-prefill-source-ready-r5849.json']=hashlib.sha256((r/'GLM-prefill-source-ready-r5849.json').read_bytes()).hexdigest()
host='aclsylqidf@zzeshell.scnet.cn';remote='/public/home/aclsylqidf/hygon-inference-refactor-20261014';incoming=remote+'/incoming/'
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+str(r/'known_hosts')]
def ssh(code):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(code)],text=True,timeout=240)
code='import pathlib,json,hashlib;r=pathlib.Path('+repr(incoming)+');h='+repr(files)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))'
old=json.loads(ssh(code));assert all(h is None or h==files[n]for n,h in old.items()),old
missing=[n for n in files if old[n]is None]
for n,h in files.items():
 p=r/n;assert p.parent==r and p.is_file()and hashlib.sha256(p.read_bytes()).hexdigest()==h
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
if missing:subprocess.run(['scp','-q','-P','65032',*opts,*[str(r/n)for n in missing],host+':'+incoming],check=True,timeout=150)
assert json.loads(ssh(code))==files
lease=json.loads(ssh("import subprocess,json,re;p=subprocess.check_output(['scontrol','show','job','910376','-o'],text=True);assert 'JobState=RUNNING'in p;n=re.search(r'(?<!\\S)NodeList=(\\S+)',p).group(1);hosts=subprocess.check_output(['scontrol','show','hostnames',n],text=True).splitlines();assert set(['a02r4n12','a02r4n13'])<=set(hosts);print(json.dumps({'parent':'910376','nodes':hosts,'RUNNING':True}))"))
print('REMOTE_SHA_AND_LEASE_VERIFIED',len(files),flush=True)
# Unknown SSH outcome after this one invocation must be resolved by reading the
# unique remote launch/watch files. Never call this helper again to retry it.
result=json.loads(ssh('import runpy;runpy.run_path('+repr(incoming+ready['entry'])+',run_name="__main__")'))
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'saved':out.name,**{k:result.get(k)for k in ('run','pid','watch_pid','utc')}}))
