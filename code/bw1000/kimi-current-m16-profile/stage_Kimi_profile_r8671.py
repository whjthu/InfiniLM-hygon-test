"""Normal batch authorized source upload; unique entry only after source+claim push."""
import pathlib,json,hashlib,shlex,subprocess,sys,datetime
r=pathlib.Path(__file__).resolve().parent
ready=json.loads((r/'Kimi-current-profile-source-ready-r8671.json').read_text());files=dict(ready['files_SHA256']);files['Kimi-current-profile-source-ready-r8671.json']=hashlib.sha256((r/'Kimi-current-profile-source-ready-r8671.json').read_bytes()).hexdigest()
host='aclsylqidf@zzeshell.scnet.cn';remote='/public/home/aclsylqidf/hygon-inference-refactor-20261014';incoming=remote+'/incoming/'
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+str(r/'known_hosts')]
def ssh(code):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(code)],text=True,timeout=240)
code='import pathlib,json,hashlib;r=pathlib.Path('+repr(incoming)+');h='+repr(files)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))'
old=json.loads(ssh(code));assert all(h is None or h==files[n]for n,h in old.items()),old
for n,h in files.items():
 p=r/n;assert p.parent==r and p.is_file()and hashlib.sha256(p.read_bytes()).hexdigest()==h
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
missing=[n for n in files if old[n]is None]
if missing:subprocess.run(['scp','-q','-P','65032',*opts,*[str(r/n)for n in missing],host+':'+incoming],check=True,timeout=180)
assert json.loads(ssh(code))==files
lease=json.loads(ssh("import subprocess,json,re;p=subprocess.check_output(['scontrol','show','job','910375','-o'],text=True);assert 'JobState=RUNNING'in p;n=re.search(r'(?<!\\S)NodeList=(\\S+)',p).group(1);hosts=subprocess.check_output(['scontrol','show','hostnames',n],text=True).splitlines();assert set(['a02r4n08','a02r4n09','a02r4n10','a02r4n11'])<=set(hosts);print(json.dumps({'parent':'910375','nodes':hosts,'RUNNING':True}))"))
print('BATCH_SHA_LEASE_VERIFIED',len(files),flush=True)
if sys.argv[1]=='upload':
 p=r/'r8671-Kimi-profile-upload.json';assert not p.exists();p.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_SHA256=files,lease=lease),indent=2));raise SystemExit
assert sys.argv[1]=='launch'
pub=json.loads((r/'Kimi-profile-publication-r8671.json').read_text());assert pub['pushed']and pub['branch']=='bw1000-experiments'
intent=r/'r8671-Kimi-profile-local-launch-intent.json'
with intent.open('x')as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),publication=pub['commit'],entry=ready['entry']),f)
# After this invocation starts, any SSH uncertainty is resolved by reading original remote launch/watch, NEVER retry.
result=json.loads(ssh('import runpy;runpy.run_path('+repr(incoming+ready['entry'])+',run_name="__main__")'))
out=r/'r8671-Kimi-profile-launch.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(saved=out.name,**{k:result.get(k)for k in ['run','state','nodes','controller_pid']})))
