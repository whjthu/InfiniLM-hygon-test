from pathlib import Path
import json,hashlib,base64,subprocess,ast,datetime
r=Path.cwd();v=Path(__file__).resolve().parent;src=v/'glm-m15-component-r121'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mp=src/'manifest-r121.json';assert sha(mp)=='d14a99ebc95f9d81318830b3cb964720a7562daf74723026e778f81011926646'
m=json.loads(mp.read_bytes());names=dict(m['source_SHA256']);names[mp.name]=sha(mp);names['launch_CPU_only_r121.py']=sha(src/'launch_CPU_only_r121.py')
files={};allowed={'.py','.sh','.cpp','.hpp','.h','.inc','.json'}
for n,h in names.items():
    p=src/n;data=p.read_bytes();assert p.name==n and p.suffix in allowed and sha(p)==h
    if p.suffix=='.py':ast.parse(data.decode('utf-8-sig'))
    if p.suffix=='.sh':assert b'\r' not in data
    files[n]={'SHA256':h,'data':base64.b64encode(data).decode()}
assert len(files)==10 and sum(len(x['data'])for x in files.values())<400000
code='''import pathlib,json,base64,hashlib,os,uuid
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming').resolve();dst=base/'GLM-M15-component-r121';dst.mkdir(exist_ok=True);assert dst.resolve().parent==base
files=PAYLOAD;out={'stage_only':True,'executed_code':False,'GPU_submitted':False,'CPU_submitted':False,'files':{}}
for name,v in files.items():
 assert pathlib.PurePosixPath(name).name==name and pathlib.Path(name).suffix in {'.py','.sh','.cpp','.hpp','.h','.inc','.json'}
 b=base64.b64decode(v['data']);assert hashlib.sha256(b).hexdigest()==v['SHA256'];p=dst/name;assert p.resolve().parent==dst.resolve()
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256'];state='reused_exact'
 else:
  tmp=dst/('.stage-'+uuid.uuid4().hex);tmp.write_bytes(b)
  try:os.link(tmp,p);state='created'
  except FileExistsError:assert hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256'];state='concurrent_exact'
  finally:tmp.unlink()
 out['files'][name]={'SHA256':v['SHA256'],'bytes':len(b),'state':state}
print(json.dumps(out))
'''.replace('PAYLOAD',repr(files))
args=['ssh','-T','-p','65032','-i',str(Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -']
z=subprocess.run(args,input=code,text=True,capture_output=True,timeout=180);assert z.returncode==0,z.stderr
o=json.loads(z.stdout);o['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();o['manifest_SHA256']=sha(mp)
o['authorization_scope']='Human explicitly allows current/future necessary project Python/Shell/C++/headers/designJSON upload to this exact cluster incoming. No keys/weights/rawactivations/binaries; this action only stages, never executes.'
p=r/'GLM-M15-r121-stage-root-r9304.json';assert not p.exists();p.write_bytes((json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode());print('STAGED',len(files),'NO_EXECUTION',p.name)
