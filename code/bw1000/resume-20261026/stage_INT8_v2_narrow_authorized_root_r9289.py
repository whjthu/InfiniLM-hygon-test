from pathlib import Path
import json,hashlib,base64,subprocess,sys,datetime,ast
r=Path(__file__).resolve().parent;p=r/'Kimi-W2-INT8-register-v2-private-r113c/source-producer-index-r113v2b.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='90aa66306dfe61e3124afbdbfeb7f69744b57634a3ee7db783308654d442bf13';d=json.loads(p.read_bytes());assert d['remote_incoming_directory']=='Kimi-W2-INT8-register-v2-r113c';files={}
for n,h in d['runtime_files_SHA256'].items():
 q=r/n;b=q.read_bytes();assert q.suffix in {'.py','.sh','.json','.cpp','.cu'} and hashlib.sha256(b).hexdigest()==h
 if q.suffix=='.py':ast.parse(b.decode('utf-8-sig'))
 if q.suffix=='.sh':assert b'\r'not in b
 assert q.name not in files;files[q.name]={'SHA256':h,'data':base64.b64encode(b).decode()}
assert len(files)==11 and sum(len(x['data'])for x in files.values())<512000
remote="""import pathlib,json,base64,hashlib,os,uuid
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming').resolve();dst=base/'Kimi-W2-INT8-register-v2-r113c';dst.mkdir(exist_ok=True);assert dst.resolve().parent==base
files=PAYLOAD;out={'stage_only':True,'executed_code':False,'GPU_submitted':False,'files':{}}
for name,v in files.items():
 assert pathlib.PurePosixPath(name).name==name and pathlib.Path(name).suffix in {'.py','.sh','.json','.cpp','.cu'}
 b=base64.b64decode(v['data']);assert hashlib.sha256(b).hexdigest()==v['SHA256'];p=dst/name;assert p.resolve().parent==dst.resolve()
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256'];state='reused_exact'
 else:
  tmp=dst/('.stage-'+uuid.uuid4().hex);tmp.write_bytes(b)
  try:os.link(tmp,p);state='created'
  except FileExistsError:assert hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256'];state='concurrent_exact'
  finally:tmp.unlink()
 out['files'][name]={'SHA256':v['SHA256'],'bytes':len(b),'state':state}
print(json.dumps(out))
""".replace('PAYLOAD',repr(files))
a=['ssh','-T','-p','65032','-i',str(Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -'];z=subprocess.run(a,input=remote,text=True,capture_output=True,timeout=180);assert z.returncode==0,z.stderr;o=json.loads(z.stdout);o['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();o['authorization_scope']='Human call439 allows current/future necessary project Python/Shell/C++/headers/designJSON uploads to this exactcluster incoming; no key/modelweights/rawactivations included';q=r/'Kimi-W2-INT8-v2-narrow-stage-root-r9289.json';assert not q.exists();q.write_bytes((json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode());print('STAGED',len(o['files']),'files; no execution/GPU',q.name)
