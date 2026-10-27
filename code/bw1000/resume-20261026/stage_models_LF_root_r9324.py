from pathlib import Path
import json,hashlib,base64,ast,subprocess,datetime
r=Path.cwd();v=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={}
def add(p,d,h=None):
    p=Path(p);b=p.read_bytes();actual=sha(p)
    if h:assert actual==h,(p,h)
    assert p.suffix in {'.py','.sh','.cpp','.hpp','.h','.inc','.json'}and '..'not in Path(d).parts
    if p.suffix=='.py':ast.parse(b.decode('utf-8-sig'))
    if p.suffix=='.sh':assert b'\r'not in b,p
    assert d not in files;files[d]={'SHA256':actual,'data':base64.b64encode(b).decode()}
folder=r/'Kimi-ctx2k-Model-controller-private-r127';mp=folder/'stage-source-manifest-r127.json';assert sha(mp)=='e250c20b74c3eb0b9ac05dd2d4bff6e4ed3eb7072b26e5e7cd19d052c078c2cf';m=json.loads(mp.read_bytes())
for n,h in m['files_SHA256'].items():add(folder/n,'Kimi-ctx2k-Model-r127/'+n,h)
d=r/'glm-m15-model-entry-r129b';mp=d/'GPU-manifest.json';assert sha(mp)=='b2045541a6a75a6820860d7708f387a86cee5734e4846b18f248888bd4cfaa05';m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():add(d/n,'GLM-M15-Model-r129/'+n,h)
add(mp,'GLM-M15-Model-r129/'+mp.name)
assert len(files)==91
code='''import pathlib,json,base64,hashlib,os,uuid
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming').resolve();files=PAYLOAD;out={'stage_only':True,'executed_code':False,'CPU_submitted':False,'GPU_submitted':False,'files':{}}
for rel,v in files.items():
 pp=pathlib.PurePosixPath(rel);assert len(pp.parts)>=2 and pp.parts[0]in['Kimi-ctx2k-Model-r127','GLM-M15-Model-r129']and '..'not in pp.parts and not pp.is_absolute()
 p=base/rel;assert p.resolve().is_relative_to(base/pp.parts[0])and p.suffix in {'.py','.sh','.cpp','.hpp','.h','.inc','.json'};p.parent.mkdir(parents=True,exist_ok=True);assert p.resolve().is_relative_to(base/pp.parts[0])
 b=base64.b64decode(v['data']);assert hashlib.sha256(b).hexdigest()==v['SHA256']
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256'];state='reused_exact'
 else:
  t=p.parent/('.stage-'+uuid.uuid4().hex);t.write_bytes(b)
  try:os.link(t,p);state='created'
  except FileExistsError:assert hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256'];state='concurrent_exact'
  finally:t.unlink()
 out['files'][rel]={'SHA256':v['SHA256'],'bytes':len(b),'state':state}
print(json.dumps(out))
'''.replace('PAYLOAD',repr(files))
args=['ssh','-T','-p','65032','-i',str(Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -']
z=subprocess.run(args,input=code,text=True,capture_output=True,timeout=180);assert z.returncode==0,z.stderr
q=json.loads(z.stdout);q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();q['scope']='Onlyauthorizednecessaryprojectcode/headers/designJSON; noexe/weights/activations/socketframe/keys. Stageonly, CPUinvocationseparate.'
p=r/'models-stage-root-r9323.json';assert not p.exists();p.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print('STAGED',len(files),'NO_EXECUTION',p.name)
