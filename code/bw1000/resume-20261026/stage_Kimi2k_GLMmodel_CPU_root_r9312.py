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
km=r/'Kimi-ctx2048-CPU-entry-private-r123b/CPU-stage-file-manifest-r123b.json';assert sha(km)=='a238d32bb43950fc2b0799a39111e3b5ac7b5a704e03b9159e7cc03de213d131';k=json.loads(km.read_bytes())
for n,h in k['files_SHA256'].items():add(r/k['local_sources'][n],'Kimi-ctx2048-r123/'+n,h)
add(km,'Kimi-ctx2048-r123/'+km.name)
g=v/'glm-m15-oproj-model-r125';gm=g/'manifest-r125.json';assert sha(gm)=='c874180c3109a37969074dfcb698d191d077849781ad5e40bf1d61a495c3b41f';m=json.loads(gm.read_bytes())
for n,h in m['source_SHA256'].items():add(g/n,'GLM-M15-oproj-Model-r125/'+n,h)
add(gm,'GLM-M15-oproj-Model-r125/'+gm.name)
assert len(files)==86 and sum(len(x['data'])for x in files.values())<2000000
code='''import pathlib,json,base64,hashlib,os,uuid
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming').resolve();files=PAYLOAD;out={'stage_only':True,'executed_code':False,'CPU_submitted':False,'GPU_submitted':False,'files':{}}
for rel,v in files.items():
 pp=pathlib.PurePosixPath(rel);assert len(pp.parts)>=2 and pp.parts[0]in['Kimi-ctx2048-r123','GLM-M15-oproj-Model-r125']and '..'not in pp.parts and not pp.is_absolute()
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
p=r/'Kimi2k-GLMmodel-CPU-stage-root-r9312.json';assert not p.exists();p.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print('STAGED',len(files),'NO_EXECUTION',p.name)
