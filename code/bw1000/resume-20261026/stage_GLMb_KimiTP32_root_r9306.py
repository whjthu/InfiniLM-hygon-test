from pathlib import Path
import hashlib,json,base64,subprocess,ast,datetime
r=Path.cwd();v=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={}
allowed={'.py','.sh','.cpp','.hpp','.h','.inc','.json','.md'}
def add(p,d,h=None):
    p=Path(p);b=p.read_bytes();actual=sha(p)
    if h:assert actual==h,(p,h)
    assert p.suffix in allowed and not Path(d).is_absolute() and '..'not in Path(d).parts
    if p.suffix=='.py':ast.parse(b.decode('utf-8-sig'))
    if p.suffix=='.sh':assert b'\r'not in b
    assert d not in files;files[d]={'SHA256':actual,'data':base64.b64encode(b).decode()}
g=v/'glm-m15-component-r121b'
for mn,mh in [('manifest-r121b.json','e6238e8d119ed8ad7e9beb247b461464b719b0ea2ab0edc0d2cc1f2d6d32ae56'),('GPU-manifest-r121b.json','f551cfa760f88b761f4012e664fd9a52ca641605dbf9fdf19e0b6aca7fb8bd44')]:
    mp=g/mn;add(mp,'GLM-M15-component-r121b/'+mn,mh)
    for n,h in json.loads(mp.read_bytes())['source_SHA256'].items():add(g/n,'GLM-M15-component-r121b/'+n,h)
k=v/'Kimi-TP32-I96-MoE-private-r120';mp=k/'source-manifest-r120.json';add(mp,'Kimi-TP32-I96-MoE-r120/'+mp.name,'d8c1491ba823c70e6f07b3a503b1a3c2cee63e9be42325379dd9d604e002cc39')
for n,h in json.loads(mp.read_bytes())['package_source_SHA256'].items():add(k/n,'Kimi-TP32-I96-MoE-r120/'+n,h)
e=v/'Kimi-TP32-I96-CPU-entry-private-r120b'
for n,d,h in [('CPU_inside_r120b.sh','Kimi-TP32-I96-MoE-r120/CPU_inside_r120b.sh','1dc11cd1e0af2c50edd940966f18c3f8336f649627b721f9974506cf56ac9fe6'),('CPU-entry-manifest-r120b.json','Kimi-TP32-I96-MoE-r120/CPU-entry-manifest-r120b.json','872ab304a2ef5424af46b8ef5b05942a8c0cb7f1411ce87f23f99a4ba0f4ab2d'),('launch_CPU_r120b.py','launch_CPU_r120b.py','acc8ed312ea54fd044df89160d8c67d15e7e709eb53741aa4cb888d3b7af5105')]:add(e/n,d,h)
assert len(files)==24 and sum(len(x['data'])for x in files.values())<500000
code='''import pathlib,json,base64,hashlib,os,uuid
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming').resolve();files=PAYLOAD;out={'stage_only':True,'executed_code':False,'CPU_submitted':False,'GPU_submitted':False,'files':{}}
for rel,v in files.items():
 pp=pathlib.PurePosixPath(rel);assert len(pp.parts)in[1,2]and '..'not in pp.parts and not pp.is_absolute()
 assert pp.parts[0]in['GLM-M15-component-r121b','Kimi-TP32-I96-MoE-r120','launch_CPU_r120b.py']
 p=base/rel;p.parent.mkdir(exist_ok=True);assert p.resolve().is_relative_to(base)and p.suffix in {'.py','.sh','.cpp','.hpp','.h','.inc','.json','.md'}
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
o=json.loads(z.stdout);o['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();o['scope']='Authorized necessary project source/headers/designJSON and checksum-required source review metadata only; no keys/weights/activations/binaries; staging never executes.'
p=r/'GLMb-KimiTP32-stage-root-r9306.json';assert not p.exists();p.write_bytes((json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode());print('STAGED',len(files),'NO_EXECUTION',p.name)
