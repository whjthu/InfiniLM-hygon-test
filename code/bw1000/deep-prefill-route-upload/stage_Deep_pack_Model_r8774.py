"""Authorized source-only bulk transfer, exact immutable hashes, sole reviewed entry."""
import pathlib,json,hashlib,subprocess,shlex,ast
r=pathlib.Path(__file__).parent;m=json.loads((r/'Deep-pack-Model-manifest-r8774.json').read_text());assert hashlib.sha256((r/'Deep-pack-Model-manifest-r8774.json').read_bytes()).hexdigest()=='013aba91c2fb08e6e88651de1f646b825f0dcd06104207e20c215639bcfde139'
h=dict(m['incoming_source_SHA256'],**m['existing_incoming_dependency_SHA256']);h['Deep-pack-Model-manifest-r8774.json']='013aba91c2fb08e6e88651de1f646b825f0dcd06104207e20c215639bcfde139'
for n,v in h.items():
 assert hashlib.sha256((r/n).read_bytes()).hexdigest()==v,n
 if n.endswith('.py'):ast.parse((r/n).read_text(encoding='utf-8-sig'))
host='aclsylqidf@zzeshell.scnet.cn';incoming='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts')]
def ssh(c):return subprocess.check_output(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],text=True,timeout=160)
c='import pathlib,hashlib,json;r=pathlib.Path('+repr(incoming)+');h='+repr(h)+';print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in h}))'
old=json.loads(ssh(c));assert all(v is None or v==h[n]for n,v in old.items()),old
missing=[n for n,v in old.items()if v is None]
if missing:subprocess.run(['scp','-q','-P','65032',*opts,*[str(r/n)for n in missing],host+':'+incoming],check=True,timeout=160)
assert json.loads(ssh(c))==h
with(r/'Deep-pack-Model-local-intent-r8774.json').open('x')as f:json.dump({'source_SHA256':h,'scope':'Single authorized finite component, not Model TPS'},f)
print(ssh('import runpy;runpy.run_path('+repr(incoming+'launch_Deep_pack_Model_r8774.py')+',run_name="__main__")'))
