"""One SHA probe, one bulk scp for missing code, one final SHA probe.

Only user-authorized project source/JSON to incoming. Never launches a Model,
reads key bytes, uploads weights/raw activations, or modifies shared SDK files.
"""
import pathlib,json,hashlib,subprocess,shlex,sys,datetime
r=pathlib.Path(__file__).resolve().parent;package=r/'Kimi-speculative-AUX0-n01-package-r8475'
spec=json.loads((package/'Kimi-AUX0-runtime-manifest-r8475.json').read_text());expected={}
for name in list(spec['sources_SHA256'])+['Kimi-AUX0-runtime-manifest-r8475.json']:
    assert pathlib.PurePath(name).name==name and pathlib.Path(name).suffix in ('.py','.sh','.json')
    raw=(package/name).read_bytes();expected[name]=hashlib.sha256(raw).hexdigest()
    if name in spec['sources_SHA256']:assert expected[name]==spec['sources_SHA256'][name]
base=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+str(r/'known_hosts')]
host='aclsylqidf@zzeshell.scnet.cn';destination='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
def probe():
    code="import pathlib,json,hashlib\nr=pathlib.Path("+repr(destination)+")\nexpected="+repr(expected)+"\nactual={n:hashlib.sha256((r/n).read_bytes()).hexdigest()if(r/n).exists()else None for n in expected}\nprint(json.dumps(actual))"
    z=subprocess.run(['ssh','-T','-p','65032',*base,host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=120)
    assert z.returncode==0,z.stderr
    return json.loads(z.stdout)
before=probe();missing=[n for n,h in expected.items()if before[n]!=h]
if missing:subprocess.run(['scp','-q','-P','65032',*base,*[str(package/n)for n in missing],host+':'+destination],check=True,timeout=120)
after=probe();assert after==expected
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_SHA256=before,uploaded_once=missing,unchanged_skipped=[n for n in expected if n not in missing],final_SHA256=after,all_match=True,Model_submitted=False,scope='Authorized project code/JSON only; 2SSH probes plus <=1bulk scp, no tar/per-file handshake')
if len(sys.argv)>1:
    p=r/sys.argv[1];assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(out))
