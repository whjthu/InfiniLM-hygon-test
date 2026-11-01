import pathlib,subprocess,sys
L=pathlib.Path(__file__).resolve().parents[4]
cmd=['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(L/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -']
z=subprocess.run(cmd,input=pathlib.Path(sys.argv[1]).read_text(encoding='utf-8-sig'),capture_output=True,text=True,timeout=200)
if len(sys.argv)>2 and z.returncode==0:
 import json
 out=pathlib.Path(sys.argv[2]);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(json.loads(z.stdout),indent=2),encoding='utf8');print(str(out))
else:print(z.stdout)
print(z.stderr,file=sys.stderr);sys.exit(z.returncode)
