import pathlib,json,hashlib,subprocess,shlex,ast
w=pathlib.Path(__file__).parent;p=w/'Deep-Cast-Model-private-r9060';m=json.loads((p/'Deep-Cast-Model-manifest-r9060.json').read_text());names=list(m['incoming_source_SHA256'])+['Deep-Cast-Model-manifest-r9060.json'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in names:
 if n.endswith('.py'):ast.parse((p/n).read_text())
 if n.endswith('.sh'):assert b'\r'not in(p/n).read_bytes()
h={n:sha(p/n)for n in names};opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(w/'known_hosts')];host='aclsylqidf@zzeshell.scnet.cn';inc='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
subprocess.run(['scp','-q','-P','65032',*opts,*[str(p/n)for n in names],host+':'+inc],check=True,timeout=120)
c='import pathlib,hashlib,runpy,subprocess;r=pathlib.Path('+repr(inc)+');h='+repr(h)+';assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in h.items());[subprocess.run(["bash","-n",str(r/n)],check=True)for n in h if n.endswith(".sh")];runpy.run_path(str(r/"launch_Deep_Cast_frontend_CPU_r9060.py"),run_name="__main__")'
z=subprocess.run(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],capture_output=True,text=True,timeout=100);assert z.returncode==0,z.stderr;print(z.stdout)
