import pathlib,hashlib,subprocess,shlex,json
r=pathlib.Path.cwd();m=json.loads((r/'Kimi-A8-bitfield-source-r8996.json').read_text());names=list(m['source_SHA256'])+['Kimi-A8-bitfield-source-r8996.json'];private=r/'Kimi-W13-A8-bitfield-private-r8996'
files=[(private/n if(private/n).is_file()else r/n)for n in names];h={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
opts=['-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(r/'known_hosts')];host='aclsylqidf@zzeshell.scnet.cn';inc='/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'
subprocess.run(['scp','-q','-P','65032',*opts,*map(str,files),host+':'+inc],check=True,timeout=120)
c='import pathlib,hashlib,runpy;r=pathlib.Path('+repr(inc)+');h='+repr(h)+';assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in h.items());runpy.run_path(str(r/"launch_Kimi_A8_bitfield_CPU_r8996.py"),run_name="__main__")'
z=subprocess.run(['ssh','-T','-p','65032',*opts,host,'python3 -c '+shlex.quote(c)],capture_output=True,text=True,timeout=100);assert z.returncode==0,z.stderr;print(z.stdout)
