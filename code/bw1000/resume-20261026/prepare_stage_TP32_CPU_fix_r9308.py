from pathlib import Path
import json,hashlib,base64,subprocess,ast,datetime
r=Path.cwd();v=Path(__file__).resolve().parent;out=v/'Kimi-TP32-I96-CPU-fix-r120c';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=v/'Kimi-TP32-I96-MoE-private-r120/CPU_build_r120.py';s=src.read_text(encoding='utf8')
assert s.count("assert '-Wl,--no-undefined' in cmd")==1
s=s.replace('import pathlib,json,hashlib,subprocess,shlex,time','import pathlib,json,hashlib,subprocess,shlex,time,os')
s=s.replace("b=R/'candidates/Kimi-TP32-I96-MoE-r120'","b=R/'candidates/Kimi-TP32-I96-MoE-r120c'")
s=s.replace("assert '-Wl,--no-undefined' in cmd","if '-Wl,--no-undefined' not in cmd:cmd.insert(cmd.index('-o'),'-Wl,--no-undefined')")
s=s.replace("(b/'CPU-build.json').write_text", "receipt['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');receipt['original_failed_receipt_SHA256']=sha(R/'candidates/Kimi-TP32-I96-MoE-r120/CPU-build.json');receipt['repair']='Addstrictundefinedlinkflag missingfromparentcommand; noC++orGPUmathchange';(b/'CPU-build.json').write_text")
cp=out/'CPU_build_r120c.py';cp.write_bytes(s.encode());ast.parse(s)
b=v/'Kimi-TP32-I96-CPU-entry-private-r120b'
s=(b/'CPU_inside_r120b.sh').read_text(encoding='utf8').replace('CPU_build_r120.py','CPU_build_r120c.py').replace('CPU-r120b-','CPU-r120c-');ep=out/'CPU_inside_r120c.sh';ep.write_bytes(s.encode());assert b'\r'not in ep.read_bytes()
s=(b/'launch_CPU_r120b.py').read_text(encoding='utf8').replace('r120b','r120c')
s=s.replace("assert sha(entry)==m['entry_SHA256']","assert sha(entry)==m['entry_SHA256'];assert sha(entry.parent/'CPU_build_r120c.py')==m['CPU_build_SHA256']")
lp=out/'launch_CPU_r120c.py';lp.write_bytes(s.encode());ast.parse(s)
note=json.loads((b/'CPU-entry-manifest-r120b.json').read_bytes());note.update(status='CPU_ONLY_LINK_FLAG_REPAIR_NO_CXX_CHANGE',entry_SHA256=sha(ep),launcher_SHA256=sha(lp),CPU_build_SHA256=sha(cp),original_CPU440_failure='Beforecompiler,commands[]: parentreusedlinkcommanddoesnotcontain--no-undefined. PreservefailedprivatecandidateandappendflaginnewCPUentry.',WAVE_INSIDE_R2323='Kimi-TP32-I96-MoE-r120/CPU_inside_r120c.sh');note['exact_stagefiles']=[{'remote':'incoming/Kimi-TP32-I96-MoE-r120/'+p.name,'local':p.name}for p in[cp,ep]]+[{'remote':'incoming/launch_CPU_r120c.py','local':lp.name}]
np=out/'CPU-entry-manifest-r120c.json';np.write_bytes((json.dumps(note,indent=2)+'\n').encode())
files={}
for p in[cp,ep,np,lp]:
    rel=p.name if p==lp else'Kimi-TP32-I96-MoE-r120/'+p.name
    files[rel]={'SHA256':sha(p),'data':base64.b64encode(p.read_bytes()).decode()}
code='''import pathlib,json,base64,hashlib,os,uuid
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming').resolve();out={'stage_only':True,'executed_code':False,'CPU_submitted':False,'GPU_submitted':False,'files':{}};files=PAYLOAD
for rel,v in files.items():
 p=base/rel;assert p.resolve().is_relative_to(base)and p.suffix in {'.py','.sh','.json'}and p.parent in [base,base/'Kimi-TP32-I96-MoE-r120'];b=base64.b64decode(v['data']);assert hashlib.sha256(b).hexdigest()==v['SHA256']
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
receipt=json.loads(z.stdout);receipt.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),launcher_SHA256=sha(lp),manifest_SHA256=sha(np));p=r/'TP32-CPU-fix-stage-root-r9308.json';assert not p.exists();p.write_bytes((json.dumps(receipt,indent=2)+'\n').encode());print('STAGED4_NO_EXECUTION',sha(lp),sha(np))
