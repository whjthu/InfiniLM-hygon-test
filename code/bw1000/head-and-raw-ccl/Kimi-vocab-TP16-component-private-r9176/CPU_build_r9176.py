import pathlib,json,hashlib,subprocess,shlex,time,os
r=pathlib.Path('/refactor');s=r/'incoming/Kimi-vocab-TP16-component-r9176';b=r/'candidates/Kimi-vocab-TP16-component-r9176';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'host_TUs':1,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};start=time.monotonic()
try:
 parent=json.loads((r/'candidates/Kimi-speculative-batch-Model-r9122c/CPU-build.json').read_bytes());assert parent['completed']and sha(parent['native'])==parent['native_SHA256']=='9647453d13f5e56b81fda3a6ec699f5329a83157a294e7e8e2dcffeeb4775e61'
 for p,h in parent['all_linked_object_SHA256'].items():assert sha(p)==h,p
 flags=[x for x in parent['commands'][0]if x.startswith(('-I','-D'))];base=['g++','-std=c++17','-O2','-fPIC','-pthread','-I'+str(s),*flags,'-I/opt/dtk/cuda/cuda/include'];cmd=[*base,'-MD','-MF',str(b/'component.d'),'-c',str(s/'head_component_r9176.cpp'),'-o',str(b/'component.o')];q['commands']=[cmd];z=subprocess.run(cmd,capture_output=True,text=True,timeout=220);(b/'compile.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-8000:]
 cmd=[*base,'-M','-MF',str(b/'fresh.d'),str(s/'head_component_r9176.cpp')];z=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert z.returncode==0,z.stderr
 deps=lambda p:sorted(set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])));md=deps(b/'component.d');fresh=deps(b/'fresh.d');assert md==fresh
 link=parent['commands'][-1][:];link.insert(1,str(b/'component.o'));native=b/'libKimi_vocab_TP16_component_r9176.so';link[link.index('-o')+1]=str(native)
 for j,a in enumerate(link):
  if a.startswith('-Wl,-Map'):link[j]='-Wl,-Map,'+str(b/'link.map')
 q['commands'].append(link);z=subprocess.run(link,capture_output=True,text=True,timeout=180);(b/'link.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-8000:];nm=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(x in nm for x in ['modern_kimi_vocab_TP16_component_r9176','modern_kimi_vocab_TP16_result_r9176']);ldd=subprocess.check_output(['ldd',str(native)],text=True);assert 'not found'not in ldd
 q.update(completed=True,native=str(native),native_SHA256=sha(native),all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},fresh_MD_match=True,complete_component_MD_SHA256={p:sha(p)for p in md},source_SHA256={p.name:sha(p)for p in s.iterdir()if p.is_file()},ldd=ldd,elapsed_s=time.monotonic()-start)
except BaseException as e:q['error']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({'completed':q['completed'],'error':q.get('error'),'native_SHA256':q.get('native_SHA256'),'actual_step':q['actual_step']}));assert q['completed']
