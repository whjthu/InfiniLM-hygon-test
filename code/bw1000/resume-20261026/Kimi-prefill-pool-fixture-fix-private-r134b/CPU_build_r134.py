"""One fixture HOST wrapper rebuild; reuse exact qualified CPU458 native. No GPU/dlopen."""
import pathlib,json,hashlib,subprocess,os,shlex,time,traceback
D=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/refactor/candidates/Kimi-prefill-pool-fixture-r134b');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert os.environ.get('KIMI_FIXTURE_CPU_R134')=='1';assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'new_GPU_TUs':0,'new_HOST_component_TUs':1,'CPU_contract_probe':True,'commands':[],'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};t=time.monotonic()
def run(c,label):
 q['commands'].append(c);z=subprocess.run(c,capture_output=True,text=True,timeout=180);(B/(label+'.log')).write_bytes((z.stdout+z.stderr).encode());assert z.returncode==0,(label,z.stderr[-6000:]);return z
try:
 pins=json.loads((D/'CPU-stage-manifest-r134.json').read_bytes());e=json.loads((D/'actual-CPU458-context.json').read_bytes())
 for p,h in pins['files_SHA256'].items():assert sha(D/p)==h,p
 assert sha(e['receipt_path'])==e['receipt_SHA256'];parent=json.loads(pathlib.Path(e['receipt_path']).read_bytes());assert parent['completed']
 for p,h in e['all_linked_object_SHA256'].items():assert sha(p)==h,p
 assert sha(e['native'])==e['native_SHA256']and sha(e['helper'])==e['helper_SHA256']
 for p,h in e['parent_actual_MD'].items():assert sha(p)==h,p
 cmd=e['component_argv'][:];oldsource=next(x for x in cmd if x.endswith('/src/prefill_pool_component_r132.cpp'));cmd[cmd.index(oldsource)]=str(D/'src/prefill_pool_component_r134.cpp');cmd[cmd.index('-o')+1]=str(B/'libKimi_prefill_pool_component_r134.so');cmd[cmd.index('-MF')+1]=str(B/'component.d');assert e['native']in cmd;run(cmd,'component')
 probe=['g++','-std=c++17','-O2','-pthread','-I'+str(D/'src'),str(D/'src/owner_contract_CPU_r134.cpp'),'-o',str(B/'owner-contract')];run(probe,'contract-build');run([str(B/'owner-contract')],'contract-run')
 # Fresh-M replays every semantic flag irrespective argv ordering.
 sem=[cmd[0]];i=1
 while i<len(cmd):
  x=cmd[i]
  if x in ['-o','-MF','-MT','-MQ','-MJ','-L','-l']:i+=2;continue
  if x in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if x.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or x.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  sem.append(x);i+=1
 run(sem+['-M','-MF',str(B/'fresh.d'),'-MT',str(B/'component.o'),str(D/'src/prefill_pool_component_r134.cpp')],'freshMD');parse=lambda p:set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]));assert parse(B/'component.d')==parse(B/'fresh.d')
 env=dict(os.environ,LD_LIBRARY_PATH=e['LD_LIBRARY_PATH']);z=subprocess.run(['ldd','-r',str(B/'libKimi_prefill_pool_component_r134.so')],text=True,capture_output=True,timeout=60,env=env);assert z.returncode==0 and 'not found'not in z.stdout and 'undefined symbol:'not in z.stdout+z.stderr and e['helper']in z.stdout
 q.update(completed=True,native=e['native'],native_SHA256=e['native_SHA256'],library=str(B/'libKimi_prefill_pool_component_r134.so'),library_SHA256=sha(B/'libKimi_prefill_pool_component_r134.so'),helper=e['helper'],helper_SHA256=e['helper_SHA256'],actual_MD={p:sha(p)for p in parse(B/'component.d')},ldd=z.stdout,LD_LIBRARY_PATH=e['LD_LIBRARY_PATH'],Model_fit='PARK')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-t;(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','library_SHA256','error']}));assert q['completed']
