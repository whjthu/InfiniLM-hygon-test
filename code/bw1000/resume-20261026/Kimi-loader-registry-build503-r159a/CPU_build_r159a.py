"""CPU-only private worker/registry build on the exact completed503 parent."""
import argparse,hashlib,json,os,pathlib,re,shlex,subprocess,time,traceback
from parent_receipt_r159a import normalize,audit_symbols,PARENT_NATIVE,PARENT_RECEIPT,OLDWORKER,OLDWORKER_SHA,PRESERVED
D=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--source-manifest-sha',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
B=pathlib.Path(args.out).resolve();B.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'commands':[]}
def run(cmd,label,limit=100,env=None):
 assert cmd and pathlib.Path(cmd[0]).name not in('nvcc','hipcc','hcc') and not any(str(x).endswith(('.cu','.hip'))for x in cmd),'no GPU TU compiler/source permitted'
 remaining=170-(time.monotonic()-start);assert remaining>0,'whole CPU build budget exhausted'
 q['commands'].append(cmd)
 try:r=subprocess.run(cmd,text=True,capture_output=True,timeout=min(limit,remaining),env=env)
 except subprocess.TimeoutExpired as error:
  text=(error.stdout or b'')+(error.stderr or b'') if isinstance(error.stdout,bytes) else (error.stdout or '')+(error.stderr or '')
  (B/(label+'.log')).write_text(text.decode(errors='replace')if isinstance(text,bytes)else text);raise
 (B/(label+'.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,(label,r.returncode,r.stderr[-4096:]);return r.stdout
try:
 assert sha(D/'source-manifest-r159a.json')==args.source_manifest_sha
 source=json.loads((D/'source-manifest-r159a.json').read_bytes())
 for name,h in source['build_files_SHA256'].items():assert sha(D/name)==h,name
 pp=pathlib.Path('/refactor/candidates/Kimi-MoE-phase-component-r155/CPU-build.json');assert sha(pp)==PARENT_RECEIPT
 parent=json.loads(pp.read_bytes());pins=normalize(parent);assert sha(parent['native'])==PARENT_NATIVE
 for path,h in pins['objects'].items():assert sha(path)==h,path
 # Dependencies are the actual recorded503 compile MD, not invented include policies.
 for table in pins['new_MD'].values():
  for path,h in table.items():assert sha(path)==h,('parent MD changed',path)
 templates=json.loads((D/'compile-templates-r159a.json').read_bytes())
 assert templates['qualified_native_SHA256']=='2e5c21ba5738ce3711e1dfd2c033ebb641f99fa934268af3757b28010127ec75'
 for name in('CPU_contract','CPU_clock_contract_r159'):
  run(['g++','-std=c++17','-O2','-pthread','-I'+str(D),str(D/(name+'.cpp')),'-o',str(B/name)],'compile-'+name,40)
  q[name]=json.loads(run([str(B/name)],'execute-'+name,10));assert q[name]['passed']
 for name,file in(('worker','worker_r9212.cpp'),('weights','weights_r9212.cpp')):
  cmd=templates['commands'][name][:];assert cmd[0]=='g++'and '-c'in cmd and '-MF'in cmd
  cmd.insert(1,'-I'+str(D));cmd[cmd.index('-c')+1]=str(D/file);cmd[cmd.index('-o')+1]=str(B/(name+'.raw.o'));cmd[cmd.index('-MF')+1]=str(B/(name+'.d'));run(cmd,'compile-'+name)
 raw=run(['nm',str(B/'worker.raw.o')],'worker-symbols',10);names=[x.split()[-1]for x in raw.splitlines()if 'load_parameters_no_sync'in x and ' U 'in x];assert len(names)==1;oldsymbol=names[0];unique='modern_kimi_load_parameters_cached_r9212'
 assert oldsymbol in run(['nm','--defined-only',str(B/'weights.raw.o')],'weights-symbols',10)
 for name in('worker','weights'):run(['objcopy','--redefine-sym',oldsymbol+'='+unique,str(B/(name+'.raw.o')),str(B/(name+'.o'))],'private-rename-'+name,10)
 raw=run(['nm','--defined-only',str(B/'weights.o')],'weights-raw',10).splitlines();dem=run(['nm','-C','--defined-only',str(B/'weights.o')],'weights-demangle',10).splitlines();assert len(raw)==len(dem)
 localized=[a.split()[-1]for a,b in zip(raw,dem)if ' T 'in a and a.split()[-1].startswith(('_ZN8infinilm7weights','_ZNK8infinilm7weights'))];assert localized
 run(['objcopy',*[a for s in localized for a in('--localize-symbol',s)],str(B/'weights.o')],'localize-private-Weight',10)
 link=pins['link'][:];assert link.count(OLDWORKER)==1;link[link.index(OLDWORKER)]=str(B/'worker.o');link.insert(link.index('-o'),str(B/'weights.o'));native=B/'libKimi_loading_registry_r159a.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map')else x for x in link]
 assert not any(x.endswith(('.cu','.cpp'))for x in link),'link must use preserved math objects only'
 run(link,'link-private-Native',40)
 exports=run(['nm','-D','-C','--defined-only',str(native)],'exports',10);assert unique in exports and 'modern_kimi_loading_registry_work_r159'in exports
 assert not any(' T 'in x and 'infinilm::weights::'in x for x in exports.splitlines()),'public Weight implementation interposition'
 oldraw=run(['nm','-D','--defined-only',parent['native']],'parent-exports',10);newraw=run(['nm','-D','--defined-only',str(native)],'candidate-exports',10)
 q.update(audit_symbols(oldraw,newraw,unique))
 needed={}
 for label,path in(('parent',parent['native']),('candidate',str(native))):
  dynamic=run(['readelf','-d',path],label+'-dynamic',10);needed[label]=sorted(re.findall(r'Shared library: \[([^]]+)\]',dynamic));run(['readelf','--version-info',path],label+'-versions',10)
 assert needed['parent']==needed['candidate'],needed;q['DT_NEEDED']=needed
 env=dict(os.environ,LD_LIBRARY_PATH=pins['LD_LIBRARY_PATH']);deps=run(['ldd','-r','-v',str(native)],'ldd-r-version-closure',20,env);assert 'not found'not in deps and 'undefined symbol:'not in deps
 md={}
 for f in B.glob('*.d'):
  paths=shlex.split(f.read_text().replace('\\\n',' ').split(':',1)[1]);md[f.name]={x:sha(x)for x in paths if pathlib.Path(x).is_file()}
 kept={x:h for x,h in pins['objects'].items()if x!=OLDWORKER};assert all(sha(x)==h for x,h in kept.items())
 q.update(completed=True,native=str(native),native_SHA256=sha(native),parent_build_SHA256=PARENT_RECEIPT,parent_native_SHA256=PARENT_NATIVE,source_manifest_SHA256=args.source_manifest_sha,all_linked_object_SHA256={x:sha(x)for x in link if x.endswith('.o')},preserved_math_objects=PRESERVED,unchanged_object_SHA256=kept,new_MD=md,LD_LIBRARY_PATH=pins['LD_LIBRARY_PATH'],GPU_TUs_compiled=0,host_TUs_compiled=2,Core_modified=False,H2D_DMA_attribution='UNKNOWN',actual_GPU_qualified=False,source_only_not_Model_GO=True)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
finally:
 q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in('completed','actual_step','elapsed_s','native_SHA256','error')}));assert q['completed']
