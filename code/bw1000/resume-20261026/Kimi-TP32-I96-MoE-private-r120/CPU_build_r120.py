"""Proposed HOST-only compile entry. Not executed by source preparation."""
import pathlib,json,hashlib,subprocess,shlex,time
D=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/refactor')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads((D/'source-manifest-r120.json').read_bytes())
 for name,h in m['package_source_SHA256'].items():assert sha(D/name)==h,name
 for p,h in m['pinned_libraries'].items():assert sha(p)==h,p
 parent_path=R/'candidates/Kimi-W2-pair-r9280/CPU-build.json'
 parent=json.loads(parent_path.read_bytes());assert parent['completed'] and parent['native_SHA256']==m['native_SHA256']
 commands=[c for c in parent['commands'] if any(str(v).endswith('/W2_pair_runner_r9280.cpp') for v in c)]
 assert len(commands)==1
 old=commands[0];assert pathlib.Path(old[0]).name in ['g++','c++'] and '-shared' in old
 b=R/'candidates/Kimi-TP32-I96-MoE-r120';b.mkdir(exist_ok=False)
 receipt={'completed':False,'CPU_only':True,'GPU':False,'GPU_compile':False,'dlopen':False,'Runtime_create':False,'HOST_TUs':1,'commands':[],'parent_receipt_SHA256':sha(parent_path)}
 start=time.monotonic()
 try:
  src=D/'TP32_I96_MoE_r120.cpp';lib=b/'libKimi_TP32_I96_MoE_r120.so';cmd=[]
  for v in old:
   if str(v).endswith('/W2_pair_runner_r9280.cpp'):v=str(src)
   elif v=='-MMD':v='-MD'
   elif str(v).endswith('/runner.d'):v=str(b/'adapter.d')
   elif str(v).startswith('-Wl,-Map'):v='-Wl,-Map,'+str(b/'link.map')
   cmd.append(v)
  cmd[cmd.index('-o')+1]=str(lib)
  if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(b/'adapter.d')
  else:cmd[1:1]=['-MD','-MF',str(b/'adapter.d')]
  cmd[1:1]=['-I'+str(D)]
  if '-pthread' not in cmd:cmd.insert(1,'-pthread')
  assert cmd.count(m['native'])==1,'link exactNative305726 once'
  assert '-Wl,--no-undefined' in cmd
  receipt['commands'].append(cmd);p=subprocess.run(cmd,capture_output=True,text=True,timeout=180);(b/'compile.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-8000:]
  # A second preprocessor dependency listing is CPU-only; no compiler output changes.
  flags=[x for x in cmd[1:] if x.startswith(('-I','-D','-std=')) or x=='-pthread']
  fresh=[cmd[0],*flags,'-M','-MF',str(b/'adapter.fresh.d'),str(src)];receipt['commands'].append(fresh)
  p=subprocess.run(fresh,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr[-4000:]
  def deps(p):return sorted(set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])))
  a=deps(b/'adapter.d');assert a==deps(b/'adapter.fresh.d'),'actual/fresh dependency mismatch'
  dependency_SHA256={p:sha(p) for p in a}
  for suffix,expected in m['header_source_suffix_SHA256'].items():
   hits=[p for p in a if p.endswith(suffix)];assert hits and all(sha(p)==expected for p in hits),(suffix,hits)
  nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);assert 'modern_kimi_TP32_moe_r120' in nm and 'modern_kimi_TP32_moe_result_r120' in nm
  ldd=subprocess.check_output(['ldd',str(lib)],text=True);assert 'not found' not in ldd
  receipt.update(completed=True,library=str(lib),library_SHA256=sha(lib),native=m['native'],native_SHA256=m['native_SHA256'],dependency_SHA256=dependency_SHA256,source_manifest_SHA256=sha(D/'source-manifest-r120.json'),ldd=ldd,elapsed_s=time.monotonic()-start)
 except BaseException as e:receipt['error']=repr(e)
 (b/'CPU-build.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt.get(k) for k in ['completed','error','library','library_SHA256']}));assert receipt['completed']
if __name__=='__main__':main()
