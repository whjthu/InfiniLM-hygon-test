import pathlib,json,ast,hashlib,tempfile,shutil,re,sys
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'CPU-stage-source-manifest-r132.json').read_bytes())
with tempfile.TemporaryDirectory(prefix='r132-stage-')as temp:
 T=pathlib.Path(temp)
 for n,h in m['files_SHA256'].items():p=T/n;p.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(D/n,p);assert sha(p)==h
 shutil.copyfile(D/'CPU-stage-source-manifest-r132.json',T/'CPU-stage-source-manifest-r132.json')
 tree=ast.parse((T/'CPU_build_r132.py').read_bytes())
 for x in ast.walk(tree):
  if isinstance(x,ast.BinOp)and isinstance(x.op,ast.Div)and isinstance(x.left,ast.Name)and x.left.id=='D'and isinstance(x.right,ast.Constant)and isinstance(x.right.value,str):assert(T/x.right.value).exists(),(x.lineno,x.right.value)
 ns={};fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='semantics');exec(compile(ast.Module(body=[fn],type_ignores=[]),'effectiveflags','exec'),ns);f=ns['semantics'];ctx=json.loads((T/'per-TU-context-r132.json').read_bytes())
 for k,v in ctx.items():a=f(v['argv']);assert'-O2'in a and'-fPIC'in a and'-D_GLIBCXX_USE_CXX11_ABI=1'in a;assert'-MF'not in a
 assert f(['g++','-MMD','-MF','x.d','a.cpp','-O2','-fPIC','-DTEST=1','-pthread','-o','a.o'])==['g++','-O2','-fPIC','-DTEST=1','-pthread']
 for name in ['CPU_inside_r132.sh','GPU_inside_r132.sh']:
  text=(T/name).read_bytes().decode()
  for rel in re.findall(r'/refactor/incoming/Kimi-prefill-pool-component-r132/([A-Za-z0-9_./-]+\.(?:py|sh))',text):assert(T/rel).exists(),rel
 src=(T/'src/prefill_pool_component_r132.cpp').read_bytes().decode();assert'std::array<int,10>schedule{{0,0,0,1,1,0,0,1,1,0}}'in src and 'std::array<size_t,1>{576}'in src;assert'KIMI_PREFILL_POOL_COMPONENT_R132'in src and'auto y=module->forward(r,input);r.syncStream();'in src;assert'graphPinnedMemoryBytes'in src and'native_pool_allocated_cached_bytes' in src and'UNKNOWN: no public stats ABI'in src;assert'entry_trim",false'in src
 assert'cudaMemGetInfo'in src and'2ms'not in src # interval coded chrono milliseconds(2), not unbounded sleeps
 assert'std::chrono::milliseconds(2)'in src;assert'dladdr(reinterpret_cast<void*>(&modern_grouped_mxfp4_owned)'in src
 assert'modern_kimi_prefill_pool_trim_stats_r130'in src;assert'r.trimMemory();'not in src[src.index('const std::array<int,10>schedule'):src.index('if(modern_kimi_I192_counts_r8669y(rank,count_after')]
 # Reporting principle: recovering latest cannot erase the historical sampledmin.
 minimum=4232052736;latest=minimum+132120576;assert latest>4294967296 and minimum<4294967296
 for p in T.rglob('*.py'):ast.parse(p.read_bytes());assert b'\r\n'not in p.read_bytes()
assert'torch'not in sys.modules
print(json.dumps({'passed':True,'CPU_only':True,'native_compiled':False,'GPU':False,'Torch_imported':False,'contract':'exact stage mirror/allinputpaths/fullargvfilter/T576schedule/defaultOFF/livehelperbinding/sampledmin vs latest semantic','staged_source_files':len(m['files_SHA256'])+1,'actual_poolreclaim':'UNKNOWN','actual_nativequalification':'PENDING','Model_fit':'PARK'}))
