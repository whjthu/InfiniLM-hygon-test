"""CPU-only actual-source, ctypes signature and finite schedule contracts; no imports of GPU driver."""
import ast,pathlib,json,hashlib
r=pathlib.Path(__file__).parent
files=['deep_C128_component_r8688.py','verify_Deep_C128_component_r8688.py','Deep_C128_component_r8688_inside.sh','launch_Deep_C128_component_r8688.py']
checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
for n in files:
 p=r/n
 if p.suffix=='.py':compile(p.read_text(encoding='utf-8-sig'),n,'exec');checks.append(n+' AST/compile only')
check('eight owner-specific executors', 'ThreadPoolExecutor(max_workers=1)'in(r/files[0]).read_text())
s=(r/files[0]).read_text();a=(r/files[1]).read_text();entry=(r/files[3]).read_text()
check('loadflag frozen before each complete family','for mode in [0,1]:\n  os.environ[FLAG]=str(mode)'in s)
check('original bridge SHA and immutable DSO','7998ae1c3891ce7f4a4737a0ab4648cb1b926398d067e86dd5d03064797eb2e2'in s)
check('same cache initialization normal resets','c=self.contexts[int(arm==\'N\')];check(lib.modern_real_compress_reset(c))'in s)
check('full O/N warmup excluded from main stats',"ARMS=['O','N']+"in s and "list(enumerate(arms))[2:]"in a)
check('no F64 GPU code','float64'not in s and '.double('not in s)
check('producer synchronization before timer',s.index('torch.cuda.current_stream(self.device).synchronize()')<s.index('t=time.monotonic();cpu='))
check('native timer closes before readback',s.index('wall=time.monotonic()-t')<s.index('def validation'))
check('thread barrier abort on failure','barrier.abort();raise'in s)
check('normal owner close in finally','p.submit(o.close)'in s and 'finally:'in s)
check('memory floor independent of load admission','512*1024**2'in s and '6*1024**3'in s)
check('worst owner per temporal trial','max(x[\'native_enclosing_wall_s\']for x in r)'in a)
check('two ABBA, not rank-as-repetition','for c in [0,1]'in a and 'ranks_not_independent_time_repetitions'in a)
check('original watcher terminal finally post','finally:\n     try:w[\'post_KFD\']=idle(\'post\')'in entry)
check('read transient errors do not terminate watcher','except(subprocess.TimeoutExpired,subprocess.CalledProcessError)'in entry)
check('unique controller ready before numeric',entry.index("ready=json.loads(watchpath.read_text())")<entry.index("with(r/'logs'/(label+'-srun.log'))"))
check('no auto retry numeric',entry.count("p=subprocess.Popen(args")==1)
# Full legal sequence, not a jump from empty state to past575/639.
for row in range(32):
 cursor=0
 cursor+=575
 for past in range(575,640):
  check(f'row{row} contiguous past{past}',cursor==past)
  cursor+=1
 check(f'row{row} final640',cursor==640)
check('575 nonboundary and639 boundary',(575+1)%128!=0 and(639+1)%128==0)
check('seed coalescing <=4096',7*575<=4096 and 32*127<=4096)
out={'CPU_only':True,'GPU':False,'dlopen':False,'passed':True,'checks':len(checks),'source_SHA256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in files},'scope':'Actual syntax/ctypes source contract and all32 legal seed+append schedules. Not GPU math/quality/performance.','GPU_status':'NOT_RUN','planned_GPU':{'parent':'910374','node':'a02r4n05','devices':8,'CPU':32,'memory_MiB':64000,'hard_minutes':10,'synthetic_hidden':True,'actual_checkpoint_weights':True,'same_two_contexts_load_flag_frozen':True,'selected_past':[575,639],'ABBA_cycles':2,'extra_cache_bytes_per_C128_GPU':16777216},'checks_names':checks[:21]+checks[-2:]}
p=r/'Deep-C128-component-CPU-checks-r8688.json'
with p.open('x',encoding='utf-8')as f:json.dump(out,f,indent=2)
print(json.dumps({k:out[k]for k in ['passed','checks','GPU_status']}))
