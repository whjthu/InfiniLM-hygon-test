"""Recording control/finite plan and real tape consumer; no Slurm/Native/GPU."""
from pathlib import Path
import tempfile,json,ast,hashlib,copy
import controller
from plan import FIXTURES,records
from profile_contract import REQUIRED,FIELDS,validate_profile
D=Path(__file__).resolve().parent;calls=[];cases=[]
class Recording:
 pid=370000
 def __init__(self,args,**kw):calls.append((args,kw))
with tempfile.TemporaryDirectory(prefix='glm370-CPU-')as name:
 root=Path(name);(root/'logs').mkdir();p,args=controller.start(root,'b'*64,popen=Recording);assert p.pid==370000 and len(calls)==1 and args==controller.gpu_command(root)
 args,kw=calls[0];assert args[:2]==['srun','--jobid=910371']and args.count('--gres=dcu:8')==1 and args[args.index('-w')+1]=='a02r3n03'and args[args.index('-t')+1]=='00:10:00'and'-c32'in args and'-n1'in args
 assert kw['start_new_session']is True and kw['env']['GLM370_GPU_SOURCE_SHA256']=='b'*64 and kw['env']['WAVE_INSIDE_R2323']==controller.NAME+'/GPU_inside.sh'
 assert Path(args[args.index('--export=ALL')+2]).is_absolute()and Path(next(a.split('=',1)[1]for a in args if a.startswith('--output='))).is_absolute()
 try:controller.start(root,'b'*64,popen=Recording)
 except FileExistsError:pass
 else:raise AssertionError('Second actual start accepted')
 assert len(calls)==1;cases.append('actual_Popen_argv_duplicate_once')
rows=list(records(0));assert FIXTURES==1 and len(rows)==14 and sum(r['warm']for r in rows)==2 and sum(not r['warm']for r in rows)==12
assert[r['mode']for r in rows[:2]]==[0,1]and[rows[2+i*3]['mode']for i in range(4)]==[0,1,1,0];cases.append('actual_finite_OFF_ON_ON_OFF_plan')
base=dict(N=1,B=1,layers=[38],profile=0,stages=[]);validate_profile(base,0)
names=list(REQUIRED);names[14:14]=['extra.quant','extra.quant'];stages=[dict(**{'from':a,'to':b},**{k:.1 for k in FIELDS})for a,b in zip(names,names[1:])];on={**base,'profile':1,'stages':stages};validate_profile(on,1);assert len(on['stages'])==len(stages);cases.append('all_adjacent_tape_preserved_repeated_nonstructural_allowed')
for label,mutate in [('missing_required',lambda q:q['stages'].pop(8)),('negative_interval',lambda q:q['stages'][0].update(host_wall_ms=-1)),('nan_interval',lambda q:q['stages'][0].update(threadCPU_ms=float('nan'))),('crossed_chain',lambda q:q['stages'][4].update({'from':'wrong'})),('wrong_profile',lambda q:q.update(profile=0))]:
 bad=copy.deepcopy(on);mutate(bad)
 try:validate_profile(bad,1)
 except(AssertionError,KeyError):cases.append('reject_'+label)
 else:raise AssertionError(label)
tree=ast.parse((D/'GPU_run.py').read_bytes());native=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='glm_layer_forward_r5670'];assert len(native)==2
assert[(ast.literal_eval(n.args[1]),ast.literal_eval(n.args[2]),ast.unparse(n.args[-1]))for n in native]==[(512,1,'0'),(1,1,'mode')];assert"q['input_fixtures'][1]"not in(D/'GPU_run.py').read_text();cases.append('actual_callback_AST_prefix0_decode_mode_single_fixture')
der=json.loads((D/'DERIVATION.json').read_bytes());assert der['checkpoint_loader_byteidentical']and der['no_new_CPP_or_GPU_TU']
for name in ['GPU_inside.sh']:data=(D/name).read_bytes();assert b'\r'not in data and data.endswith(b'\n')
for p in D.glob('*.py'):compile(p.read_bytes(),p.name,'exec')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=dict(completed=True,passed=True,cases=cases,CPU_only=True,Native_dlopen=False,GPU=False,actual_Slurm=False,actual_finite_plan_and_Popen_recording=True,actual_tape_consumer_CPU=True,planned_per_owner=dict(warm=2,formal=12,profileON_formal=6,profileOFF_formal=6),callback_AST='Prefix N512/B1/profile0; decodeN1/B1/profilemode0or1',existing_R367_actual_GPU_loader_and_hardware_reused=True,actual_profile_GPU_numeric_or_overhead_qualified=False,source_SHA256={n:sha(D/n)for n in ['GPU_run.py','GPU_inside.sh','controller.py','resource_probe.py','plan.py','profile_contract.py','check_host_CPU.py','DERIVATION.json']})
(D/'host_CPU_result.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(q))
