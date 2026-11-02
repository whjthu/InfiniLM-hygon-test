"""Actual controller argv/Popen and actual finite plan; recording boundary only."""
from pathlib import Path
import tempfile,json,hashlib,ast
import controller
from plan import FIXTURES,records
D=Path(__file__).resolve().parent;calls=[]
class Recording:
 pid=367000
 def __init__(self,args,**kw):calls.append((args,kw))
with tempfile.TemporaryDirectory(prefix='glm367-CPU-')as name:
 root=Path(name);(root/'logs').mkdir();p,args=controller.start(root,'a'*64,popen=Recording)
 assert p.pid==367000 and len(calls)==1 and args==controller.gpu_command(root)
 args,kw=calls[0];assert args[:2]==['srun','--jobid=910371']and args.count('--gres=dcu:8')==1 and args[args.index('-w')+1]=='a02r3n03'and args[args.index('-t')+1]=='00:10:00'and'-c32'in args and'-n1'in args
 assert kw['start_new_session']is True and kw['env']['GLM367_GPU_SOURCE_SHA256']=='a'*64 and kw['env']['WAVE_INSIDE_R2323']==controller.NAME+'/GPU_inside.sh'
 assert Path(args[args.index('--export=ALL')+2]).is_absolute()and Path(next(a.split('=',1)[1]for a in args if a.startswith('--output='))).is_absolute()
 try:controller.start(root,'a'*64,popen=Recording)
 except FileExistsError:pass
 else:raise AssertionError('Second same controller start accepted')
 assert len(calls)==1
all_records=[(fixture,r)for fixture in range(FIXTURES)for r in records(fixture)]
assert len(all_records)==52 and sum(r['warm']for _,r in all_records)==4 and sum(not r['warm']for _,r in all_records)==48
for fixture in range(2):
 rows=list(records(fixture));assert[rows[i]['mode']for i in [0,1]]==[0,1]
 for pair,order in enumerate([(0,1),(1,0),(0,1),(1,0)]):
  for arm,mode in enumerate(order):assert len([r for r in rows if not r['warm']and r['pair']==pair and r['arm']==arm and r['mode']==mode])==3
tree=ast.parse((D/'GPU_run.py').read_bytes());calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='glm_layer_forward_r5670']
assert len(calls)==2 and sorted((ast.literal_eval(c.args[1]),ast.literal_eval(c.args[2]),ast.literal_eval(c.args[-1]))for c in calls)==[(1,1,0),(512,1,0)]
for name in ['GPU_inside.sh','CPU_inside.sh']:data=(D/name).read_bytes();assert b'\r'not in data and data.endswith(b'\n')
for p in D.glob('*.py'):compile(p.read_bytes(),p.name,'exec')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=True,passed=True,CPU_only=True,GPU=False,Native_dlopen=False,actual_controller_Popen_recording=True,no_actual_Slurm_submission=True,duplicate_log_start_rejected=True,planned_warm=4,planned_measured=48,actual_Native_entry_AST=[dict(N=1,B=1,profile=0),dict(N=512,B=1,profile=0)],source_SHA256={n:sha(D/n)for n in ['controller.py','GPU_run.py','GPU_inside.sh','plan.py','check_host_CPU.py','resource_probe.py']})
(D/'host_CPU_result.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(q))
