import pathlib,sys,os,json,hashlib,ast,importlib
import common
D=common.SRC;r=common.ROOT;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pin=json.loads((D/'model-source-pins.json').read_bytes());mp=r/'incoming/Kimi-adaptive-Model-AB-r9225/runtime-A/model-manifest-r9183.json'
assert sha(mp)==pin['manifest_SHA256']=='33de1ad4f5a3dcc5d984fee3311263b023a010bb98486fcc07a4652244ae8697'
m=json.loads(mp.read_bytes());assert m['topology']=={'tp_nodes':2,'pp':2,'logical_capacity':16}and m['nodes']==common.RES['target_nodes']and m['resources_proposal']['hard_seconds']==10800 and m['resources_proposal']['inner_seconds']==10600
for name in ['secrets','re','datetime','socket','traceback','subprocess','time']:importlib.import_module(name)
for path,h in pin['files'].items():assert sha(path.replace('/refactor',str(r),1))==h,path
for p in D.glob('*.py'):raw=p.read_bytes();assert bytes([13])not in raw;ast.parse(raw.decode('utf8'))
closure=json.loads((r/'results/Kimi-adaptive-Model-AB-CPUclosure-r9225.json').read_bytes());assert closure['passed']and closure['completed']and not closure['GPU_submitted']and all(v['terminal'][2:4]==['COMPLETED','0:0']for v in closure['groups'])
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];common.atomic(r/'results'/('Kimi-adaptive-A-controller-CPU-r9234-'+step+'.json'),{'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'actual_step':step,'manifest_SHA256':sha(mp),'sourcepins_SHA256':sha(D/'model-source-pins.json'),'existing_actual12node_CPU_qualification':'394/459/733/493 COMPLETED0','new_controller_LF_AST_dependencies_bindings_checked':True,'GPU_gate_next':'parentownership/freshKFD/watch, then uniqueGPU fromsamekeeper'})
os.execv(sys.executable,[sys.executable,str(D/'model_keeper.py')])
