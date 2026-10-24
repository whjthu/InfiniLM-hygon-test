import pathlib,json,hashlib,ast,types,sys,importlib.util
r=pathlib.Path(__file__).resolve().parent;d=r/'GLM-DP-Model-private-r8864';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8864.json';m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():
 assert sha(d/n)==h,n
 if n.endswith('.py'):ast.parse((d/n).read_bytes(),filename=n)
assert m['model_SHA256']=='68acfe089f757014e564c0d0ca5c896ac1f47cd98fa3a843df8b2dd5bf8c63c6'and m['decode_micro_fixed']==8 and m['maximum_loads_per_replica']==1 and m['maximum_full128_arms_per_replica']==2
assert m['resident_physical_bytes_per_layer']==160*1024**2 and m['reserve_bytes']==6*1024**3 and m['allowance_bytes']==64*1024**2 and m['runtime_floor_bytes']==512*1024**2
# Execute actual node namespace statements only, without importing the entry/GPU worker.
t=ast.parse((d/'run_GLM_DP_node_r8864.py').read_bytes());selected=[]
for n in t.body:
 if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id in('slot','a','stage','common')for v in n.targets):selected.append(n)
 elif isinstance(n,ast.Assert)and 'socket.gethostname()'in ast.unparse(n.test):selected.append(n)
 elif isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='os.environ.update':selected.append(n)
code=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'actual-node-namespace','exec');namespaces=[]
for slot,node in enumerate(m['DP_nodes_by_slot']):
 env={'SLURM_PROCID':str(slot),'GLM_RUN_R5040':'CPU-common'};ns={'os':types.SimpleNamespace(environ=env),'socket':types.SimpleNamespace(gethostname=lambda node=node:node),'m':m,'r':pathlib.PurePosixPath('/refactor')};exec(code,ns)
 a='A'if slot<2 else'B';assert env['GLM_STAGE_R5040']==str(slot%2)and env['GLM_DP_REPLICA_R8864']==a and env['GLM_RUN_R5040']=='CPU-common-'+a and env['GLM_PIPELINE_TAIL_NODE']==m['DP_nodes_by_slot'][1 if a=='A'else 3];namespaces.append(dict(env))
try:exec(code,{'os':types.SimpleNamespace(environ={'SLURM_PROCID':'0','GLM_RUN_R5040':'CPU-common'}),'socket':types.SimpleNamespace(gethostname=lambda:'wrong-node'),'m':m,'r':pathlib.PurePosixPath('/refactor')})
except AssertionError:pass
else:raise AssertionError('wrong node accepted')
# Actual probe mapping statements and original reference contract.
probe=ast.parse((d/'glm_prefill_micro_probe_r8864.py').read_bytes());fn=next(x for x in probe.body if isinstance(x,ast.FunctionDef)and x.name=='run');mapping=[]
for n in fn.body:
 if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id in('global_rows','i512_inputs','historical_i512')for v in n.targets)and('global_rows' in ast.unparse(n.value)or ast.unparse(n.value).startswith('list(range(')):mapping.append(n)
assert len(mapping)==3;mc=compile(ast.fix_missing_locations(ast.Module(body=mapping,type_ignores=[])),'actual-probe-global-rowmap','exec');gold=json.loads((d/'GLM477-frozen-gold-r8844.json').read_bytes());contract={};exec((d/'glm_crossrequest_contract_r8864.py').read_bytes(),contract)
for a,rows in m['DP_global_rows'].items():
 ns={'replica':a,'i512_inputs':gold['input_ids'],'historical_i512':gold};exec(mc,ns);assert ns['global_rows']==rows and ns['i512_inputs']==[gold['input_ids'][i]for i in rows]
 mapped=ns['historical_i512'];ok,_=contract['validate_reference']('I512','O',ns['i512_inputs'],mapped['generated_ids'],mapped['stopped_at_EOS'],128,{},ns['i512_inputs'],{},mapped);assert ok
 bad=[list(v)for v in mapped['generated_ids']];bad[0][0]=(bad[0][0]+1)%154880;assert not contract['validate_reference']('I512','O',ns['i512_inputs'],bad,mapped['stopped_at_EOS'],128,{},ns['i512_inputs'],{},mapped)[0]
# Actual commonclock metrics on CPU-only real-socket synthetic evidence.
sys.path.insert(0,str(d));from glm_DP_clock_metrics_r8864 import common_metrics
clock=json.loads((r/'GLM-DP-synthetic-clock-CPU-r8864.json').read_bytes());reports={a:[{'phase':f'perf-I512-DP-{a}-{p}-N-r0','batch':16,'micro':8,'input_lengths':[512]*16,'generated_ids':[[1,2,3]for _ in range(16)]}for p in('iso'+a,'joint')]for a in('A','B')}
metrics=common_metrics(clock,reports,set());assert metrics['joint']['cards']==32 and metrics['joint']['useful_postallprefill_decisions']==64 and metrics['isoA']['useful_postallprefill_decisions']==32 and metrics['joint']['actual_joint_overlap_proved']
assert all(x['nonEOS_tokens']==3 for x in metrics['joint']['per_request'])
eos_metrics=common_metrics(clock,reports,{3});assert eos_metrics['joint']['useful_postallprefill_decisions']==32 and all(x['nonEOS_tokens']==2 for x in eos_metrics['joint']['per_request'])
bad=json.loads(json.dumps(clock));bad['records'][0]['ids'][0]=999
try:common_metrics(bad,reports,set())
except AssertionError:pass
else:raise AssertionError('wrong recorded ID accepted')
# Frozen actual resource expressions and final protocol are explicit.
lt=ast.parse((d/'launch_GLM_prefill_Model_r8864.py').read_bytes());args=ast.literal_eval(next(x.value for x in lt.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='args'for t in x.targets)))
assert args[args.index('-N4')+1]=='-n4'and args[args.index('-w')+1]==','.join(m['DP_nodes_by_slot'])and args[args.index('-t')+1]=='01:00:00'and '--jobid=910371'in args and '--mem=240000M'in args
assert '3500 python -u /refactor/incoming/run_GLM_DP_node_r8864.py'in(d/'glm_prefill_micro_Model_r8864_inside.sh').read_text()
assert 'SLURM_PROCID'not in(d/'GLM_DP_worker_inside_r8864.sh').read_text().split('GLM_STAGE_R5040=')[-1] if 'GLM_STAGE_R5040='in(d/'GLM_DP_worker_inside_r8864.sh').read_text()else True
protocol=json.loads((r/'GLM-DP-protocol-CPUchecks-r8864.json').read_bytes());assert protocol['passed']and protocol['actual_feedback_clients']==32
out={'CPU_only':True,'GPU':False,'dlopen':False,'passed':True,'manifest_SHA256':sha(mp),'entry_SHA256':sha(d/'dispatch_GLM_resident_exact_Model_r8864.py'),'DSO_SHA256':m['model_SHA256'],'actual_node_namespace_4slots_and_wrongnode_negative':'PASS','actual_probe_global477_rowmap_and_badtoken_negative':'PASS','actual_feedback_32client_protocol':protocol,'actual_commonclock_metrics_overlap_counts_EOS_wrongID_negative':'PASS','actual_resource_910371_4nodes_c64_m240000_60min3500':'PASS','namespaces':namespaces,'scope':'CPU protocol/namespace/reference only. Model/provider GPU qualification inherited561M8; DP concurrent quality/performance/peak and nodeKFD UNKNOWN until authorizedentry.'}
(r/'GLM-DP-Model-CPUchecks-r8864.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in out.items()if k not in('namespaces','actual_feedback_32client_protocol')}))
