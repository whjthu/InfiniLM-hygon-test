import ast,pathlib,json,ctypes,hashlib
r=pathlib.Path('.');proof={};ns={}
def getdefs(name,names):
 tree=ast.parse((r/name).read_text());body=[v for v in tree.body if isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))and v.name in names];assert len(body)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),name,'exec'),ns)
getdefs('modern_Kimi_TP16PP2_I192_r8669zc.py',['verify_compact_constructor_r8669zc','preload_memory_gate_r8669zc'])
getdefs('kimi_TP16PP2_I192_protocol_r8669zc.py',['postQA_memory_gate_r8669zc'])
class Counter:
 def __init__(self,n,bad=False):self.n=n;self.bad=bad
 def __call__(self,d,out,n):out[0]=self.n-(self.bad and d==7);out[1]=self.n*4468480;return 0
class Lib:pass
for stage in [0,1]:
 x=Lib();x.modern_kimi_I192_workspace_r8669zc=Counter([45,47][stage]);assert ns['verify_compact_constructor_r8669zc'](x,stage)==[[[45,47][stage],[45,47][stage]*4468480]]*8
 x.modern_kimi_I192_workspace_r8669zc=Counter([45,47][stage],True)
 try:ns['verify_compact_constructor_r8669zc'](x,stage);raise RuntimeError('missing reject')
 except AssertionError:pass
proof['actual_constructor_counter_consumer_stage0_stage1_and_bad_rank']=True
class Control:
 def __init__(self,free,events):self.free=free;self.events=events
 def collect(self,label,packet):self.events.append(label);return [{'memory':[(self.free,64*1024**3)]*8}for _ in range(4)]
for delta in [1,0,-1]:
 events=[];report={'memory_after_create':[(4*1024**3+delta,64*1024**3)]*8};c=Control(4*1024**3+delta,events)
 try:ns['preload_memory_gate_r8669zc'](c,report,lambda:events.append('save'));passed=True
 except AssertionError:passed=False
 assert passed==(delta>0);assert events==['compact-preload-memory4GiB','save'];assert report['preload_memory_gate']['minimum_free_bytes']==4*1024**3+delta
proof['actual_preload_strict4GiB_boundary_saved_before_reject']=True
for after in [int(4.38*1024**3),int(3.12*1024**3)]:
 events=[];report={};samples=iter([[(int(2.48*1024**3),64*1024**3)]*8,[(after,64*1024**3)]*8]);g={'memory':lambda:next(samples)};c=Control(after,events)
 try:ns['postQA_memory_gate_r8669zc'](g,report,lambda:events.append('save'),lambda name:events.append('reset:'+name),c);passed=True
 except AssertionError:passed=False
 assert passed==(after>4*1024**3);assert events==['save','reset:after-QA-before-geometry','save','geometry-memory:16:16','save'];assert report['postQA_reset_memory']['after'][0][0]==after
proof['actual_postQA_reset_before_after_and_global_gate_no_lowering']=True
for fn in ['forward','generate']:
 trees=[ast.parse((r/n).read_text())for n in ['kimi_TP16PP2_I192_protocol_r8669za.py','kimi_TP16PP2_I192_protocol_r8669zc.py']];defs=[next(x for x in ast.walk(t)if isinstance(x,ast.FunctionDef)and x.name==fn)for t in trees];assert ast.dump(defs[0],include_attributes=False)==ast.dump(defs[1],include_attributes=False)
proof['actual_forward_and_generate_AST_unchanged_including_flags_counter_decode_and_finalIDs']=True
for name in ['launch_Kimi_TP16PP2_I192_r8669zc.py','Kimi_TP16PP2_I192_binding_r8669zc.py']:
 s=(r/name).read_text();assert 'results/Kimi-I192-compact-CPU-r8669zc.json'in s and 'results/Kimi-TP16-head-CPU-r8669z.json'not in s
s=(r/'Kimi_TP16PP2_I192_r8669zc_inside.sh').read_text();assert '2900 python'in s and 'INFINILM_KIMI_I192_COMPACT_R8669ZC=1'in s
proof['actual_newreceipt_binding_and_inner2900']=True
proof['GPU_or_CAPI_executed']=False
p=r/'Kimi-compact-runtime-CPU-r8669zc.json';p.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(proof)
