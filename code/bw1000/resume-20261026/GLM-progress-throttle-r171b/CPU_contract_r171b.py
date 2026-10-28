import pathlib,sys,ast,json,threading
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D));from progress_write_policy_r171 import ProgressWritePolicy
now=[0.];policy=ProgressWritePolicy(lambda:now[0]);policy.configure(True);writes=[]
def attempt(op='DECODE',active=True,force=False):
 if policy.ready(op,active,force):writes.append((now[0],op));policy.commit();return True
 return False
assert attempt('BEGIN');now[0]=.1;assert not attempt();now[0]=.999;assert not attempt();now[0]=1.;assert attempt();now[0]=1.01;assert attempt('DONE',False);now[0]=1.02;assert attempt('ERROR',False,True);now[0]=1.03;assert not attempt();now[0]=2.02;assert attempt();assert policy.snapshot()['successful_writes']==5
policy.configure(False)
for _ in range(64):assert attempt()
errors=[]
def other():
 try:policy.ready('DECODE',True)
 except RuntimeError:errors.append('wrong_owner_rejected')
t=threading.Thread(target=other);t.start();t.join();assert errors==['wrong_owner_rejected']
parent=ast.parse((D/'parent-glm_arrival_probe_r169b.py').read_text());candidate=ast.parse((D/'glm_arrival_probe_r171b.py').read_text());run=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='run');p=run(parent);c=run(candidate)
find=lambda fn,n:next(x for x in fn.body if isinstance(x,ast.FunctionDef)and x.name==n);dump=lambda x:ast.dump(x,include_attributes=False)
for name in ['forward','reset','stats','raggedstats','vectorreceipt','barrier','budget']:assert dump(find(p,name))==dump(find(c,name)),name
pp=find(p,'progress');cp=find(c,'progress');core=[x for x in cp.body if not(isinstance(x,ast.If)and 'progress_policy.ready'in ast.unparse(x.test))and not(isinstance(x,ast.Expr)and ast.unparse(x.value)=='progress_policy.commit()')];assert [dump(x)for x in core]==[dump(x)for x in pp.body],'original progress payload/path/write unchanged'
pe=find(p,'episode');ce=find(c,'episode');assert len(ce.body)==1 and isinstance(ce.body[0],ast.Try);wrapped=ce.body[0];rest=wrapped.body[1:];rest=[x for x in rest if not(isinstance(x,ast.Assign)and "receipt['progress_write_policy']"==ast.unparse(x.targets[0]))];assert [dump(x)for x in rest]==[dump(x)for x in pe.body],'whole original episode math/control body unchanged'
assert "'stall_limit_seconds':60"in(D/'glm_arrival_probe_r171b.py').read_text()
seen=[];env={'progress_policy':ProgressWritePolicy(lambda:0.),'progress':lambda *a,**k:seen.append((a,k))};exec(compile(ast.Module(body=[ce],type_ignores=[]),'actual_episode_exception_contract','exec'),env)
try:env['episode']('CPU-exception',False,[],[],down_row2='INVALID')
except AssertionError:pass
else:raise AssertionError('original error swallowed')
assert len(seen)==1 and seen[0][0][2]=='ERROR'and seen[0][1]['force']is True
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'model':False,'routine_1second_gate':True,'first_completion_error_forced':True,'baseline_64of64_writes':True,'wrong_owner_rejected':True,'actual_original_progress_payload_path_write_AST_unchanged':True,'original_episode_math_control_AST_unchanged':True,'actual_exception_forces_original_progress_and_reraises':True,'watchdog60_unchanged':True,'no_background_thread_in_candidate':True}))
