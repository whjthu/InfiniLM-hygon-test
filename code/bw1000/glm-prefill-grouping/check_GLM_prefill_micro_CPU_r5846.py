"""CPU host geometry and extracted real emitter decision; no Torch/dlopen."""
import ast,hashlib,json,pathlib
from glm_large_schedule_r3627 import groups
r=pathlib.Path(__file__).resolve().parent;checks=[]
def ok(k,v):assert v,k;checks.append(k)
rows=list(range(16));lengths=[512]*16
for prefill in(4,8):
 g=groups(rows,prefill,True,lengths)
 ok('eachrow_once-prefill'+str(prefill),sum(g,[])==rows)
 ok('legal_token_budget-prefill'+str(prefill),all(sum(lengths[i]for i in c)<=4096 for c in g))
 ok('call_count-prefill'+str(prefill),len(g)==16//prefill)
 generated=[[]for _ in rows]
 for cohort in g:
  for i in cohort:generated[i].append(i+100)
 ok('one_firstdecision_before_decode-prefill'+str(prefill),all(len(v)==1 for v in generated))
 ok('decodecohorts_unchanged8-prefill'+str(prefill),groups(rows,8,False,lengths)==[rows[:8],rows[8:]])
ok('requested16clips8_not_real8192',groups(rows,16,True,lengths)==[rows[:8],rows[8:]])
text=(r/'glm_prefill_micro_probe_r5846.py').read_text();tree=ast.parse(text)
block=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=='participant == emitter'and any(isinstance(v,ast.Name)and v.id=='gain'for v in ast.walk(n)))
module=ast.Module(body=[ast.Assign(targets=[ast.Name(id='decision',ctx=ast.Store())],value=ast.Constant(0)),block],type_ignores=[]);ast.fix_missing_locations(module)
code=compile(module,'actual-firstcycle-emitter-decision','exec')
def fixture(prefill,e2e):return {'prefill_factor_arm':prefill[0],'prefill_input_tokens_per_service_second':prefill[1],'tail_ready_metrics':{'including_prefill_useful_tokens_s':e2e}}
good=[fixture(('O8',72),12),fixture(('N4',85),13),fixture(('N4',85),13),fixture(('O8',72),12)]
for participant in range(16):
 ns={'participant':participant,'emitter':8,'report':{'multi_batch_cases':good if participant==8 else[{'tail_ready_metrics':None}]*4},'save':lambda:None}
 exec(code,ns);ok('only_emitter_reads_metrics-rank'+str(participant),ns['decision']==int(participant==8))
for label,cases,want in [('smallgain',[fixture(('O8',72),12),fixture(('N4',73),13),fixture(('N4',73),13),fixture(('O8',72),12)],0),('E2Eregression',[fixture(('O8',72),12),fixture(('N4',85),11),fixture(('N4',85),11),fixture(('O8',72),12)],0)]:
 ns={'participant':8,'emitter':8,'report':{'multi_batch_cases':cases},'save':lambda:None};exec(code,ns);ok('skip-extra-cycle-'+label,ns['decision']==want)
base=ast.parse((r/'glm_layer77_probe_r5819.py').read_text())
forward=lambda t:[ast.dump(n)for n in ast.walk(t)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_glm_model_forward']
ok('actual_native_forward_args_unmodified',forward(base)==forward(tree))
ok('feedback_decodecohort_uses_micro8',"cohorts=[list(range(a,min(a+micro,batch)))for a in range(0,batch,micro)]"in text)
ok('firsttokens_not_withheld',"for row in group:tail_ready[row].append(ready_stamp)"in text)
ok('all_firsts_before_decode_gate_retained',"assert all(len(g)==len(t)==1"in text)
ok('samehistory495_bound',"bf4f2c56a54bfd9979003f9d203fbfd20ac3f60bc621f78756667fb6e1bbe42d"in text)
names=['glm_prefill_micro_probe_r5846.py','modern_glm_prefill_micro_r5846.py','GLM_transient_preflight_r5846.py','GLM_transient_preengine_r5846.py','glm_prefill_micro_Model_r5846_inside.sh']
for name in names:
 if name.endswith('.py'):ast.parse((r/name).read_text());ok('AST:'+name,True)
ok('fixedsinglecycle', 'for repeat in(1,):'in text and "report['second_cycle_not_planned']=True"in text and 'for repeat in(1,2)'not in text)
out=r/'GLM-prefill-micro-CPU-r5846.json';assert not out.exists();out.write_text(json.dumps({'completed':True,'checks':len(checks),'passed':checks,'source_SHA256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in names},'scope':'CPU metadata and actual extracted emittercontrol; native_Model quality/TPS NOT_RUN. No Torch/native/GPU initialized.'},indent=2)+'\n',encoding='utf-8');print(out.name,len(checks),hashlib.sha256(out.read_bytes()).hexdigest())
