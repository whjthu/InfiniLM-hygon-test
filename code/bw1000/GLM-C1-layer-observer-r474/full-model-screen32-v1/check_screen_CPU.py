"""CPU source/caller recording for frozen33 requests; no Model/Native/SDK/GPU."""
from pathlib import Path
import ast,json,subprocess,sys
from observer_trial import schedule as base_schedule,phases as base_phases
from resident_trial import schedule,phases
from check_deadlines_CPU import verify
D=Path(__file__).resolve().parent
assert len(schedule())==33 and schedule()[0]['diagnostic']and all(not r['diagnostic']for r in schedule()[1:])and[r['case_index']for r in schedule()[1:]]==list(range(32))and len(phases())==37 and len(set(phases()))==37
for r,a in zip(schedule(),base_schedule()):assert all(r[k]==v for k,v in a.items())and r['owner_cores']==4
assert phases()==base_phases();t=(D/'owner.py').read_text();tree=ast.parse(t);calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='before_AR_continuation'];assert len(calls)==1
loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='generated'];assert len(loops)==1 and calls[0]in list(ast.walk(loops[0]));assert t.index('observer.prepare()')>t.index('load_checkpoint(ctx,')and t.index('observer.prepare()')<t.index('for ordinal,arm')
assert t.index('frame_report=observer.read()')<t.index("observer.policy(False)")and "memory('after_observer_prepare')"in t and "forward(chunk,past,'prefill')"in t and "forward([token],past,'decode')"in t and "kind=='prefill'"in t
assert "request_enter_ns=time.monotonic_ns()"in t and "begin=request_enter_ns"in t
assert "len(r['requests'])==33 and r['resets']==66"in(D/'controller.py').read_text();assert 'GLM_C1_LAYER_OBSERVER_R474=1'in(D/'inside.sh').read_text();assert 'all5_MD_equal_fresh_M'in(D/'shared.py').read_text()and "cpu['actual_CPU']==bind['actual_CPU']=='910407.1119'"in(D/'shared.py').read_text()
input=json.loads((D/'INPUT_BINDING.json').read_bytes());assert len(input['selected_ordinals'])==len(set(input['selected_ordinals']))==32 and len(input['private_plan']['requests'])==32
for r in input['private_plan']['requests']:
 assert r['max_new_tokens']==r['prefill_window']==4096 and r['prompt_tokens']+4096<=32768
 lengths=[min(512,r['prompt_tokens']-n)for n in range(0,r['prompt_tokens'],512)];assert sum(lengths)==r['prompt_tokens']and all(0<n<=512 for n in lengths)
for p in D.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
print(json.dumps(dict(passed=True,scope='CPUrecordingonly; no Native/SDK/Slurm/GPU',planned=33,screen_unique=32,barriers=37,reset_expected=66,markedonly_AR_generated_loop=True,allprompt_retained=True,singleprefill_tail_classification_fixed=True,Native_dlopen=False,deadline=verify(D))))
