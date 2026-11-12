"""New launch plumbing only; no imports of training frameworks or accelerators."""
import ast,importlib.util,json,math
from pathlib import Path
P=Path(__file__).parent
def module(name):
 spec=importlib.util.spec_from_file_location(name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
clock=module('deadline_contract');owned=module('owned_processes');epoch=1000.0
assert clock.schedule(epoch,epoch)==dict(epoch=epoch,native_deadline=1820.,supervisor_stop=1850.,deadline=1900.)
for now in [1000.,1040.,1899.]:assert clock.remaining(epoch,1900.,now)==1900.-now
rejected=0
for e,d,n in [(0,900,1),(1000,1000,1000),(1000,1901,1001),(1000,1900,999),(1000,1900,1900),(float('nan'),1900,1100),(1000,float('inf'),1100),(1000,1900,float('nan'))]:
 try:clock.remaining(e,d,n)
 except ValueError:rejected+=1
 else:raise AssertionError('invalid original epoch accepted')
assert rejected==8
root=Path('/finite/original');args=[b'/env/bin/python',b'-B',b'-u',str(root/'src/run_resume_tail.py').encode(),str(root/'train').encode()]
assert owned.native_command(args,root)
for bad in [args[:-1],args+[b'extra'],args[:3]+[b'/foreign/src/run_resume_tail.py',args[4]],args[:4]+[b'/foreign/train']]:assert not owned.native_command(bad,root)
for p in P.rglob('*.py'):ast.parse(p.read_bytes(),filename=p.name)
native=ast.parse((P/'run_resume_tail.py').read_bytes())
calls=[n for n in ast.walk(native) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_vggt_training'];assert len(calls)==1
assert len(calls[0].args)==4 and not calls[0].keywords
claim=json.loads((P/'CLAIM.json').read_bytes());assert claim['whole_epoch_seconds']==900 and claim['native_deadline_offset']==820 and claim['supervisor_stop_offset']==850 and claim['native_GPU_launches']==claim['GPU_cards']==1 and claim['original_steps']==25 and not claim['formal_performance'] and claim['raw_export_bytes']==0
print(json.dumps(dict(passed=True,valid_clock_cases=3,invalid_clock_rejections=8,owned_command_positive=1,foreign_command_rejections=4,native_call_sites=1,GPU_imports=0)))
