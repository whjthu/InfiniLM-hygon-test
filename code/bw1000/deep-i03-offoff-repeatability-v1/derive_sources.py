"""Freeze a distinct two-request OFF/OFF control, reusing the qualified Native."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
P=G/'code/bw1000/deep-i03-c4-history-bank-alias-http-v1'
label='i03-deep-offoff-repeatability-v1';old='i03-deep-c4-alias-http-v1';changes={}
for name in ('pilot_inputs.py','pilot_observation.py','wire_client_r290.py','scoring.py','input_binding.json','split_admission.py','diagnostic_binding.py','pilot_run.py','pilot_host_gate.py','pilot_inside.sh','service_config.json'):
 raw=(P/name).read_bytes();text=raw.decode().replace(old,label).replace('43886','43888')
 if name.endswith('.py'):ast.parse(text)
 if name.endswith('.json'):json.loads(text)
 (D/name).write_bytes(text.encode());changes[name]=dict(parent_SHA256=hashlib.sha256(raw).hexdigest(),new_SHA256=hashlib.sha256(text.encode()).hexdigest(),only_label_port_replacements=True)
text=(P/'alias_policy.py').read_text(encoding='utf8')
text=text.replace('Validate C4 coverage on the owning worker and audit a bounded set of BF16 logits.','Same immutable alias-capable Native,all requests OFF; bounded repeatability audit.')
schedule='SCHEDULE = ((0, 0), (0, 1), (0, 0), (0, 1), (1, 1), (1, 0), (0, 0), (0, 1))';assert schedule in text
text=text.replace(schedule,'SCHEDULE = ((0, 0), (0, 0))')
text=text.replace('require(len(reference) == len(observed) == AUDIT_FORWARDS, "short logits audit")','require(2 <= len(reference) <= AUDIT_FORWARDS and 2 <= len(observed) <= AUDIT_FORWARDS, "missing natural-length prefill audit")')
text=text.replace('off_logits_SHA256','first_logits_SHA256').replace('on_logits_SHA256','repeat_logits_SHA256')
text=text.replace('require(expected[1] > 0 and delta == [expected] * 8, "actual eight-owner C4 coverage differs")','require(mode == 0 and delta == [expected] * 8 and expected[2:] == [0, 0], "OFF-only eight-owner C4 coverage differs")')
ast.parse(text);(D/'alias_policy.py').write_bytes(text.encode())
changes['alias_policy.py']=dict(parent_SHA256=hashlib.sha256((P/'alias_policy.py').read_bytes()).hexdigest(),new_SHA256=hashlib.sha256(text.encode()).hexdigest(),changes=['two identical GSM requests,mode0/mode0','allow natural audit length2..16,never force EOS','first/repeat metric names','require zero actual aliases and saved bytes'])
# Only control summary changes. The Native math,loader,HTTP,scorer and all input bytes are reused.
print(json.dumps(dict(derived_files=len(changes),requests=2,unique_cases=1,alias_modes=[0,0],Native_recompile=False)))
with (D/'SOURCE_DIFF_PROOF.json').open('x',encoding='utf8') as f:json.dump(dict(parent_source=str(P),files=changes,Native_recompile=False,context=32768,max_new_tokens_each=4096,scope='New immutable OFF/OFF causality control,not replay of the terminal OFF/ON optimization screen.'),f,indent=2)
