"""Offline CPU tokenizer fixture; no Model/CUDA/weight tensor operations."""
import pathlib,json,hashlib,os,datetime,sys
assert all(os.environ.get(k,'')==''for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
from glm_chat_template_r20 import prepare_tokenizer,normalize_ids
r=pathlib.Path('/refactor');out=r/'results/glm-I512x32-real-inputs-r5732.json';receipt=r/'results/glm-I512x32-real-inputs-receipt-r5732.json';assert not out.exists()and not receipt.exists()
tokenizer,kwargs,provenance=prepare_tokenizer()
document='''The regional library operates a main building and six neighborhood branches. Its delivery team moves returned books, reserved items, teaching kits, and accessible reading equipment between those sites. Public opening hours differ by branch, so the team plans routes around receiving windows rather than assuming every desk is staffed at the same time. Staff can ask for an urgent transfer, but the dispatcher must record the reason and the effect on the regular route.
The main building receives most returned items during the afternoon. A small sorting area separates reservations from ordinary returns and damaged items. Reservation notices should be sent only after the destination branch confirms receipt. If a box contains a mixture of ordinary returns and reservations, the branch scans the reservation items first. The daily report distinguishes a missing scan from a confirmed missing book; staff should not treat those events as equivalent.
Two vans are available for planned routes, while a third vehicle is shared with a community education program. That shared vehicle cannot be promised for the whole week. Drivers check fuel, lights, and cargo restraints before departure and report faults to the dispatcher. The route plan includes a recovery option if a van becomes unavailable. Borrowing an additional vehicle requires approval, and staff must record any change in insurance or loading restrictions before using it.
The team has a fixed staffing budget. Several experienced staff members know the routes well, but newer colleagues need written instructions and supervised practice. Training must cover safe lifting, handling of accessible equipment, the meaning of scan statuses, and the process for reporting a discrepancy. A short checklist helps people complete routine steps, but it does not replace a named person who can resolve an exception. Supervisors should review recurring exceptions before assuming the problem is individual performance.
The library wants to reduce waiting time without increasing losses or making promises that cannot be kept. Its current dashboard shows completed transfers and late arrivals but does not separate delays caused by sorting, dispatch, travel, and branch receipt. A useful pilot would define those stages consistently, compare similar working days, and record unusually large events such as school visits. The pilot should be small enough that staff can explain what changed and reverse an unsuccessful procedure.
The dispatcher also coordinates scheduled maintenance and public events. A school visit may create a temporary surge in teaching kits, while a building repair can close a loading entrance. Those exceptions belong in the route plan before departure. Staff should contact the destination branch when a receiving window becomes unrealistic, give a revised estimate, and record whether the branch accepted it. Repeatedly changing an estimate without recording the cause makes the final dashboard difficult to interpret.
The proposed review has three stages: confirm the data definitions, run a bounded pilot, and decide whether to expand it. Any recommendation should name an owner, a measurable outcome, and a condition for stopping or revising the change. Improvements in average delivery time should be checked alongside reliability and staff workload. A process that looks faster only because exceptions are no longer recorded would not satisfy the review.'''
focus=['service reliability','dispatch ownership','scan accuracy','driver training','vehicle contingency planning','branch receiving windows','pilot design','exception reporting','staff workload','maintenance coordination','reservation notices','dashboard definitions','safe handling','communications','reversible rollout','measurement fairness']
focus+=['route recovery','maintenance scheduling','priority transfer triage','accessible equipment handling','record ownership','lost-item escalation','event surge planning','shift handover','loading entrance changes','route comparison fairness','exception review cadence','staff practice and supervision','dashboard auditability','branch confirmation duties','vehicle availability promises','pilot stopping criteria']
old_path=r/'results/glm-I512-real-inputs-r5713.json';assert hashlib.sha256(old_path.read_bytes()).hexdigest()=='2f1076c4933394579097fc58fb399b53667f11cbce81a06d263aa5b4e2d83bbc'
old=json.loads(old_path.read_text());assert old['provenance']==provenance
words=document.split();cases=list(old['cases'])
for row,topic in enumerate(focus[16:],start=16):
 def encode(n,variant=0):
  excerpt=' '.join(words[:n]).rstrip('.,;:')+'.'
  lead=f'This is a synthetic operations-review scenario, case {row:02d}. Read the following bounded excerpt; do not infer omitted details.\n\n'
  question=[f'Using only this excerpt, write a detailed review focused on {topic}. Give six numbered recommendations, explain the reason for each, and identify missing information instead of inventing facts.',f'Based only on the excerpt, assess {topic}. Provide six practical recommendations with reasons, owners, and checks for success. State where the note does not provide enough evidence.',f'Prepare a structured review of {topic} from the excerpt. Explain the main risks and give at least six justified recommendations. Keep assumptions separate from the recorded facts.'][variant]
  text=lead+excerpt+'\n\n'+question
  messages=[{'role':'user','content':text}]
  ids=normalize_ids(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**kwargs))
  return ids,text,messages
 found=None
 for variant in range(3):
  low,high=1,len(words)
  while low<high:
   mid=(low+high)//2
   if len(encode(mid,variant)[0])<512:low=mid+1
   else:high=mid
  for n in range(max(1,low-80),min(len(words),low+80)+1):
   ids,text,messages=encode(n,variant)
   if len(ids)==512:found=(ids,text,messages,n,variant);break
  if found:break
 assert found is not None,('no exact512 coherentexcerpt',row)
 ids,text,messages,n,variant=found;rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
 assert len(ids)==512 and all(0<=v<154880 for v in ids) and ids[:2]==[154822,154824] and 154827 in ids and 154828 in ids
 cases.append({'logical_row':row,'topic':topic,'text':text,'rendered':rendered,'input_ids':ids,'input_tokens':len(ids),'input_ids_SHA256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'excerpt_words':n,'question_variant':variant,'generation_starts_in_thinking':rendered.rstrip().endswith('<think>')})
assert len(cases)==32 and len(set(tuple(c['input_ids'])for c in cases))==32
if'torch'in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Synthetic coherentEnglish documentreview inputs; exact512actualchat-template tokens each, noPAD/randompadding/Modelexecution. NaturalEOS generation max128 is a futureModel test, not run.','completed':True,'cases':cases,'total_effective_input_tokens':16384,'max_output_tokens':128,'capacity':640,'provenance':provenance,'tokenizer_json_SHA256':hashlib.sha256(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/tokenizer.json').read_bytes()).hexdigest(),'source_SHA256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'first16_source_fixture_SHA256':hashlib.sha256(old_path.read_bytes()).hexdigest(),'padding_requested':False,'CUDA_initialized':False}
out.write_text(json.dumps(q,ensure_ascii=False,indent=2));summary={k:v for k,v in q.items()if k!='cases'};summary.update(fixture=str(out),fixture_SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),input_lengths=[c['input_tokens']for c in cases],unique_sequences=32,excerpt_words=[c['excerpt_words']for c in cases]);receipt.write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary))
