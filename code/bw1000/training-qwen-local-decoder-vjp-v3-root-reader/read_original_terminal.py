import datetime,hashlib,json,re,subprocess
from pathlib import Path
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-local-decoder-vjp-v3')
request='b094bfb95a94cdc2f9e81bb7aabf1bb4'
def read(path,cap=262144):
 try:
  with path.open('rb') as f: raw=f.read(cap+1)
 except FileNotFoundError:return None,None
 if not 0<len(raw)<=cap:return None,None
 try:return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
 except json.JSONDecodeError:return None,None
def query(*a):
 return subprocess.run(a,text=True,capture_output=True,timeout=12,check=True).stdout.splitlines()
cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%170'
steps=[line.split('|') for line in query('sacct','-j','1087656','-n','-P','--format='+cols) if line.split('|')[1] in (root.name,root.name+'-pre',root.name+'-post')]
live=query('squeue','-h','-s','-j','1087656','-o','%i')
intent,intent_pin=read(root/'intent.json')
owner,owner_pin=read(root/'keeper-owner.json')
out,out_pin=read(root/'outcome.json')
native,native_pin=read(root/'native-outcome.json')
info=None
if owner:
 try:
  fields=(Path('/proc')/str(owner['PID'])/'stat').read_text().rsplit(') ',1)[1].split()
  info=dict(live=fields[0]!='Z' and int(fields[19])==owner['startticks'],proc_state=fields[0])
 except OSError as e:info=dict(live=False,errno=e.errno)
 terminal=bool(out and not set(live).intersection(r[0] for r in steps) and all(r[2] in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED') for r in steps))
else:terminal=False
def sanitize(value):
 if not value:return None
 error=value.get('error')
 return error and dict(type=error.get('type'),stage=error.get('stage'),detail_SHA256=hashlib.sha256(error.get('message','').encode()).hexdigest())
usable={}
for name in ('native.log','keeper.log',root.name+'.log'):
 p=root/name
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-22000));data=f.read(22000).decode(errors='replace')
  hint=('AttributeError','ValueError','ImportError','Hash mismatch','Traceback','Native input differs','Capture changed decoder operand layout','Selected original decoder input shape differs',
        'NATIVE_CHECKPOINT_REENTRANT_OR_UNKNOWN','Wrong original eight-rank lease','Unexpected training','Missing actual selected decoder output',
        'Decoder capture must stop before the optimizer','Duplicate selected decoder recompute','Native diagnostic crossed','nonfinite','FileNotFoundError','PermissionError','Failed to load')
  matches=[s for s in hint if s in data]
  allowed_lines=[line.strip() for line in data.splitlines() if 0<len(line.strip())<190 and not any(c in line for c in ('/','@',chr(92))) and line.strip().startswith(('AttributeError:','ValueError:','AssertionError:','RuntimeError:','TypeError:','ImportError:','KeyError:'))]
  causes=[]
  if name=='native.log' and terminal and p.stat().st_size<=2**18:
   bounded=p.read_text(errors='replace')
   allowed=set(re.findall(r"['\"]([A-Z][A-Za-z0-9 _./-]{8,140})['\"]",(root/'src/capture_plugin.py').read_text()+(root/'src/layer_vjp.py').read_text()+(root/'src/candidate_plugin.py').read_text()))
   for line in bounded.splitlines():
    m=re.match(r"^\[rank(\d+)\]:\s*(\w+Error): (.*)$",line)
    if m:
     rank,kind,msg=m.groups()
     causes.append(dict(rank=int(rank),type=kind,cause=msg if msg in allowed else None,detail_SHA256=hashlib.sha256(msg.encode()).hexdigest()))
  usable[name]=dict(size=p.stat().st_size,classes=matches,short_errors=sorted(set(allowed_lines))[:8],native_failure_classes=causes[:30])
payload=[]
for rank in range(8):
 p=root/'train'/f'vjp-rank-{rank:02d}.json'
 v,pin=read(p)
 if v:
  if terminal:
   assert v['rank']==rank and v['completed_microbatches']==2 and v['global_step']==20 and v['optimizer_updates']==0 and v['inputs_matched']==2
   if rank in (1,4):
    private=root/'train/decoder-operands'/f'rank-{rank:02d}.pt'; item=v['private_operand']
    assert 0<private.stat().st_size==item['bytes']<=32*1024**2
    h=hashlib.sha256()
    with private.open('rb') as f:
     for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
    assert h.hexdigest()==item['SHA256']
  previous=[]
  if terminal and rank in (1,4):
   parent=root.parent
   for branch, relative in (('continuous','training-qwen-math-sdpa-recovery-pair-v1/continuous/train/continuous-tail'),
                             ('resumed','training-qwen-math-sdpa-resume20-v1/resumed/train/resume-witness')):
    file=parent/relative/f'rank-{rank:02d}'/'loss-21.json'
    original,source=read(file)
    assert original['ordinal']==v['micro'] and original['complete']
    before=original['boundaries'][f'decoder_{v["layer"]}']
    kmap={'grad_in':'grad_input','grad_out':'grad_output','output':'output','recompute_input':'recompute_input'}
    candidate={kmap[k]:value for k,value in v['tensors'].items()}
    candidate['input']=v['input_tensor']
    logical={key:(before[key]['SHA256']==doc['SHA256']) for key,doc in candidate.items()}
    layout={key:all(before[key][field]==doc[field] for field in ('shape','stride','dtype')) for key,doc in candidate.items()}
    previous.append(dict(branch=branch,receipt=source,same_logical_digest=logical,same_layout=layout))
  payload.append(dict(row=v,pin=pin,prior_capture_comparisons=previous))
if intent:assert intent['request_id']==request
post=None
if out and out['after']:
 after=out['after']
 post=dict(step=after['step'],holders=after['kfd_holder_count'],remains=after['owned_processes_remaining'],port_clear=after['master_port_available'])
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),request=request,intent_pin=intent_pin,
        controller=info,owner_pin=owner_pin,actual_steps=steps,accounting_columns=cols.split(','),terminal=terminal,
        native_started=(root/'native-intent.json').is_file(),outcome_pin=out_pin,native_pin=native_pin,
        controller_passed=(out and out['keeper_passed']),controller_error=sanitize(out),native_error=sanitize(native),
        post=post,logs=usable,ranks=payload),allow_nan=False))
