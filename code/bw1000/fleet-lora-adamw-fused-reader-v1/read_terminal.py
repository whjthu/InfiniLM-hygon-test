import datetime,json,os,re,subprocess,time
from pathlib import Path
request='d8175ef68aaf4b2594f6b15ff98c26ce'
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/('fleet-lora-adamw-fused-v1-'+request)
def read(path):
 p=root/path
 if not p.is_file():return None
 raw=p.read_bytes()
 if len(raw)>128<<10:raise ValueError('bounded actual result')
 return json.loads(raw)
def query(*args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=8)
 return [p.returncode,p.stdout.splitlines()]
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request=request,
         intent=bool(read('intent.json')),submitted=bool(read('submitted.json')),runs={})
intent=read('intent.json')
if intent:
 out['elapsed']=round(time.time()-intent['epoch'],1)
 for parent,label in intent['labels'].items():
  rc,rows=query('sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160')
  matches=[row.split('|') for row in rows if row.split('|')[1:2]==[label]]
  out['runs'][parent]=dict(accounting_rc=rc,rows=matches)
 from collections import Counter
 from statistics import median
 errors=Counter()
 performance={}
 observed={}
 inv=ready=finished=clean_nodes=0
 approved_steps={p:rows['rows'][0][0] for p,rows in out['runs'].items() if rows['accounting_rc']==0 and len(rows['rows'])==1}
 parent_for_node={node:parent for parent,nodes in intent['parents'].items() for node in nodes}
 for nodes in intent['parents'].values():
  for node in nodes:
   relative=Path('nodes')/node
   inv+=bool(read(relative/'inventory.json'))
   node_terminal=read(relative/'terminal.json')
   if node_terminal:
    if (node_terminal['node']==node and node_terminal['step']==approved_steps.get(parent_for_node[node]) and node_terminal['error'] is None and node_terminal['post_kfd_holders']==0 and set(node_terminal['exits'])=={str(i) for i in range(8)} and set(node_terminal['exits'].values())=={0}):clean_nodes+=1
    else: errors['NODE_TERMINAL_REJECTED']+=1
   admit=read(relative/'admission-error.json')
   if admit:errors[admit['cause']]+=8
   for card in range(8):
    ready+=bool(read(relative/f'active-{card}.json'))
    receipt=read(relative/f'card-{card}.json')
    if receipt:
     assert receipt['step']==approved_steps.get(parent_for_node[node]) and receipt['node']==node and receipt['card']==card
     finished+=1
     if receipt['error']:errors[receipt['error']['kind']+':'+receipt['error']['stage']]+=1
     else:
      (measure,)=receipt['results']
      assert measure['numeric_updates']==3 and 0<=measure['numeric_max_abs']<=5e-5 and all(measure['updated_elements'])
      blocks=measure['blocks']
      assert [b['fused'] for b in blocks]==[False,True,True,False]
      for key in ('optimizer_ms','step_ms'):
       baseline=median(b[key] for b in blocks if not b['fused'])
       candidate=median(b[key] for b in blocks if b['fused'])
       if baseline>0:performance.setdefault(key,[]).append(candidate/baseline)
   for offset in (120,240,380,520,660):
    name='busy.json' if offset==120 else f'busy-{offset}.json'
    points=read(relative/name)
    if points:
     rows=points['rows']
     pci=set(read(relative/'inventory.json')['PCI_KFD'])
     assert points['node']==node and len(rows)==3 and len(pci)==8
     assert all(set(row['busy'])==pci and row['point']==i and intent['epoch']+offset+i*2 <= row['opened_epoch'] <= row['closed_epoch'] <= intent['epoch']+offset+i*2+1 and all(type(value) is int and 0 <= value <= 100 for value in row['busy'].values()) for i,row in enumerate(rows))
     observed.setdefault(offset,[]).append(rows)
 def samples(node_rows):
  return [value for x in node_rows for row in x for value in row['busy'].values()]
 out.update(inventoried=inv*8,real_numeric_updates=ready,finished_card_receipts=finished,clean_terminal_nodes=clean_nodes,errors=dict(errors),
            sampled_node_counts={k:len(v) for k,v in observed.items()})
 if observed:
  for offset,node_rows in observed.items():
   values=samples(node_rows)
   out.setdefault('busy',{})[offset]=dict(card_samples=len(values),mean=sum(values)/len(values),nonzero_samples=sum(v>0 for v in values),cards=len(node_rows)*8,card_means_strict_gt80=sum(sum(row['busy'][pci] for row in rows)>240 for rows in node_rows for pci in rows[0]['busy']))
 out.update(terminal=read('dispatch-terminal.json'),fused_over_foreach={k:dict(cards=len(v),median=median(v)) for k,v in performance.items()})
print(json.dumps(out))
