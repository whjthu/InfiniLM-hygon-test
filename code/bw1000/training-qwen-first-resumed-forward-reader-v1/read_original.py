import datetime,hashlib,json,pathlib,re,subprocess
label='training-qwen-first-resumed-forward-v1'
request='5b79632f28954cdb821d0a39b58456e9'
parent='1087656'
root=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/label
t='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%170'
p=subprocess.run(['sacct','-n','-P','-j',parent,'--format='+t],capture_output=True,text=True,timeout=12)
rows=[x.split('|') for x in p.stdout.splitlines() if x.split('|')[1:2] and x.split('|')[1] in {label,label+'-pre',label+'-post'}]
out=dict(request=request,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting_rc=p.returncode,accounting_columns=t.split(','),rows=rows,
         namespaced=root.is_dir(),original_controller_present=False)
pins={}
def read(name,limit=1048576):
 path=root/name
 try:
  with path.open('rb') as f: b=f.read(limit+1)
 except FileNotFoundError:
  return None
 assert len(b)<=limit
 try: d=json.loads(b)
 except json.JSONDecodeError:return None
 pins[name]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
 return d
i=read('intent.json',16384)
k=read('keeper-owner.json',16384)
v=read('controller-ready.json',16384)
if i:
 assert i['request_id']==request and i['resource']['node']=='a01r4n16'
 out['intent']={key:i[key] for key in ('epoch','deadline','claim_commit','source_manifest_SHA256','resource')}
if k:
 assert i and k['request_id']==request and i['epoch']==k['epoch']
 boot=hashlib.sha256(pathlib.Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()
 out['owner_boot_identity_matches']=boot==k['boot_id_SHA256']
 if out['owner_boot_identity_matches']:
  try:
   stat=(pathlib.Path('/proc')/str(k['PID'])/'stat').read_text().split(') ',1)[1].split()
   out['original_controller_present']=int(stat[19])==k['startticks'] and int(stat[2])==k['PGID'] and stat[0]!='Z'
  except FileNotFoundError: pass
out['successful_source_asset_gate']=bool(read('source-and-asset-preflight.json',16384))
b=read('before.json',131072)
a=read('after.json',131072)
out['before_is_clean']=bool(b and b['node']=='a01r4n16' and len(b['pci'])==8 and b['kfd_holder_count']==0)
out['after_is_clean']=bool(a and b and a['node']==b['node'] and a['pci']==b['pci'] and a['boot_id']==b['boot_id']
                       and a['kfd_holder_count']==0 and a['owned_processes_remaining']==[])
native=read('native-outcome.json')
host=read('outcome.json')
def error(e):
 if not e: return None
 detail=str(e.get('message',''))
 tags=['deterministic','nondeterministic','CUBLAS','hipBLAS','out of memory','timed out','checkpoint','asset','ownership','provider','runtime','does not have a deterministic implementation','does not have a deterministic']
 return dict(type=e.get('type'),stage=e.get('stage'),detail_SHA256=hashlib.sha256(detail.encode()).hexdigest(),
             tags=[x for x in tags if x.lower() in detail.lower()])
out['host']=dict(passed=host['keeper_passed'],error=error(host.get('error')),GPU_step_requested=host['GPU_step_requested']) if host else None
out['pair']=dict(normal=native['normal_native_exit'],error=error(native.get('error')),
                 phases={x:dict(normal=v['normal'],returncode=v['returncode']) for x,v in native['phases'].items()}) if native else None
progress={}
for phase in ('continuous','resumed'):
 path=root/phase
 ck=[]
 for n in (10,20,25) if phase=='continuous' else (25,):
  state=read(f'{phase}/train/checkpoint-{n}/trainer_state.json')
  if state is not None and state.get('global_step')==n:ck.append(n)
 no=read(f'{phase}/native-outcome.json')
 row=dict(branch_created=path.is_dir(),completed_checkpoint_states=ck,
          rank_policy_receipts=sum((path/f'train/math-sdpa-rank-{n:02d}.json').is_file() for n in range(8)),
          native_log_present=(path/'native.log').is_file())
 if no:
  row.update(exit=no['native_returncode'],normal=no['normal_native_exit'],
             native_error=error(no.get('error')),rank_record_counts=no['rank_record_counts'])
 log=path/'native.log'
 if log.is_file():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-131072))
   end=f.read(131073)
  decoding=end.decode(errors='replace')
  allowed=['does not have a deterministic implementation', 'does not have a deterministic', 'deterministic algorithms', 'CUBLAS_WORKSPACE_CONFIG',
           'out of memory','Traceback', 'Watchdog','timed out','RuntimeError','ValueError','TypeError', 'AttributeError',
           'ModuleNotFoundError','IndexError','Error:','NCCL']
  # Whitelist only operation identifiers in actual determinism exceptions; remaining messages are private.
  row['last_update_progress']=[int(n) for n in re.findall(r'([0-9]+)\/25',decoding)[-4:] if int(n)<=25]
  row['failure_tail_classes']=[x for x in allowed if x.lower() in decoding.lower()]
  row['deterministic_operation_names']=sorted(set(re.findall(r'([A-Za-z0-9_]+(?:_cuda|_out|_backward|_kernel)) does not have a deterministic',decoding)))[:12]
  row['failure_tail_SHA256']=hashlib.sha256(end).hexdigest()
 progress[phase]=row
out['progress']=progress
out['selected_trace_present']={
 phase:[rank for rank in (2,6)
        if (root/phase/'train'/folder/f'rank-{rank:02d}'/'loss-21.json').is_file()]
 for phase,folder in [('continuous','continuous-tail'),('resumed','resume-witness')]}
if native and native['result']:
 r=native['result']
 out['comparison']=dict(input_all=r['all_input_fingerprints_equal'],loss_all=r['all_loss_fingerprints_equal'],
                        compared=sum(x['compared_microbatches'] for x in r['ranks']),
                        matched_inputs=sum(x['matched_inputs'] for x in r['ranks']),matched_losses=sum(x['matched_losses'] for x in r['ranks']),
                        checkpoint25_equal=r['checkpoint25_equal'],
                        per_rank_policy=[dict(rank=x['rank'],phase=y['phase'],strict=y['deterministic'] and not y['warn_only'],
                                              previously_deterministic=y['previously_deterministic']) for x in r['ranks'] for y in x['route'].values()])
if native and native['result']:
 previous=root.with_name('training-qwen-deterministic-math-recovery-v1')
 verified={}
 ordered=['decoder_00_input',*[f'decoder_{i:02d}' for i in range(36)],'final_norm','lm_head']
 for rank in (2,6):
  docs=[]; input_data=[]; losses=[]; old_match=[]; entry_rng=[]
  for phase,folder in [('continuous','continuous-tail'),('resumed','resume-witness')]:
   prefix=f'{phase}/train/{folder}/rank-{rank:02d}'
   trace=read(prefix+'/loss-21.json',131072)
   assert trace['complete'] and trace['ordinal']==1 and trace['rank']==rank
   assert trace['capture_order']==ordered and list(trace['boundaries'])==ordered
   boundary=trace['boundaries']
   assert all(re.fullmatch('[0-9a-f]{64}',t['SHA256']) and 0<t['logical_bytes']<=768*1024**2 for t in boundary.values())
   docs.append(boundary)
   summary=read(prefix+'/result.json',131072)
   update=read(prefix+'/update-21.json')
   old_path=previous/prefix/'result.json'
   raw=old_path.read_bytes()
   assert len(raw)<=131072
   old=json.loads(raw)
   pins['prior/'+prefix+'/result.json']=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
   assert summary['rank']==old['rank']==rank and summary['inputs'][0]['global_step']==old['inputs'][0]['global_step']==20
   assert summary['inputs'][0]['ordinal']==old['inputs'][0]['ordinal']==1
   input_data.append(summary['inputs'][0]['full_input']['SHA256'])
   loss=summary['outputs'][0]['loss']['SHA256'];losses.append(loss)
   old_match.append(dict(input_match=summary['inputs'][0]['full_input']['SHA256']==old['inputs'][0]['full_input']['SHA256'],
                         loss_match=loss==old['outputs'][0]['loss']['SHA256']))
   start=[x['value'] for x in update['events'] if x['boundary']=='entry-1']
   assert len(start)==1
   entry_rng.append(start[0]['SHA256'])
  changed=[]; layout=[]
  for name in ordered:
   a,b=(d[name] for d in docs)
   if a['SHA256']!=b['SHA256']:changed.append(name)
   if any(a[k]!=b[k] for k in ('shape','dtype','stride','storage_offset')):layout.append(name)
  verified[str(rank)]=dict(same_input=input_data[0]==input_data[1],same_entry_rng=entry_rng[0]==entry_rng[1],
                           same_first_loss=losses[0]==losses[1],
                           old_phase_match=old_match,first_forward_difference=changed[0] if changed else None,
                           changed_logical_boundaries=changed,changed_layout_boundaries=layout,complete_boundaries=39)
 out['forward_boundaries']=verified
out['pins']=pins
print(json.dumps(out,allow_nan=False))
