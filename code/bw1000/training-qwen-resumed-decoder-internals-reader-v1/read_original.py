import datetime,hashlib,json,pathlib,re,subprocess
label='training-qwen-resumed-decoder-internals-v1'
request='ea861f52e41c471a9f2d9d3c08fff98d'
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
 previous=root.with_name('training-qwen-first-resumed-forward-v1')
 earlier_pins={"continuous/train/continuous-tail/rank-02/loss-21.json":{"bytes":10423,"SHA256":"d321f7358805f0da933f46a58d92491f70fa54018688e0ad26afde0a0609e782"},"continuous/train/continuous-tail/rank-02/result.json":{"bytes":12487,"SHA256":"3ec428506fdb764c608ba485fba3c4cc5c8a729f1f3f8831fb29f4150f26574d"},"resumed/train/resume-witness/rank-02/loss-21.json":{"bytes":10423,"SHA256":"2c68a3161ce302e120ad1777a4cafe1e2cb23e7c5eaeed06dc9658b56c3d45fb"},"resumed/train/resume-witness/rank-02/result.json":{"bytes":24154,"SHA256":"3f4e8f3a67ebd2da25b84a79742b48cad5310594edde39a0516d2342aecaf0c2"},"continuous/train/continuous-tail/rank-06/loss-21.json":{"bytes":10423,"SHA256":"06a19756d005bb9d41a0fb91e2ebd32586aca0c40f8bd4aa8c86f4d52d621161"},"continuous/train/continuous-tail/rank-06/result.json":{"bytes":12485,"SHA256":"8b0748186fafc7f831cc269f6e807210cfc3345e8d06a2be51bb9af3a55e4d6b"},"resumed/train/resume-witness/rank-06/loss-21.json":{"bytes":10423,"SHA256":"96becfc24670172f9bd14250ad3a25d78db1fe32d6f14fd28daa745ac0fea8e0"},"resumed/train/resume-witness/rank-06/result.json":{"bytes":24151,"SHA256":"371d6c13e1e270534f3ef0c22ee55ac64a6c1f72bf97e92103c64e2290121aa1"}}
 verified={}
 ordered=['decoder_00_input',*[f'decoder_{i:02d}' for i in range(36)],'final_norm','lm_head']
 for rank in (2,6):
  docs=[]; input_data=[]; losses=[]; old_match=[]; entry_rng=[]; actual_orders=[]
  for phase,folder in [('continuous','continuous-tail'),('resumed','resume-witness')]:
   prefix=f'{phase}/train/{folder}/rank-{rank:02d}'
   trace=read(prefix+'/loss-21.json',131072)
   assert trace['complete'] and trace['ordinal']==1 and trace['rank']==rank
   assert trace['selected_decoder']=={2:26,6:5}[rank]
   order=trace['capture_order']
   assert len(order)==len(set(order))==78 and [n for n in order if n in ordered]==ordered
   assert list(trace['boundaries'])==order
   actual_orders.append(order)
   boundary=trace['boundaries']
   assert all(t is None or re.fullmatch('[0-9a-f]{64}',t['SHA256']) and 0<t['logical_bytes']<=768*1024**2 for t in boundary.values())
   docs.append(boundary)
   summary=read(prefix+'/result.json',131072)
   update=read(prefix+'/update-21.json')
   old_path=previous/prefix/'result.json'
   raw=old_path.read_bytes()
   assert len(raw)<=131072
   old=json.loads(raw)
   pins['prior/'+prefix+'/result.json']=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
   assert pins['prior/'+prefix+'/result.json']==earlier_pins[prefix+'/result.json']
   old_raw=(previous/prefix/'loss-21.json').read_bytes()
   pins['prior/'+prefix+'/loss-21.json']=dict(bytes=len(old_raw),SHA256=hashlib.sha256(old_raw).hexdigest())
   assert pins['prior/'+prefix+'/loss-21.json']==earlier_pins[prefix+'/loss-21.json']
   old_trace=json.loads(old_raw)
   assert old_trace['capture_order']==ordered and old_trace['complete'] is True
   prior_coarse_changes=[n for n in ordered if old_trace['boundaries'][n]['SHA256'] != boundary[n]['SHA256']]
   assert summary['rank']==old['rank']==rank and summary['inputs'][0]['global_step']==old['inputs'][0]['global_step']==20
   assert summary['inputs'][0]['ordinal']==old['inputs'][0]['ordinal']==1
   input_data.append(summary['inputs'][0]['full_input']['SHA256'])
   loss=summary['outputs'][0]['loss']['SHA256'];losses.append(loss)
   old_match.append(dict(input_match=summary['inputs'][0]['full_input']['SHA256']==old['inputs'][0]['full_input']['SHA256'],
                         loss_match=loss==old['outputs'][0]['loss']['SHA256'],
                         same_coarse_slots=39-len(prior_coarse_changes),
                         first_coarse_change_from_prior=prior_coarse_changes[0] if prior_coarse_changes else None))
   start=[x['value'] for x in update['events'] if x['boundary']=='entry-1']
   assert len(start)==1
   entry_rng.append(start[0]['SHA256'])
  changed=[]; layout=[]; optional=[]
  assert actual_orders[0]==actual_orders[1]
  for name in actual_orders[0]:
   a,b=(d[name] for d in docs)
   if a is None or b is None:
    if a!=b: changed.append(name)
    else: optional.append(name)
    continue
   if a['SHA256']!=b['SHA256']:changed.append(name)
   if any(a[k]!=b[k] for k in ('shape','dtype','stride','storage_offset')):layout.append(name)
  coarse=[n for n in changed if n in ordered]
  internal=[n for n in changed if n not in ordered]
  verified[str(rank)]=dict(same_input=input_data[0]==input_data[1],same_entry_rng=entry_rng[0]==entry_rng[1],
                           same_first_loss=losses[0]==losses[1],selected_decoder={2:26,6:5}[rank],
                           selected_decoder_input_matches='selected.input' not in changed,
                           old_phase_match=old_match,first_comparable_decoder_boundary=coarse[0] if coarse else None,
                           first_selected_internal_difference=internal[0] if internal else None,
                           changed_logical_boundaries=changed,changed_layout_boundaries=layout,
                           absent_optional_in_both=optional,complete_boundaries=78)

 out['forward_boundaries']=verified
out['pins']=pins
print(json.dumps(out,allow_nan=False))
