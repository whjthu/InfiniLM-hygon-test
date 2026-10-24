import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-quant1-r8884';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing originalcontroller/watch/results numeric summaries and SHA only; no logs, outputs, activations or weights exported','sourceclaim':'42b4b97'}
for key in ['launch','watch']:
 p=r/(label+'-'+key+'.json')
 if not p.exists():continue
 z=json.loads(p.read_text());d[key]={k:v for k,v in z.items() if k not in ['args','fresh_before_KFD','CPU_result_audit_stdout','CPU_result_audit_stderr','error_warning_lines']}
 if key=='watch' and z.get('CPU_result_audit_stdout'):
  try:d['original_audit']=json.loads(z['CPU_result_audit_stdout'])
  except ValueError:d['original_audit_parseable']=False
 if key=='watch' and z.get('actual_handle'):
  h=z['actual_handle'];p=r/'results'/f'{label}-{h}.json'
  if p.exists():
   raw=p.read_bytes();q=json.loads(raw);d['report_SHA256']=hashlib.sha256(raw).hexdigest();d['report_bytes']=len(raw);d['report']={k:q.get(k) for k in ['completed','error','tokens','routes','unique_experts','max_expert_rows','extra_source_bank_bytes','input_prepare_seconds','CPU_checkpoint_read_hash_slice_seconds','CPU_bank_bytes','driver_wall_includes_CPUweights_H2D_readback_seconds','arms','library_SHA256','CPU_receipt_SHA256','weight_keys','source_weight_manifest_SHA256','TP0_bank_SHA256']}
   d['owners']=[{'dev':o['dev'],'records':len(o.get('records',[])),'normal_owner_closed':o.get('normal_owner_closed'),'error':o.get('error'),'quant1_boundary_cases':len(o.get('quant1_boundary_actual_quality',[])),'free_after_load':o.get('free_after_load'),'observed_memory_floor_bytes':o.get('observed_memory_floor_bytes')} for o in q.get('owners',[])]
  d['owner_progress']=[]
  for op in (r/'results').glob(f'Deep-quant1-r8884-{h}-owner*.json'):
   o=json.loads(op.read_text());d['owner_progress'].append({'dev':o['dev'],'records':len(o.get('records',[])),'normal_owner_closed':o.get('normal_owner_closed'),'error':o.get('error'),'last_mode':o.get('records',[{}])[-1].get('mode') if o.get('records') else None})
  p=r/'logs'/f'{label}-{h}.log'
  if p.exists():raw=p.read_bytes();d['log_SHA256']=hashlib.sha256(raw).hexdigest();d['log_bytes']=len(raw)
print(json.dumps(d))
