from pathlib import Path
import datetime,hashlib,json,re,statistics
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-paired-native-recovery-v1');audit=json.loads((W/'independent-pair-audit.json').read_bytes());assert audit['completed'] and audit['passed'];rows=[];pins=[]
for rank in range(8):
 records=[]
 for arm in ('control','restored'):
  p=W/arm/'train/resume-witness'/('rank-%02d'%rank)/'result.json';raw=p.read_bytes();value=json.loads(raw);pin=next(r for r in audit['original_rank_records'] if r['arm']==arm and r['rank']==rank);assert hashlib.sha256(raw).hexdigest()==pin['SHA256'];records.append(value);pins.append(dict(arm=arm,rank=rank,SHA256=pin['SHA256']))
 a,b=records
 for i,(left,right) in enumerate(zip(a['outputs'][-20:],b['outputs']),1):
  x,y=left['loss'],right['loss'];assert x['tensors']==y['tensors']==1 and x['elements']==y['elements']==1 and x['finite'] and y['finite'];av,bv=x['floating_l2'],y['floating_l2'];delta=abs(av-bv);rows.append(dict(rank=rank,tail_ordinal=i,global_step_before_update=20+(i-1)//4,control_scalar_loss_magnitude=av,restored_scalar_loss_magnitude=bv,magnitude_abs_delta=delta,magnitude_relative_delta=delta/max(abs(av),1e-30),exact=x['SHA256']==y['SHA256']))
logs=[]
for arm in ('control','restored'):
 p=W/arm/'native.log';raw=p.read_bytes();assert len(raw)<4*1024**2;text=re.sub(r'\x1b\[[0-9;]*m','',raw.decode(errors='replace'));warnings=[]
 for line in text.splitlines():
  if re.search(r'Some weights .* (?:not initialized|not used|newly initialized)',line):warnings.append(dict(line_SHA256=hashlib.sha256(line.encode()).hexdigest(),tensor_names=re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_.]+)['\"]",line)[:64]))
 logs.append(dict(arm=arm,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),uninitialized_or_unused_weight_warning_count=len(warnings),warnings=warnings))
first=[r for r in rows if r['tail_ordinal']==1];before_first_update=[r for r in rows if r['tail_ordinal']<=4]
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rank_record_pins=pins,denominator=160,exact_losses=sum(r['exact'] for r in rows),loss_magnitude_abs_delta=dict(median=statistics.median(r['magnitude_abs_delta'] for r in rows),maximum=max(r['magnitude_abs_delta'] for r in rows)),loss_magnitude_relative_delta=dict(median=statistics.median(r['magnitude_relative_delta'] for r in rows),maximum=max(r['magnitude_relative_delta'] for r in rows)),first_forward_by_rank=first,before_first_optimizer_update=dict(forwards=len(before_first_update),exact=sum(r['exact'] for r in before_first_update),maximum_abs_delta=max(r['magnitude_abs_delta'] for r in before_first_update)),full160rows=rows,filtered_original_loader_logs=logs,GPU_cards=0,raw_tensor_transfer_bytes=0,qualification='Existing complete scalar loss fingerprints only; magnitudes are CPUF64 norms of scalar losses. No new model/tensor execution, original numerical tolerance or acceptance inferred.')))
