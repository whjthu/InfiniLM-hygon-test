"""Independent CPU terminal audit. Reuse supervisor; never submits GPU/cleanup."""
import pathlib,json,hashlib,datetime
from kimi_matched_trace_r8467 import validate_records
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';issues=[]
d=json.loads((r/'Kimi-matched-r8474-launch.json').read_text());run=d['run']
w=json.loads((r/'Kimi-matched-r8474-watch.json').read_text())
def actual_host_path(value):
 p=pathlib.PurePosixPath(value)
 if str(p).startswith('/refactor/'):
  rel=p.relative_to('/refactor')
  if '..' in rel.parts:raise ValueError('invalid container path')
  return r.joinpath(*rel.parts)
 return pathlib.Path(value)
reports=[];hashes={}
for n in range(4):
 p=r/'results'/f'kimi-matched-r8472-{run}-node{n}.json'
 if not p.exists():issues.append('missing_owner'+str(n));reports.append({});continue
 q=json.loads(p.read_text());reports.append(q);hashes[str(n)]=hashlib.sha256(p.read_bytes()).hexdigest()
 if not(q.get('completed') and q.get('all8_owned_workers_shutdown_completed') and q.get('historical_B16M4_anchor_passed') and q.get('before8QA_passed')):issues.append('owner_or_anchor'+str(n))
 if len(q.get('QA',[]))!=16 or len({(v['epoch'],v['task'])for v in q['QA']})!=16:issues.append('QA'+str(n))
 if q.get('library_sha256')!='9001bf2bfa7e0821d52d9bec61ed5d9bf028c93380673fa880af3ed52e9c7388':issues.append('DSO'+str(n))
 if not q.get('matched_trace_result',{}).get('qualified') or not q.get('arrival_reference_qualification',{}).get('qualified') or not q.get('live_trace_matched_CPU_oracle_qualified'):issues.append('matched_or_live'+str(n))
q=reports[0];live=next((v for v in q.get('actual_arrival_epochs',[])if not v['full_logit_reference']),None)
if live is None:issues.append('no_live_epoch')
else:
 raw=actual_host_path(live['capture_path']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=live['capture_SHA256']:issues.append('live_capture_SHA')
 jobs=[json.loads(s)for s in raw.decode().splitlines()];checks=q.get('live_oracle_checks',[])
 if len(checks)!=len(jobs):issues.append('live_oracle_count')
 for a,b in zip(jobs,checks):
  oracle=b['reference_oracle']
  if a['job']['seq']!=b['seq'] or a['job']['rows']!=b['rows'] or a['actual_ids']!=b['live_actual_ids'] or a['actual_ids']!=oracle['cpu_argmax_ids'] or not(oracle['finite'] and oracle['native_equals_CPU']):issues.append('live_oracle_ID')
 for output in live['actual_outputs']:
  if not output or not(len(output)==128 or output[-1]in(163585,163586)) or any(t in(163585,163586)for t in output[:-1]):issues.append('EOS')
 for key in ['matched_trace_source']:
  p=actual_host_path(q[key]);raw=p.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=q['matched_trace_SHA256']:issues.append('static_reference_SHA')
  v=json.loads(raw);validate_records(v['header'],v['records'])
if w.get('terminal',{}).get('State')!='COMPLETED' or w.get('terminal',{}).get('ExitCode')!='0:0' or not w.get('post_KFD_passed'):issues.append('terminal_KFD')
out=dict(auditor_revision=8479,original_watch_audit_exit=w.get('audit_exit'),path_fix_only='/refactor/ maps to exact original hostroot, all numeric checks unchanged',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),qualified=not issues,issues=sorted(set(issues)),scope='MATCHED_SHAPE_JOBTRACE_ONLY; real boundedlive71 coordinatorfeedback metrics separately; noHTTP/clientdelivery/ABBAgain/infinite service claim',performance_adopted=False,live_metric=live['metric']if live else None,report_SHA256=hashes,terminal=w.get('terminal'),post_KFD_passed=w.get('post_KFD_passed'))
p=r/'results'/f'Kimi-matched-r8479-audit-{run}.json'
with p.open('x')as f:json.dump(out,f,indent=2)
print(json.dumps(dict(path=str(p),qualified=out['qualified'],issues=out['issues'])))
