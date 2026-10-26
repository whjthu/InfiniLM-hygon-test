import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='GLM-irregular-vector-r9270b';q={'launch':None,'watch':None,'Slurm':[],'report':None,'log_tail':None}
for key in ['launch','watch']:
 f=r/(label+'-'+key+'.json')
 if f.exists():q[key]=json.loads(f.read_bytes())
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%100,State%40,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30).splitlines()];q['Slurm']=[x for x in rows if x[1]in[label,'GLM-irregular-vector-keeper-r9270b',label+'-before',label+'-post']];handle=(q['watch']or{}).get('actual_handle')
if handle:
 f=r/'results'/(label+'-'+handle+'.json')
 if f.exists():
  a=json.loads(f.read_bytes());q['report']={k:a.get(k)for k in ['completed','normal','error','weight_load_seconds','load_progress_projection','free_after_load','free_after_release','routing','performance','performance_admission_passed','actual_symbol_owners','input_readonly','weight_readonly_full_byte_sum','scale_readonly_sum','actual_counter_errors','policy_closed','counts_after']};q['report']['trial_count']=len(a.get('trials',[]))
 f=r/'logs'/(label+'-'+handle+'.log')
 if f.exists():
  with f.open('rb')as h:h.seek(max(0,f.stat().st_size-5000));q['log_tail']=h.read().decode(errors='replace')
print(json.dumps(q))
