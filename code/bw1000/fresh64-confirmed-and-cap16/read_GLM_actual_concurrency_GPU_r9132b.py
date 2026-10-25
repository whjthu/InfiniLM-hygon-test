import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in ['GLM-G98A-r9132b-launch.json','results/GLM-G98A-r9132b-watch.json']:
 p=r/name
 if p.exists():q[name]=json.loads(p.read_text())
q['sacct']=[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed,NodeList%100'],text=True).splitlines()if 'G98A'in x];q['logs']={p.name:p.read_text(errors='replace')[-3000:]for p in(r/'logs').glob('GLM-G98A-r9132b-*.log')};run=q.get('GLM-G98A-r9132b-launch.json',{}).get('run');q['owners']=[]
if run:
 for stage in [0,1]:
  for rank in range(8):
   p=r/'results'/f'glm-actual-concurrency-r9132b-{run}-stage{stage}-rank{rank}.json';phase=r/'results'/f'GLM-Model-phase-r9132b-{run}-stage{stage}-rank{rank}.jsonl';v={'stage':stage,'rank':rank}
   if phase.exists():v['last_phase']=json.loads(phase.read_text().splitlines()[-1])
   if p.exists():
    d=json.loads(p.read_text());v['report']={k:d.get(k)for k in ['completed','actual_producer','scope','remaining_process_seconds','benchmark_complete','budget_partial','MMLU_progress','service_progress','minimum_driver_free_bytes','actual_owner_weight_load_wall_seconds','worker_shutdown_completed','post_close_Torch_liveness']};v['finished_service']=[{k:a.get(k)for k in ['label','C','M','ON','native_prefill_seconds','native_prefill_input_tok_s','native_decode_useful_tok_s','decode_common_window_useful_tok_s','cohort_counter_delta','budget_stop']}for a in d.get('service',[])];
    if stage==1 and rank==0:v['MMLU']=[{'ON':a['ON'],'cases':len(a['cases']),'correct':sum(x['flexible_correct']for x in a['cases'])}for a in d.get('MMLU',[])];v['paired_MMLU']=d.get('paired_MMLU')
   q['owners'].append(v)
print(json.dumps(q))
