import pathlib,json,subprocess,hashlib,datetime,re
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in ['GLM-G100A-r9136-launch.json','results/GLM-G100A-r9136-watch.json']:
 p=r/name
 if p.exists():q[name]=json.loads(p.read_text())
q['sacct']=[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed,NodeList%100'],text=True).splitlines()if 'G100A'in x];q['logs']={p.name:p.read_text(errors='replace')[-3000:]for p in(r/'logs').glob('GLM-G100A-r9136-*.log')};run=q.get('GLM-G100A-r9136-launch.json',{}).get('run');q['owners']=[]
if run:
 for stage in [0,1]:
  for rank in range(8):
   p=r/'results'/f'glm-actual-concurrency-r9136-{run}-stage{stage}-rank{rank}.json';phase=r/'results'/f'GLM-Model-phase-r9136-{run}-stage{stage}-rank{rank}.jsonl';v={'stage':stage,'rank':rank}
   if phase.exists():v['last_phase']=json.loads(phase.read_text().splitlines()[-1])
   if p.exists():
    d=json.loads(p.read_text());v['report']={k:d.get(k)for k in ['completed','actual_producer','scope','remaining_process_seconds','benchmark_complete','budget_partial','MMLU_progress','service_progress','minimum_driver_free_bytes','actual_owner_weight_load_wall_seconds','worker_shutdown_completed','post_close_Torch_liveness']};v['finished_service']=[{k:a.get(k)for k in ['label','C','M','ON','native_prefill_seconds','native_prefill_input_tok_s','native_decode_useful_tok_s','decode_common_window_useful_tok_s','cohort_counter_delta','ragged_counter_delta','ragged_Bhist_delta','budget_stop']}for a in d.get('service',[])];
    if stage==1 and rank==0:v['MMLU']=[{'ON':a['ON'],'cases':len(a['cases']),'correct':sum(x['flexible_correct']for x in a['cases'])}for a in d.get('MMLU',[])];v['paired_MMLU']=d.get('paired_MMLU')
   q['owners'].append(v)
q['actual_network_by_node']=[];q['actual_loader_progress']=[]
for node in [0,1]:
 p=r/'logs'/f'GLM-G100A-r9136-910376.694-{node}.log'
 if not p.exists():continue
 raw=p.read_bytes();txt=raw.decode(errors='replace');using_ib=txt.count('Using network IBext_v8');using_socket=txt.count('Using network Socket');gdr=[line for line in txt.splitlines()if 'GDRDMA'in line and('via NET/'in line or'Channel'in line)]
 q['actual_network_by_node'].append({'node_task_index':node,'path':str(p),'snapshot_SHA256':hashlib.sha256(raw).hexdigest(),'actual_Using_network_IBext_v8_count':using_ib,'actual_Using_network_Socket_data_count':using_socket,'bootstrap_Socket_not_fallback':True,'GDR_status':'ACTUAL_CHANNEL_GDRDMA_LOGGED'if gdr else'UNKNOWN','actual_GDR_channel_lines_count':len(gdr)})
 last={}
 for m in re.finditer(r'GLM78_LOAD_PROGRESS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)',txt):
  stage,rank,loaded,total,seconds=m.groups();last[rank]={'stage':int(stage),'rank':int(rank),'loaded_parameter_keys':int(loaded),'total_parameter_keys':int(total),'reported_load_seconds':float(seconds),'key_count_notbyteprogress_orETA':True}
 q['actual_loader_progress'].extend(last.values())
print(json.dumps(q))
