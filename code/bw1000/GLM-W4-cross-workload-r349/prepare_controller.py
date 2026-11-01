"""Minimal308 controller extension, original namespace/DSO proofs keep scope."""
import pathlib,json,hashlib
D=pathlib.Path(__file__).parent;P=D.parent/'public-native-bridge/GLM-prefill-W2-W4-controller-r308';O=D/'controller';O.mkdir(exist_ok=True)
old=json.loads((P/'SOURCE_MANIFEST.json').read_bytes());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for n,h in old['files_SHA256'].items():
 assert sha(P/n)==h;(O/n).write_bytes((P/n).read_bytes())
(O/'parent_candidate_binding_r308.py').write_bytes((P/'candidate_binding.py').read_bytes())
s=(P/'controller_r308.py').read_text()
replacements=[
 ('One root-selected original32card scoped W2/W4 pair after816terminal; no replay.','One cross-workload32card residency, complete3fixtures x6 arms; no retry/reload.'),
 ("S=R/'incoming/GLM-prefill-W2-W4-owner-r307';LABEL='GLM-prefill-W2-W4-Model-r308'", "S=R/'incoming/GLM-W4-cross-workload-r349/runtime_v4/owner';LABEL='GLM-W4-cross-workload-r349'"),
 ('from candidate_binding import verify_candidate,require_baseline_terminal,CPU628_SHA','from candidate_binding_r349 import verify_candidate,require_baseline_terminal,CPU628_SHA,verify_whole_CPU'),
 ("OWNER_SHA='3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'", "OWNER_SHA='"+sha(D/'owner/SOURCE_MANIFEST.json')+"'"),
 ("(32,8,4,32768,4096,8192,6,6)","(32,8,4,32768,4096,8192,18,18)"),
 ("g['GPU_max_s']==29400 and g['Native_max_s']==28800 and g['CPU_keeper_max_s']==32400", "g['GPU_max_s']==10800 and g['Native_max_s']==10200 and g['CPU_keeper_max_s']==12000"),
 (' verify_candidate(R,D,S)', ' verify_candidate(R,D,S);verify_whole_CPU(R,S,g)'),
 ("'scope':'Scoped fixed8900-token W2/W4 six-request pair:2warmups+1ABBA; no fullagent/benchmark/clientSLO qualification'", "'scope':'Three complete recorded tool workloads, each2warmups+1ABBA; anchor is cross-residency control; no fullbenchmark/onlineSLO qualification'"),
 ("'GPU_max_s':29400,'CPU_keeper_max_s':32400", "'GPU_max_s':10800,'CPU_keeper_max_s':12000"),
 ("2026-10-31", "2026-10-24"),
 (" save()\n try:\n  assert account()is None", " with W.open('x')as first:json.dump(w,first,indent=2);first.flush();os.fsync(first.fileno())\n try:\n  assert account()is None"),
 ('WAVE_INSIDE_R2323=GLM-prefill-W2-W4-owner-r307/GPU_inside_r290.sh', 'WAVE_INSIDE_R2323=GLM-W4-cross-workload-r349/runtime_v4/owner/GPU_inside_r290.sh'),
 ("'08:10:00'", "'03:00:00'"),
 ('end=start+30000','end=start+11400'),
 ("  w['post']=idle('post');w['post_empty']=True", "  exit_until=time.monotonic()+90\n  while proc.poll()is None and time.monotonic()<exit_until:time.sleep(1)\n  w['srun_exit_code']=proc.poll();save()\n  assert w['srun_exit_code']is not None,'Slurm terminal but srun exit unresolved; no restart'\n  w['post']=idle('post');w['post_empty']=True"),
 ("  if terminal_failure(w['terminal']):", "  if terminal_failure(w['terminal'])or w['srun_exit_code']!=0:"),
 ("  owners=[];signatures=[];logs=[]", "  from artifact_visibility_r349 import terminal_report,terminal_log\n  card=json.loads((S/'FIXTURE_CARD.json').read_bytes());fixtures=card['fixtures'];windows=[2,4,2,4,4,2]*3\n  owners=[];signatures=[];logs=[]"),
 ("text=log.read_text(errors='replace');normal=text.count", "text=terminal_log(log,'GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage));normal=text.count"),
 ("q=json.loads(p.read_bytes());assert q['completed']", "q=terminal_report(p);assert q['completed']"),
 ("len(rows)==q['full_requests_completed']==6", "len(rows)==q['full_requests_completed']==q['planned_Native']==q['attempted_Native']==q['completed_Native']==18"),
 ("==[2,4,2,4,4,2] and[r['pair_index']for r in rows]==list(range(6)) and[r['paired_warmup']for r in rows]==[True,True,False,False,False,False]", "==windows and[r['pair_index']for r in rows]==list(range(18)) and[r['paired_warmup']for r in rows]==[True,True,False,False,False,False]*3"),
 ("    pair=rows;assert all(r['prompt_tokens']==8900 and r['requested_output_budget']==4096 and r['generated_natural_EOS']and r['reason']=='eos_token'for r in pair)and len({r['actual_prompt_token_ids_SHA256']for r in pair})==1", "    for fi,f in enumerate(fixtures):\n     pair=rows[fi*6:(fi+1)*6];assert len(pair)==6 and all(r['fixture_id']==f['id'] and r['fixture_index']==fi and r['arm_index']==i and r['global_index']==fi*6+i and r['prompt_tokens']==f['current_prepared_prompt_tokens'] and r['requested_output_budget']==4096 and r['generated_natural_EOS']and r['reason']=='eos_token' and r['actual_prompt_token_ids_SHA256']==f['current_prepared_prompt_IDs_SHA256'] for i,r in enumerate(pair))"),
 ("'Native_requests':6", "'Native_requests':18"),
 ("client=json.loads(p.read_bytes());assert", "client=terminal_report(p);assert"),
 ("client['actual_HTTP']==6 and client['actual_Native']==6", "client['planned_HTTP']==client['planned_Native']==client['attempted_HTTP']==client['completed_HTTP']==client['actual_HTTP']==client['actual_Native']==18"),
 ("state='SCOPED_W2_W4_PAIR_PASS_PENDING_ROOT_PERF_AUDIT'", "state='THREE_WORKLOAD_W2_W4_PASS_PENDING_ROOT_PERF_AUDIT'"),
 ('actual_HTTP=6,actual_Native=6','actual_HTTP=18,actual_Native=18'),
 ]
for a,b in replacements:
 assert a in s,a;s=s.replace(a,b)
(O/'controller_r349.py').write_bytes(s.encode());(O/'controller_r308.py').unlink()
invoke=(P/'invoke_once_r308.py').read_text().replace('from candidate_binding import','from candidate_binding_r349 import').replace('controller_r308','controller_r349').replace("'09:00:00'","'03:20:00'")
(O/'invoke_once_r349.py').write_bytes(invoke.encode());(O/'invoke_once_r308.py').unlink()
template=json.loads((P/'execution-GO.template.json').read_bytes())
template.update(owner_manifest_SHA256=sha(D/'owner/SOURCE_MANIFEST.json'),max_HTTP=18,max_Native=18,GPU_max_s=10800,Native_max_s=10200,CPU_keeper_max_s=12000,baseline_result_consumed=True,baseline_result_record='results/bw1000/2026-11-02/GLM308823-root-audited-W4-scoped-Model-gain-PASS-clean.json',whole_CPU_receipt_path='REQUIRES_REAL_CPU_WHOLE_GLUE_RECEIPT',whole_CPU_receipt_SHA256='REQUIRES_REAL_RECEIPT_SHA')
(O/'execution-GO.template.json').write_bytes((json.dumps(template,indent=2)+'\n').encode())
print(json.dumps({'controller_AST':bool(compile(s,'controller_r349.py','exec')),'GPU_GO':False}))
