"""Copy the complete proven307 entry; make enumerated workload/budget deltas."""
import pathlib,hashlib,json
D=pathlib.Path(__file__).parent;P=D.parent/'public-native-bridge/GLM-prefill-W2-W4-owner-r307';O=D/'owner'
manifest=json.loads((P/'SOURCE_MANIFEST.json').read_bytes())
assert hashlib.sha256((P/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
O.mkdir(exist_ok=True)
for n,h in manifest['files_SHA256'].items():
 b=(P/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def change(n,changes):
 p=O/n;s=p.read_bytes().decode()
 for old,new in changes:
  assert old in s,(n,old);s=s.replace(old,new)
 p.write_bytes(s.encode())
change('deadline_r290.py',[('SERVICE_SECONDS=27720','SERVICE_SECONDS=9120'),('HARD_NATIVE_SECONDS=28800','HARD_NATIVE_SECONDS=10200')])
change('service_budget_r204.py',[('PROCESS_SECONDS=28800','PROCESS_SECONDS=10200'),('SHELL_HARD_SECONDS=28800','SHELL_HARD_SECONDS=10200'),('GPU_SLURM_SECONDS=29400','GPU_SLURM_SECONDS=10800'),('KEEPER_SECONDS=32400','KEEPER_SECONDS=12000')])
change('GPU_inside_r290.sh',[('GLM-prefill-W2-W4-owner-r307','GLM-W4-cross-workload-r349/runtime_v4/owner'),('28800','10200')])
change('model_entry_r290.py',[("'request_file':'/refactor/results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json','request_SHA256':'5ee8ea9cc99e12969d4b0505d4dd70f50e6436079206e8de5a6e908c5854c9dd'", "'fixture_card':str(pathlib.Path(__file__).parent/'FIXTURE_CARD.json')")])
change('owner_r290.py',[
 ('    windows=(2,4)+(2,4,4,2)', "    from workload_r349 import load_plan\n    fixture_card,workload_plan=load_plan(folder/'FIXTURE_CARD.json')\n    windows=tuple(arm['window'] for arm in workload_plan)"),
 ("    assert set(paired)=={'request_file','request_SHA256','output_root'}", "    assert set(paired)=={'fixture_card','output_root'}\n    assert pathlib.Path(paired['fixture_card']).resolve()==(folder/'FIXTURE_CARD.json').resolve()"),
 ("    assert len(paired['request_SHA256'])==64 and all(c in '0123456789abcdef' for c in paired['request_SHA256'])", "    assert len(workload_plan)==18"),
 ('request_index=0;paired_input=None','request_index=0;paired_input={}'),
 ('paired_warmup_requests=2,','paired_warmup_requests=6,'),('paired_measured_requests=4,','paired_measured_requests=12,'),
 ('six identical frozen8900-token requests, two warmups and one ABBA block','three frozen complete workloads, six warmups and three ABBA blocks'),
 ("plan={'HTTP':6,'Native':6,'phase_Native_end':{'paired':6}}", "plan={'HTTP':18,'Native':18,'phase_Native_end':{'paired':18}},planned_HTTP=18,planned_Native=18,attempted_Native=0,completed_Native=0"),
 ("'prefill_pair_windows':[2,4]+[2,4,4,2],'paired_request_SHA256':paired['request_SHA256']", "'prefill_pair_windows':list(windows),'fixture_card_SHA256':__import__('hashlib').sha256((folder/'FIXTURE_CARD.json').read_bytes()).hexdigest(),'fixture_request_SHA256':[x['request_SHA256'] for x in fixture_card['fixtures']]"),
 ('+27720)', '+9120)'),
 ("'Scoped six-request Native plan exhausted'", "'Scoped eighteen-request Native plan exhausted'"),
 ("            if request_index>=0:\n                if paired_input is None:paired_input=current_input\n                assert current_input==paired_input,'Paired full prompt IDs/output budget changed'\n            prefill_window=windows[request_index];pair_index=request_index;request_index+=1", "            arm=workload_plan[request_index];fixture=fixture_card['fixtures'][arm['fixture_index']]\n            key=arm['fixture_id']\n            if key not in paired_input:paired_input[key]=current_input\n            assert current_input==paired_input[key],'Per-fixture full prompt IDs/output budget changed'\n            actual_ids_SHA=__import__('hashlib').sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()\n            assert actual_ids_SHA==fixture['current_prepared_prompt_IDs_SHA256'] and len(prompt)==fixture['current_prepared_prompt_tokens'] and limit==4096\n            prefill_window=windows[request_index];pair_index=request_index;request_index+=1\n            report['attempted_Native']=request_index;save()"),
 ('paired_warmup=0<=pair_index<2,','paired_warmup=arm[\'warmup\'],fixture_id=arm[\'fixture_id\'],fixture_index=arm[\'fixture_index\'],arm_index=arm[\'arm_index\'],global_index=pair_index,'),
 ("report['service_requests'].append(row);save()", "report['service_requests'].append(row);report['completed_Native']=len(report['service_requests']);save()"),
 ('request_index==6','request_index==18'),
 ])
# Preserve untouched loader/preweight/prepared identity and cleanup files byte-for-byte.
unchanged=['preweight_r290.py','prepared_family_entry_r268.py','loader_audit_r290.py','client_owner_r290.py','cleanup_owned_r290.py']
for n in unchanged:
 if (P/n).exists():assert (P/n).read_bytes()==(O/n).read_bytes()
print(json.dumps({'owner_files':len(manifest['files_SHA256']),'GPU_GO':False,'loader_identity_preserved':True}))
