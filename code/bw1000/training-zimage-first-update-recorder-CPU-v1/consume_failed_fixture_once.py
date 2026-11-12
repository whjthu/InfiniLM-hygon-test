"""Independent terminal consumer preserving FAIL and its two passed tests."""
from pathlib import Path
import ast,hashlib,json,subprocess
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
i=json.loads((D/'launch-intent.json').read_bytes());claim=i['claim'];request=i['request_id']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
t=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
files={}
for target,n in t['remote_files'].items():
 raw=subprocess.check_output(git+['show',claim+':'+t['source']+'/'+n]);assert hashlib.sha256(raw).hexdigest()==t['source_SHA256'][n]
 files[target]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
s=(D/'read_original_terminal.py').read_text(encoding='utf8')
changes={
 "assert o['returncode']==0 and o['passed'] and o['owned_group_reaped'] and o['stop_reason'] is None":"assert o['returncode']==1 and o['passed'] is False and o['owned_group_reaped'] and o['stop_reason'] is None",
 "assert r['passed'] and r['real_CPU_tests']==3 and r['failures']==r['errors']==r['skipped']==0 and r['framework_files']==362":"assert r['passed'] is False and r['real_CPU_tests']==3 and r['failures']==r['skipped']==0 and r['errors']==1 and r['framework_files']==362",
 "assert artifacts['CPU-tests.private.log']['SHA256']==r['tests_log_SHA256'] and 'Ran 3 tests' in artifacts['CPU-tests.private.log']['text']":"""assert artifacts['CPU-tests.private.log']['SHA256']==r['tests_log_SHA256'] and 'Ran 3 tests' in artifacts['CPU-tests.private.log']['text']
 text=artifacts['CPU-tests.private.log']['text']
 assert 'test_full25_exact_loss_parameters_optimizer_and_rng (test_zimage_first_update_capture.RealCPUCapture.test_full25_exact_loss_parameters_optimizer_and_rng) ... ok' in text
 assert 'test_prepared_input_mutation_preserves_failure_denominator (test_zimage_first_update_capture.RealCPUCapture.test_prepared_input_mutation_preserves_failure_denominator) ... ok' in text
 assert 'first-update artifact exceeds its byte budget' in text and 'RuntimeError: [enforce fail' in text""",
 "artifacts=artifacts,original_source_files=len(EXPECTED_FILES),framework_files=362,":"artifacts=artifacts,original_run_passed=False,original_returncode=1,passed_tests_retained=2,failed_tests_retained=1,original_failure_preserved=True,original_source_files=len(EXPECTED_FILES),framework_files=362,"
}
for old,new in changes.items():assert s.count(old)==1;s=s.replace(old,new)
ast.parse(s);(D/'read_failed_fixture_terminal.py').write_bytes(s.encode())
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nEXPECTED_FILES='+repr(files)+'\n').encode()+s.encode()
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -'],input=payload,capture_output=True,timeout=85)
(D/'failed-fixture-reader.stdout.private.json').write_bytes(r.stdout);(D/'failed-fixture-reader.stderr.private.txt').write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
with (D/'failed-fixture-reader-result.json').open('x') as f:json.dump(dict(transport_rc=r.returncode,value=v,source_SHA256=hashlib.sha256(s.encode()).hexdigest(),new_CPU_or_GPU_invocations=0),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,reader_passed=v.get('passed') if v else None,failure=v.get('failure') if v else None,
 original_run_passed=False,passed_tests_retained=2,failed_tests_retained=1,new_CPU_or_GPU_invocations=0)))
