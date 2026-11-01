"""Exact metadata-only pathspec; never stage external raw payloads or IDs."""
import pathlib,json,hashlib
D=pathlib.Path(__file__).parent;repo=D.parents[2];results=repo/'results/bw1000/2026-11-02';paths=[]
for directory in [D/'owner',D/'controller']:
 m=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():
  p=directory/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;paths.append(p)
 paths.append(directory/'SOURCE_MANIFEST.json')
names=['CPU_SOURCE_MANIFEST.json','CPU_encode_fixtures.py','CPU_encode_fixtures_v2.py','CPU_encode_fixtures_v3.py','CPU_encode_inside.sh','CPU_encode_inside_v2.sh','CPU_encode_inside_v3.sh','CPU_whole_glue.py','CPU_whole_inside.sh','check_controller_CPU.py','freeze_CPU_package_v4.py','freeze_current_encoding.py','freeze_final_runtime.py','finalize_metadata.py','prepare_owner.py','prepare_driver.py','prepare_controller.py','read_encoder_status.py','read_glue_failure.py','read_whole_status.py','prepare_encoder_v2.py','prepare_encoder_v3.py','prepare_encoder_upload.py','prepare_encoder_upload_v2.py','prepare_encoder_upload_v3.py','prepare_closed_inventory_v4.py','prepare_scoped_stage.py']
paths.extend(D/n for n in names);paths.append(D/'fixtures/FIXTURE_CARD.json')
for directory in [D/'failed-runtime-v1',D/'failed-runtime-v2']:paths.extend(p for p in directory.iterdir()if p.is_file())
resultnames=['GLM349-official-encoding-CPU673-PASS-metadata.json','GLM349-whole-glue-CPU674-FAIL.json','GLM349-whole-glue-CPU676-FAIL.json','GLM349-whole-glue-CPU677-PASS.json','GLM349-whole-glue-CPU678-PASS.json','GLM349-whole-glue-failure-674-676-details.json','GLM349-pre-GPU-CPU-and-source-history.json','GLM349-final-source-CPU-resource-claim.json','GLM349-controller-v3-actual-pins-and-recording-CPU.json']
account={'CPU_terminal_accounting':json.loads((results/'GLM349-whole-glue-v4-observed.json').read_bytes())['sacct'],'encoding_terminal_accounting':json.loads((results/'GLM349-official-encoder-v3-observed.json').read_bytes())['sacct'],'GPU_submitted':False,'not_performance_results':True}
(results/'GLM349-CPU-terminal-accounting.json').write_bytes((json.dumps(account,indent=2)+'\n').encode());resultnames.append('GLM349-CPU-terminal-accounting.json')
paths.extend(results/n for n in resultnames);paths.append(repo/'coordination/tasks/GLM-W4-cross-workload-r349.json')
assert all(p.is_file()and p.is_relative_to(repo)for p in paths)
relative=sorted(set(p.relative_to(repo).as_posix()for p in paths));assert all('request-00'not in n and 'remote_upload_'not in n and 'private-execution-'not in n for n in relative)
(repo.parent/'GLM349-stage-paths.txt').write_bytes(b'\x00'.join(n.encode()for n in relative)+b'\x00')
print(json.dumps({'explicit_paths':len(relative),'metadata_only':True,'raw_HTTP_IDs_included':False}))
