"""Audit exact CPU962 receipt, command bytes and actual scope; stdlib only."""
import argparse,base64,hashlib,json
from pathlib import Path
MF='bc7541b86afd15894d374c774ca3ed5093c623134bc493467518e4b7b3a2a98a'
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
PROFILE='deep-hc0-fullpre-c1'
RAW='c46a737c87e1d82e41783ae6a9f9860a4312624494bd858de0c8fec0e38d07a2'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def analyze(observation,source):
 q=json.loads(Path(observation).read_bytes());files={}
 for name,row in q['files'].items():
  raw=base64.b64decode(row['base64'],validate=True)
  assert len(raw)==row['bytes']and digest(raw)==row['SHA256'],name
  files[name]=raw
 prefix='candidates/Deep454-fixed-profile-CPU-v1/'
 raw=files[prefix+'result.json'];assert digest(raw)==RAW;r=json.loads(raw)
 assert r['actual_CPU']=='910407.962'and r['completed']is True and r['passed']is True
 assert r['source_manifest_SHA256']==MF and r['affinity']==[0,1,2,3]
 assert all(r[k]is False for k in ['Native_dlopen','Runtime_execution','Model_loaded','GPU_GO','Model_GO','request_mode_control','old_math_or_kernel_suites_repeated'])
 assert r['GPU']==r['new_HOST_TUs']==r['new_GPU_TUs']==r['r366_fixed']==r['default_mode']==0
 assert r['explicit_engine_profile']==PROFILE and r['performance']=='UNMEASURED'
 commands=r['commands'];names=['changed-fullpre-worker-and-typed-CABI','actual50object-derivation-NO-dlopen','actual-CLI-default-OFF-NOT-serving','actual-CLI-explicit-profile-NOT-serving']
 assert [c['name']for c in commands]==names
 outputs=[]
 for command in commands:
  assert command['rc']==0 and command['argv'][0]=='/usr/bin/python3'and command['argv'][1]=='-B'
  out=files[prefix+command['name']+'.stdout'];err=files[prefix+command['name']+'.stderr']
  assert digest(out)==command['stdout_SHA256']and digest(err)==command['stderr_SHA256']and not err
  outputs.append(json.loads(out))
 assert outputs==[r['worker_control'],r['actual_binding'],r['actual_whole_CLI_default'],r['actual_whole_CLI_explicit']]
 consumer=outputs[0];assert consumer['passed']and consumer['recording_processor_HTTP_Native_leaves']and consumer['actual_Native_or_GPU']is False and consumer['old_encoding_or_938_suite_replayed']is False
 cases=consumer['cases'];assert len(cases)==9
 expected=[(None,'normal',7,29),(PROFILE,'normal',7,29),(PROFILE,'early-eos-no-T1',1,1),(PROFILE,'disconnect',1,2),(PROFILE,'timeout',1,2),(PROFILE,'missing-hit',1,3),(PROFILE,'forward-failure',1,1),(PROFILE,'reset-drain-failure',1,3)]
 for case,(profile,scenario,count,forwards)in zip(cases,expected):
  assert case['profile']==profile and case['scenario']==scenario and case['requests']==count and case['recording_Native_forwards']==forwards
  assert case['passed']and case['actual_prepared_factory_FIFO_worker_generate_reset']and case['actual_Native_or_GPU']is False
 last=cases[-1]
 assert all(last[k]is True for k in ['passed','actual_build_server_provider_factory','actual_Actors_ready_reader','actual_provider_start_READY_publication','foreign_request_controls_rejected','C1_context_and_budget_guards_retained','HTTP_and_Native_ready_rows_recording_only'])
 assert last['fixed_mode']==1 and last['Model_READY_proved']is False
 binding=outputs[1];assert binding['library_SHA256']==NATIVE and binding['objects']==50 and binding['guard_retained']and binding['Native_dlopen']is False and binding['project_maps']==[]and set(binding['hidden'].values())=={'-1'}
 base=dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 assert outputs[2]==dict(base,experimental_profile=None)and outputs[3]==dict(base,experimental_profile=PROFILE)
 assert commands[3]['argv']==commands[2]['argv']+['--experimental-profile',PROFILE]
 assert '--serve'not in commands[2]['argv']and '--serve'not in commands[3]['argv']
 source=Path(source);mraw=(source/'SOURCE_MANIFEST.json').read_bytes();assert digest(mraw)==MF;m=json.loads(mraw)
 assert m['CPU_GO']is True and m['GPU_GO']is m['Model_GO']is False
 assert m['default_mode']==0 and m['fixed_explicit_mode']==1 and m['experimental_profile']==PROFILE
 for name,h in m['files_SHA256'].items():assert digest((source/name).read_bytes())==h,name
 assert m['parents_manifest_SHA256']['/refactor/incoming/public-Deep-native-provider-r427/source-v3']=='79285a56e96c4f3586f94b3af6a56bbde085bfdfda2250c9b0335361aaaddec0'
 account=q['accounting'];assert account['rc']==0
 rows=[s.split('|')for s in account['stdout'].splitlines()if s.startswith('910407.962|')]
 assert len(rows)==1 and rows[0][1:]==['Deep454-fixed-profile-CPU-v1','COMPLETED','0:0','3','a14r2n20']
 intent=json.loads(files['results/Deep454-fixed-profile-CPU-v1-intent.json']);launch=json.loads(files['results/Deep454-fixed-profile-CPU-v1-launch.json'])
 assert intent['source_manifest_SHA256']==launch['source_manifest_SHA256']==MF and launch['srun_PID']==1239957 and launch['epoch']==1793764577.2745852
 assert launch['publication_commit']=='e0449f0ad3b7730216f14a7ee0b7854578905fa0'
 assert '--gres=none'in launch['command']and'-c4'in launch['command']and'--mem=8192M'in launch['command']
 return dict(passed=True,scope='New fixed experimental public entry/source and changed CPU consumer only',actual_CPU='910407.962',state='COMPLETED',exit='0:0',elapsed_s=3,receipt_SHA256=RAW,source_manifest_SHA256=MF,source_publication=launch['publication_commit'],Native_SHA256=NATIVE,raw_files=len(files),new_cases=9,default_OFF=True,explicit_engine_fixed_ON=PROFILE,actual_whole_nonserve_CLI_profiles=2,actual50object_dependency_metadata=True,actual_factory_FIFO_worker_generate_reset_consumed=True,actual_ready_reader_and_publication_recording_only=True,Native_dlopen=False,GPU=False,Model=False,production_or_default_promotion=False,new_performance='UNMEASURED',limitations=['HTTP/processor/Native operations in new consumer are explicit recording leaves','No fresh full serve/Model execution of this general entry','Prior449 speed is scoped;450 screen is not NI and seven discordants persist;451 baseline quality failure persists'])
def main():
 p=argparse.ArgumentParser();p.add_argument('observation',type=Path);p.add_argument('--source',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.observation,a.source),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
