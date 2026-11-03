"""Changed roles/time/client bindings; no Native/math/model test repetition."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,os,sys,tempfile,time
from unittest.mock import patch
D=Path(__file__).resolve().parent;OWNER=D/'owner';POLICY=D.parents[1]/'GLM-NoV-W4-normal-profile-r417';sys.path.insert(0,str(OWNER));sys.path.insert(0,str(D))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def reject(fn):
 try:fn()
 except(AssertionError,ValueError,KeyError):return
 raise AssertionError('Invalid changed binding accepted')
def run(config,bash,encoding_archive):
 import controller as C
 from check_shell_layout import check
 loader=sha(OWNER/'SOURCE_MANIFEST.json');profile=sha(POLICY/'SOURCE_MANIFEST.json')
 assert profile=='695b1a12bfd2929851b3ffa89a9267564f9fd9dc0045c4a9adfbcf7142b21584'
 shell=check(D,config,POLICY,loader,profile,bash);assert len(shell['all_four_roles'])==4
 B=module(OWNER/'execution_bindings_r418.py','changed_bindingsCPU');T=module(OWNER/'topology_r404.py','changed_topologyCPU');clock=module(OWNER/'deadline_r290.py','changed_deadlineCPU')
 rows=[]
 for item in shell['all_four_roles']:
  env=item['actual_Bash_environment'];layout=item['actual_typed_layout'];stage=layout['stage']
  assert layout['layer_begin']==[0,38][stage]and layout['layer_end']==[38,78][stage]
  for key,value in [('GLM_R418_REPLICA_INDEX','2'),('GLM_R418_SERVICE_SECONDS','13800'),('GLM_R418_NATIVE_SECONDS','5100'),('GLM_PIPELINE_TAIL_NODE','foreign')]:
   bad=dict(env);bad[key]=value;reject(lambda:B.validate(bad,OWNER,POLICY))
  bad=dict(env);bad['GLM_R418_REPLICA_INDEX']=str(1-item['replica']);reject(lambda:T.read_layout(bad,config,item['node']))
  epoch=float(env['GLM_R290_GPUSTART_UNIX'])
  with patch.dict(os.environ,env),patch.object(clock.time,'time',return_value=epoch+13499):assert clock.remaining()==1;clock.check('positive_before_original_service_end')
  with patch.dict(os.environ,env),patch.object(clock.time,'time',return_value=epoch+13500):
   try:clock.check('same_original_epoch_no_slide')
   except TimeoutError:pass
   else:raise AssertionError('Originalserviceepoch silently extended')
  rows.append(dict(replica=item['replica'],stage=stage,node=item['node'],Native_cleanup_reserve_s=600,time_limit_checked=True,wrong_pair_rejected=True))
 # Actual final argument producer is consumed by the real root client parser
 # and its original ready/owner_row functions. Only Native output is recorded.
 Q=module(D.parent/'client-v1/quality_client.py','actual_quality_clientCPU')
 a=json.loads(Path(encoding_archive).read_bytes())['archive'];raw=a['files'][a['receipt_path']]['rawtext'];assert hashlib.sha256(raw.encode()).hexdigest()=='2051a3a74e5ce471b53c38ddffd0f744e1a0669bdc2312590f6bd6d2a5013c12';encoding=json.loads(raw);plan=json.loads((D.parent/'fixtures-r1/requests-and-pairs.json').read_bytes());encoded={c['case_id']:c for c in encoding['cases']}
 handoffs=[]
 with tempfile.TemporaryDirectory(prefix='r418-changed-controller-')as tmp:
  root=Path(tmp);(root/'results').mkdir()
  m=dict(loader_manifest_SHA256=loader,profile_manifest_SHA256=profile,encoding_receipt_SHA256='2051a3a74e5ce471b53c38ddffd0f744e1a0669bdc2312590f6bd6d2a5013c12',encoding_source_SHA256=encoding['source_manifest_SHA256'])
  for index,nodes in enumerate(C.PAIRS):
   rec=dict(index=index,run=C.RUN+'-r'+str(index),actual_GPU='CPU-recorded.'+str(index));args=Q.parser().parse_args(C.quality_args(rec,m,time.time(),root,D.parent))
   ready=dict(ready=True,run=rec['run'],actual_step=rec['actual_GPU'],Native_SHA256=Q.NATIVE,TP=8,PP=2,cards=16,rows=1,capacity=32768,loader_parent_source_manifest_SHA256=loader,service_profile_manifest_SHA256=profile,prepared_entry_identity=dict(upstream_commit=Q.UPSTREAM),context_policy='v403-no-resident-V',context_policy_env='1',P0_text_buffered_until_completion=True,normal_output_budget=4096)
   Path(args.ready).write_text(json.dumps(ready));assert Q.validate_ready(args)==ready
   selected=[r['case_id']for ordinal,r in enumerate(plan['requests'])if(ordinal//2)%2==index];schedule=[r for r in plan['comparison_schedule']if r['case_id']in set(selected)];assert len(selected)==32 and len(schedule)==64 and sum(r['window']==2 for r in schedule[::2])==16
   item=schedule[0];e=encoded[item['case_id']];ids=[21,154820];ticks=[11.,11.2]
   row=dict(request_id='CPU-recorded-'+str(index),request_index=0,admitted_index=0,native_profile='glm-nov16-w'+str(item['window']),prefill_control_window=item['window'],service_profile_manifest_SHA256=profile,normal_reset_before=True,normal_reset_after=True,original_reset_all16_completed=True,max_new_tokens=4096,requested_output_budget=4096,prompt_tokens=e['prompt_tokens'],actual_prompt_token_ids_SHA256=e['prompt_IDs_SHA256'],admission=dict(accepted_monotonic=9.,Native_prompt_token_ids_SHA256=e['prompt_IDs_SHA256'],history_render_audit=dict(qualified=True,prequeue_bound=True,markers_sent_to_Native=False,prompt_tokens=e['prompt_tokens'])),Native_continuation=[dict(kind='prefill',tokens=e['prompt_tokens']),dict(kind='decode',tokens=1)],generated_token_ids=ids,actual_generated_tokens=2,generated_token_ids_SHA256=Q.digest(Q.compact(ids)),native_finish_reason='eos_token',backend_token_monotonic=ticks,started_monotonic=10.,ended_monotonic=12.,Native_forward_host_summary=dict(prefill_calls=1,decode_calls=1),final_metadata=dict(recorded_Native_output_only=True))
   owner=dict(actual_step=rec['actual_GPU'],run=rec['run'],Native_SHA256=Q.NATIVE,source_manifest_SHA256=loader,service_profile_manifest_SHA256=profile,service_requests=[row]);Path(args.service_report).write_text(json.dumps(owner))
   usage=dict(prompt_tokens=e['prompt_tokens'],completion_tokens=2,total_tokens=e['prompt_tokens']+2);proof=Q.owner_row(args,ready,row['request_id'],0,item['window'],e,usage,'stop');assert proof['natural_EOS']
   bad=copy.copy(args);bad.loader_manifest_sha='0'*64;reject(lambda:Q.validate_ready(bad));reject(lambda:Q.owner_row(bad,ready,row['request_id'],0,item['window'],e,usage,'stop'))
   handoffs.append(dict(replica=index,actual_parser=True,actual_validate_ready=True,actual_owner_row=True,mocked_validate_ready_or_owner_row=False,new_loader_manifest_SHA256=loader,profile_manifest_SHA256=profile,case_id=item['case_id'],body_source_SHA256=item['compact_body_SHA256'],native_offset=args.native_offset,planned_questions=32,planned_HTTP=64,W2_first_pairs=16,W4_first_pairs=16,recorded_Native_output_only=True))
 original=D.parents[1]/'GLM-NoV-16card-W4-pair-r414/owner/owner_r290.py';new=(OWNER/'owner_r290.py').read_text();old=original.read_text();start='            check(lib.modern_glm_model_reset(ctx));barrier();generated=[]';end="            row={'sequence':seq";assert new[new.index(start):new.index(end)]==old[old.index(start):old.index(end)]
 return dict(passed=True,actual_four_role_Bash=shell,new_role_time_contracts=rows,actual_client_handoffs=handoffs,original47line_math_block_SHA256=hashlib.sha256(old[old.index(start):old.index(end)].encode()).hexdigest(),root_original64HTTP_client_CPU_reused=True,new_HTTP_requests=0,Native=0,GPU=0,Slurm=0,Model=0,compile=False,weights=False,source_loader_manifest_SHA256=loader,profile_manifest_SHA256=profile,scope='Newfixedroles/time/env andactualclientparser/ready/row consumingfreshfiles+real844encodedmetadata;Nativeoutputrecorded. FinalactualhostCLI/source/librarybytes+fullPopenlifecycle recording stillpending,not devicequalification.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--bash',default='bash');p.add_argument('--encoding-archive',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()
 try:q=run(a.config,a.bash,a.encoding_archive)
 except BaseException as e:
  import traceback;q=dict(passed=False,error=repr(e),traceback=traceback.format_exc(),Native=0,GPU=0,Slurm=0,Model=0)
 out.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(dict(passed=q['passed'],output=str(out),SHA256=sha(out))));raise SystemExit(0 if q['passed']else 1)
