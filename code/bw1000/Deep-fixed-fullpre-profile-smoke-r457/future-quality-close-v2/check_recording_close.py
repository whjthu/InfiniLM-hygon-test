"""Only changed quality-negative close branch; local recording, no jobs/Native."""
from pathlib import Path
import argparse,contextlib,importlib.util,io,json,os,sys,tempfile,time,types
from unittest.mock import patch
D=Path(__file__).resolve().parent;P=D.parent/'deployment-gpu-v1';C=D.parent/'deployment-cpu-v1'
sys.path[:0]=[str(D),str(P)]
import client,collect,controller,fixture,preflight
assert Path(client.__file__).resolve().parent==D and Path(collect.__file__).resolve().parent==D and Path(controller.__file__).resolve().parent==D
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);a=parser.parse_args()
 spec=importlib.util.spec_from_file_location('prior_recording_reports',C/'check_CPU.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
 card=json.loads((P/'FIXTURE_CARD.json').read_bytes());inputs=fixture.assets(card,a.inputs);manifest=json.loads((P/'SOURCE_MANIFEST.json').read_bytes())
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();source=root/'future';source.mkdir()
  for path in P.iterdir():
   if path.is_file():(source/path.name).write_bytes(path.read_bytes())
  for name in['client.py','collect.py','controller.py']:(source/name).write_bytes((D/name).read_bytes())
  processes={};sent=[];completed_client=[];counter=[0]
  def open_http(request,timeout):
   i=len(sent);sent.append(request.data);entry,raw,body,gold=inputs[i];assert request.data==raw
   content='#### '+gold['gold_numeric']if i==0 else'Answer: B'
   assert i==0 or gold['gold_choice']!='B','Recording negative mirrors the unchanged actual known wrong answer'
   usage=dict(prompt_tokens=entry['prompt_tokens'],completion_tokens=3,total_tokens=entry['prompt_tokens']+3)
   wire=('data: '+json.dumps(dict(id='request-'+str(i),choices=[dict(index=0,delta=dict(role='assistant',content=content),finish_reason='stop')],usage=usage))+'\n\ndata: [DONE]\n\n').encode()
   return prior.Response(wire,True)
  def fake_popen(argv,**kwargs):
   assert kwargs['start_new_session']and kwargs['stdin']==controller.subprocess.DEVNULL;label=next(value.split('=',1)[1]for value in argv if value.startswith('--job-name='));counter[0]+=1
   class Process:
    pid=10000+counter[0]
    def poll(self):return 0 if label.endswith('-client')or(root/'results/Deep-fixed-profile-public-r457'/controller.RUN/'STOP.json').exists()else None
   process=Process();processes[label]=process
   if label.endswith('-client'):
    start=argv.index('--base-url');args=argv[start:]
    with patch.object(client,'assets',lambda value:fixture.assets(value,a.inputs)),patch.object(client.urllib.request,'urlopen',open_http),patch.object(client,'wait_listener',lambda *args:None):rc=client.main(args)
    report=json.loads((Path(args[args.index('--output-dir')+1])/'result.json').read_bytes());completed_client.append(report)
    assert rc==0 and report['completed']and report['execution_passed']and not report['quality_passed']and not report['passed']and report['completed_HTTP']==2
    assert report['rows'][0]['score']['correct']and not report['rows'][1]['score']['correct']
   else:
    epoch=float(kwargs['env']['DEEP_R457_MODEL_GPUSTART']);identity=dict(run=controller.RUN,actual_step='910374.99999',source_manifest_SHA256=manifest['provider_source_SHA256'],provider_config_SHA256=manifest['config_SHA256'],gpu_start_epoch=epoch,Native_seconds=5280)
    prior.reports(root/'results/Deep-fixed-profile-public-r457'/controller.RUN,identity,card)
   return process
  def output(argv,**kwargs):
   if argv[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=recording(123) NodeList=a02r4n04'
   if argv[:3]==['scontrol','show','hostnames']:return'a02r4n04'
   assert argv[0]=='sacct'
   return '\n'.join('|'.join(['910374.'+('99998'if label.endswith('-client')else'99999'),label,'COMPLETED'if process.poll()==0 else'RUNNING','0:0','a02r4n04'])for label,process in processes.items())
  def idle(argv,**kwargs):
   assert '--gres=none'in argv and argv[-1].endswith('idle.py')
   return types.SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(node='a02r4n04',CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=[],matching_component_processes=[],expected_PIDs_still_exist=[],ports_bindable={'43882':True},physical_PCI=list(map(str,range(8))))))
  proof=dict(passed=True,recording_original_CPU963_preflight=True,GPU=0,Slurm=0)
  with patch.multiple(controller,R=root,D=source,PACKAGE=P),patch.object(controller,'verify',lambda digest:manifest),patch.object(preflight,'check',lambda *args:proof),patch.dict(os.environ,dict(SLURM_JOB_ID='910407',SLURM_STEP_ID='88888',USER='recording')),patch.object(controller.socket,'gethostname',lambda:'a14r2n20'),patch.object(controller.subprocess,'Popen',fake_popen),patch.object(controller.subprocess,'check_output',output),patch.object(controller.subprocess,'run',idle),patch.object(controller.time,'sleep',lambda seconds:None),patch.object(controller.os,'killpg',side_effect=AssertionError('Quality negative must not signal the Native owner'),create=True),contextlib.redirect_stdout(io.StringIO()):rc=controller.main(['--manifest-sha','recording','--publication-commit','0'*40])
  watch=json.loads((root/'results'/(controller.LABEL+'-'+controller.RUN+'-watch.json')).read_bytes())
  assert rc==0 and watch['completed']and watch['execution_passed']and not watch['quality_passed']and not watch['passed']and watch['state']=='QUALITY_FAILED_NORMAL_CLEANUP'
  assert not watch['forced_cleanup']and watch['normal_STOP_published']and watch['normal8internalowner_destroy']and watch['post'][0]['KFD_available']and len(processes)==2 and len(sent)==2 and len(completed_client)==1
  assert watch['semantic_gold_and_natural_EOS']is False and watch['public_API_execution_scope_pass']is True
  result=dict(passed=True,actual_changed_client_and_controller_main_and_collector=True,normal_STOP_collect_post_before_quality_FAIL=True,planned_HTTP=2,completed_HTTP=2,quality_passed=False,overall_passed=False,execution_passed=True,TERM_KILL_calls=0,no_gold_or_input_change=True,Native_HTTP_Slurm_reports_recording=True,actual_Native_or_GPU_or_HTTP=False,old1413_normal_retroactively_changed=False,new_jobs=0)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
