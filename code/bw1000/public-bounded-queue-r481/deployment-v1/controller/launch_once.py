"""One keeper submission per fixed instance; no retry, epoch starts after fresh idle."""
from pathlib import Path
import argparse,json,os,re,socket,subprocess,sys,time
from support import *
from qualified_cleanup import clean_environment,TERMINAL

def first_serial_gate(m):
 matrix=read(D/'MATRIX.json')['instances']
 for name in['A-OFF','A-ON']:
  v=matrix[name];base=R/'results'/(LABEL+'-'+v['run']);w=read(base.with_name(base.name+'-watch.json'));assert w['actual_GPU']and w['deployment_source_SHA256']==SHA(DEPLOY/'SOURCE_MANIFEST.json')
  client=read(base.with_name(base.name+'-client')/'result.json');row=next(r for r in client['rows']if r['index']==0);assert row.get('server_request_id')and row.get('clock',{}).get('EOS_tokens')==1 and row.get('fully_received_EOF')and not row.get('planned_cancel')and row.get('clock',{}).get('owned_finished')and row.get('clock',{}).get('complete')
  runtime=hostpath(read(DEPLOY/v['config'])['runtime_directory'])/v['run'];owner=read(runtime/'owner.json');assert owner['identity']['actual_step']==w['actual_GPU']and owner['identity']['source_manifest_SHA256']==PROVIDER_SHA
  matches=[r for r in owner['requests']if r['request_id']==row['server_request_id']];assert len(matches)==1 and matches[0]['normal_reset_before']and matches[0]['normal_reset_after']and matches[0]['native_finish_reason']=='eos_token'
 return dict(passed=True,scope='BothAfirstfullserial EOS+actualownerreset; correctness score is not an admission gate')

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--instance',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m,v,c,cp=verify(a.manifest_sha,a.instance)
 if a.preflight_only:print(json.dumps(dict(passed=True,instance=a.instance,source_only=True,CPU1068='metadataonly',GPU=0,Slurm=0)));return
 assert m['GPU_GO']and m['Model_GO'];label=LABEL+'-'+v['run'];intent=R/'results'/(label+'-intent.json');watch=R/'results'/(label+'-watch.json');known=[intent,watch,folder(c,v),R/'results'/(label+'-client'),R/'results'/(label+'-client-server.json'),R/'results'/(label+'-client-config.json'),R/'results'/(label+'-client-identity.json')]+[R/'logs'/(label+suffix)for suffix in ['-keeper-client.log','-srun.log','-client-srun.log']];assert not any(p.exists()for p in known),'Originalscope exists; reconcile, never replay'
 gate=first_serial_gate(m)if v['pair']=='B'else None
 z=subprocess.run([sys.executable,'-B',str(D/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--instance',a.instance,'--preflight-only'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90);assert z.returncode==0,(z.stdout,z.stderr)
 for name in['inside.sh','resource_wave_host_alias_outer.sh']:subprocess.run(['bash','-n',str(D/name)],check=True,timeout=10)
 rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-11-05','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],universal_newlines=True,timeout=25).splitlines()if'|'in x];assert not any(x[1].startswith(label)for x in rows),'Originalexactlabel exists'
 for parent,node in[('910374',v['node']),('910407','a14r2n20')]:
  raw=subprocess.check_output(['scontrol','show','job',parent,'-o'],universal_newlines=True,timeout=20);assert'JobState=RUNNING'in raw and'UserId=aclsylqidf('in raw;assert node in subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)],universal_newlines=True,timeout=20).splitlines()
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','06:10:00','--job-name='+label+'-keeper','--output='+str(R/'logs'/(label+'-keeper-%J.log')),HOST_PYTHON,'-B','-u',str(D/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--instance',a.instance]
 exclusive(intent,dict(publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,instance=a.instance,CPU1068='910407.1068',first_serial_gate=gate,argv=args,created_unix=time.time(),HOST=socket.gethostname(),one_attempt=True,GPU_epoch_created=False,planned_HTTP=20,warms=0))
 with(R/'logs'/(label+'-keeper-client.log')).open('x')as log:proc=subprocess.Popen(args,env=clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(keeper_submitted=True,PID=proc.pid,PGID=proc.pid,startticks=Path('/proc/'+str(proc.pid)+'/stat').read_text().split()[21],intent=str(intent),instance=a.instance,actual_GPU=None,epoch_only_after_freshidle=True)))
if __name__=='__main__':main()
