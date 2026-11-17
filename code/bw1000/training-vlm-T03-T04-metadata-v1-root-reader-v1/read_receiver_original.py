import pathlib,subprocess,json,os,pwd,hashlib,time,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-T03-T04-metadata-v1')
name=pwd.getpwuid(os.getuid()).pw_name
out=dict(observed_epoch=time.time(),receiver_namespace_exists=R.exists(),actual_uid=os.getuid(),uid_name_length=len(name),receiver_records={},parents=[])
for n in ('intent.json','outcome.json'):
 p=R/n
 if p.is_file():
  assert p.stat().st_size<=65536 and not p.is_symlink()
  b=p.read_bytes();v=json.loads(b)
  out['receiver_records'][n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value={k:v.get(k) for k in ('request_id','publication','epoch','deadline','pid','startticks','passed','error_type','completed_members')})
raw=subprocess.check_output(['squeue','-h','-j','1087656,1103606','-o','%i|%T|%D|%N|%u'],text=True,timeout=9)
for line in raw.splitlines():
 row=line.split('|');assert len(row)==5
 f=dict(x.split('=',1) for x in subprocess.check_output(['scontrol','show','job',row[0],'-o'],text=True,timeout=9).split() if '=' in x)
 userid=f['UserId'];uid=int(re.search(r'\(([0-9]+)\)',userid).group(1))
 out['parents'].append(dict(job=row[0],queue_state=row[1],queue_nodes=row[2],queue_username_length=len(row[4]),queue_username_equals_uidname=row[4]==name,queue_username_is_truncated_prefix=name.startswith(row[4].rstrip('+')),scontrol_state=f['JobState'],scontrol_nodes=f['NumNodes'],owner_uid_matches=uid==os.getuid(),AllocTRES=f['AllocTRES'],EndTime=f['EndTime']))
print(json.dumps(out))