"""Read the current long-lived parent allocation without starting Slurm steps."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;B=D.parent
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8')
scope={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'ordinary_cluster_ssh','exec'),scope)
remote=b'''
import datetime,json,re,subprocess
def command(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=15);return r.returncode,r.stdout
rc,line=command(['squeue','-h','-j','1087656','-o','%i|%T|%D|%L|%N'])
assert rc==0 and len(line)<4000
result=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),squeue_rc=rc,parent_line=line.strip(),busy_measured=False,new_steps=0)
if line.strip():
 fields=line.strip().split('|');assert fields[0]=='1087656'
 code,nodes=command(['scontrol','show','hostnames',fields[4]]);assert code==0
 hosts=nodes.splitlines();assert len(hosts)==len(set(hosts))==int(fields[2])
 result.update(state=fields[1],node_count=len(hosts),nodes=hosts)
 code,details=command(['scontrol','show','job','-o','1087656']);assert code==0 and len(details)<64000
 for key in ('JobState','NumNodes','NumCPUs','TimeLimit','StartTime','EndTime','Restarts'):
  match=re.search(r'(?:^|\s)'+key+r'=([^\s]+)',details)
  if match:result[key]=match.group(1)
print(json.dumps(result))
'''
r=subprocess.run(scope['ssh'],input=remote,capture_output=True,timeout=60)
with (D/'parent.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'parent.stderr.private.txt').open('xb') as f:f.write(r.stderr)
value=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:value['value']=json.loads(r.stdout)
with (D/'parent-observation.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
v=value.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,state=v.get('state'),node_count=v.get('node_count'),EndTime=v.get('EndTime'),busy_measured=False,new_steps=0)))
