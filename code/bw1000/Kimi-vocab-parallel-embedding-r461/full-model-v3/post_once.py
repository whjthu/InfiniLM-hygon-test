"""One independent CPU-only four-node post after the original handles end."""
from pathlib import Path
import argparse,json,os,re,subprocess
import controller as c
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args(argv)
 c.verify(a.manifest_sha)
 watch=c.R/'results'/(c.LABEL+'-'+c.RUN+'-watch.json');w=json.loads(watch.read_bytes());assert w['completed']and w['source_manifest_SHA256']==a.manifest_sha
 rows=c.rows();expected={}
 for label,step in [(c.LABEL+'-'+c.RUN,w['actual_GPU']),(c.LABEL+'-'+c.RUN+'-keeper',w['CPU_keeper'])]:
  found=[r for r in rows if r[0]==step and r[1]==label];assert len(found)==1 and found[0][2].split()[0].rstrip('+')in c.TERMINAL,(label,found)
 for n,node in enumerate(c.NODES):
  path=c.host(c.config()['runtime_directory'])/('node%d.json'%n)
  if path.exists():
   v=json.loads(path.read_bytes());assert v['run']==c.RUN and v['source_manifest_SHA256']==a.manifest_sha;expected[node]=[v['PID']]
  else:expected[node]=[] # Missing report remains unknown; passive PID/KFD/port post still executes.
 intent=c.R/'results'/(c.LABEL+'-'+c.RUN+'-independent-post-intent.json');c.publish(intent,dict(source_manifest_SHA256=a.manifest_sha,CPU_only=True,GPU=0,expected=expected))
 env=c.clean_environment(os.environ);env['K461_PIDS']=json.dumps(expected)
 cmd=['srun','--jobid=910380','--overlap','--exact','--immediate=15','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(c.NODES),'-t','00:00:30','--job-name='+c.LABEL+'-'+c.RUN+'-independent-post','python3','-B',str(c.D/'idle.py')]
 z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=45)
 value=dict(CPU_only=True,GPU=0,argv=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr)
 c.publish(c.R/'results'/(c.LABEL+'-'+c.RUN+'-independent-post.json'),value)
 records=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')]
 assert z.returncode==0 and len(records)==4 and {v['node']for v in records}==set(c.NODES)
 print(json.dumps(dict(passed=True,CPU_only=True,GPU=0,nodes=records)))
if __name__=='__main__':main()
