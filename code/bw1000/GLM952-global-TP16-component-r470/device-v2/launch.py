"""Single finite CPU keeper; exact namespaces/label/xintent prevent retries."""
from pathlib import Path
import argparse,json,subprocess,os,time,re,socket
from shared import D,R,OUT,LABEL,H,verify

def main(publication):
 assert re.fullmatch('[0-9a-f]{40}',publication);pin=H(D/'GPU_MANIFEST.json');verify(pin);claim=json.loads((D/'GPU_RESOURCE_CLAIM.json').read_bytes());assert claim['GPU_GO']and not claim['Model_GO'];intent=R/'results'/(LABEL+'-intent.json');launch=R/'results'/(LABEL+'-launch.json');watch=R/'results'/(LABEL+'-watch.json');assert not any(p.exists()for p in [intent,launch,watch,OUT]);z=subprocess.run(['sacct','-j','910371,910407','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert z.returncode==0 and not any(LABEL in x for x in z.stdout.splitlines());a=['srun','--jobid=910407','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c2','--mem=4096M','--gres=none','-w','a14r2n20','-t','00:05:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'env','GLM470_GPU_MF='+pin,'bash',str(D/'keeper.sh')]
 with intent.open('x')as f:json.dump(dict(once=True,source_publication=publication,GPU_manifest_SHA256=pin,claim_SHA256=H(D/'GPU_RESOURCE_CLAIM.json'),epoch=time.time(),HOST=socket.gethostname(),Model=False,weights=False,HTTP=0,argv=a),f)
 with(R/'logs'/(LABEL+'-keeper-srun.log')).open('xb')as f:p=subprocess.Popen(a,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(HOST=socket.gethostname(),owned_keeper_PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],GPU=None,intent=str(intent));launch.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args();main(a.publication)
