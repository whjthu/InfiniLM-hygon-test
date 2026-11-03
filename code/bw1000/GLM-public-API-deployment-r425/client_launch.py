"""Known CPU client process identity, then exact published quality client CLI."""
from pathlib import Path
import argparse,hashlib,json,os,socket,sys
def main():
 p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--client',required=True);p.add_argument('--client-sha',required=True);a,rest=p.parse_known_args()
 source=Path(a.client);assert hashlib.sha256(source.read_bytes()).hexdigest()==a.client_sha
 record=Path(a.identity);assert not record.exists()
 with record.open('x')as f:json.dump(dict(PID=os.getpid(),node=socket.gethostname().split('.')[0],actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],client=str(source),client_SHA256=a.client_sha,argv=rest,GPU=0,Native=0),f,indent=2)
 os.execv(sys.executable,[sys.executable,'-B','-u',str(source),*rest])
if __name__=='__main__':main()
