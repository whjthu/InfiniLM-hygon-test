"""One CPU-only diagnostic step, never a model launch/fresh check."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=Path(__file__).resolve().parent
LABEL='Kimi637-loader-passive-probe-v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication',required=True);a=p.parse_args()
    raw=(D/'SOURCE_MANIFEST.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==a.manifest_sha
    for n,h in json.loads(raw)['files_SHA256'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
    intent=R/'results'/(LABEL+'-intent.json');out=R/'results'/(LABEL+'-srun.stdout');err=R/'results'/(LABEL+'-srun.stderr')
    q=dict(source_manifest_SHA256=a.manifest_sha,publication=a.publication,CPU_only=True,GPU=0,expected_original_GPU='910380.637',started_unix=time.time(),UID=os.getuid(),completed=False)
    with intent.open('x')as f:json.dump(q,f,indent=2)
    args=['srun','--jobid=910380','--overlap','--exact','--immediate=10','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=256M','-w','a14r2n08,a14r2n09,a14r2n10,a14r2n12','-t','00:00:30','--job-name='+LABEL,'env','CUDA_VISIBLE_DEVICES=-1','HIP_VISIBLE_DEVICES=-1','ROCR_VISIBLE_DEVICES=-1','python3','-B',str(D/'probe.py')]
    started=time.monotonic()
    with out.open('xb')as stdout,err.open('xb')as stderr:
        proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        q.update(srun_PID=proc.pid,owned_PGID=proc.pid,argv=args)
        (R/'results'/(LABEL+'-owned.json')).write_text(json.dumps(q,indent=2)+'\n')
        try:rc=proc.wait(timeout=max(1,50-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            q['timeout']=True
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:rc=proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                rc=proc.wait(timeout=3)
    q.update(returncode=rc,elapsed_s=time.monotonic()-started,completed=True,stdout_path=str(out),stderr_path=str(err),stdout_SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),stderr_SHA256=hashlib.sha256(err.read_bytes()).hexdigest())
    (R/'results'/(LABEL+'-result.json')).write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(q))
if __name__=='__main__':main()
