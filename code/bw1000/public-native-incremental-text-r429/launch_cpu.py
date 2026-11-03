"""One claimed CPU job. Reconcile original label/intent instead of replay."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=Path(__file__).resolve().parent
LABEL='Native-r429-incremental-text-checkpoint-CPU-v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',a.publication)and re.fullmatch('[0-9a-f]{64}',a.source_sha)
    assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==a.source_sha
    for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():
        assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h,n
    rows=subprocess.check_output(['sacct','-S','2026-10-31','-j','910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode'],text=True,timeout=25)
    assert not any(v.split('|')[1]==LABEL for v in rows.splitlines()if'|'in v),'Original same-label job must be consumed, never replayed'
    parent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20)
    assert 'JobState=RUNNING'in parent and 'a14r2n20'in parent
    intent=R/'results/Native429-CPU-v1-intent.json'
    assert not intent.exists()and not(R/'candidates/public-native-incremental-text-r429-CPU-v1').exists()
    with intent.open('x')as f:json.dump(dict(label=LABEL,source_manifest_SHA256=a.source_sha,publication=a.publication,epoch=time.time(),GPU=0,Model=0),f,indent=2)
    env=dict(os.environ,WAVE_INSIDE_R2323='public-native-incremental-text-r429/CPU_inside.sh',R429_CPU_SOURCE_SHA256=a.source_sha)
    argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:05:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export=ALL','bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']
    with(R/'logs'/(LABEL+'-client.log')).open('x')as out:
        job=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(CPU_submitted=True,owned_srun_PID=job.pid,label=LABEL,publication=a.publication,source_manifest_SHA256=a.source_sha,GPU=0,Model=0,argv=argv)))

if __name__=='__main__':main()
