"""One owned CPU keeper; original v7 controller alone submits the eight-GPU step."""
from pathlib import Path
import binding
import argparse,hashlib,json,os,re,subprocess

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
S=Path(__file__).resolve().parent;LABEL='Deep-T1-MoE-activation-quant-r438-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=lambda p:R/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publication-commit',required=True);parser.add_argument('--manifest-sha',required=True)
    args=parser.parse_args();assert re.fullmatch('[0-9a-f]{40}',args.publication_commit)
    manifest=S/'GPU_MANIFEST.json';assert sha(manifest)==args.manifest_sha
    m=json.loads(manifest.read_bytes());assert m['GPU_GO']is True
    assert all(sha(S/n)==h for n,h in m['source_SHA256'].items())
    binding.validate(m,S,host)
    assert not (R/'results'/(LABEL+'-watch.json')).exists()
    intent=R/'results'/(LABEL+'-intent.json')
    with intent.open('x')as f:
        json.dump(dict(publication_commit=args.publication_commit,manifest_SHA256=args.manifest_sha,
            CPU_keeper_only=True,GPU_submitted_by_invoker=False,no_retry=True),f,indent=2);f.write('\n')
    env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    command=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20',
        '--gres=none','--mem=4096M','-t','00:15:00','--job-name='+LABEL+'-keeper',
        '--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),
        'timeout','--signal=TERM','--kill-after=15s','890','python3','-B',str(S/'controller.py'),
        '--publication-commit',args.publication_commit,'--manifest-sha',args.manifest_sha]
    with(R/'logs'/(LABEL+'-keeper-client.log')).open('x')as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(CPU_keeper_submitted=True,PID=process.pid,label=LABEL+'-keeper',
        intent=str(intent),GPU_submitted_by_invoker=False,publication_commit=args.publication_commit)))


if __name__=='__main__':main()
