"""Published finite CPU claim; no Model/SDK/Runtime/GPU library is executed."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def launch(directory,expected,publication):
 d=Path(directory);assert re.fullmatch('[0-9a-f]{64}',expected)and re.fullmatch('[0-9a-f]{40}',publication)
 assert sha(d/'SOURCE_MANIFEST.json')==expected;m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());assert not m['GPU_GO']and all(sha(d/n)==h for n,h in m['files_SHA256'].items())
 label='GLM-value-full-fp32-leaves-r446-CPU-v1'
 p=subprocess.run(['sacct','-S','2026-11-03','-j','910407','-n','-P','-o','JobIDRaw,JobName%120,State'],capture_output=True,text=True,timeout=30);assert p.returncode==0 and not any(s.split('|')[1]==label for s in p.stdout.splitlines()if '|'in s)
 p=subprocess.run(['scontrol','show','job','910407','-o'],capture_output=True,text=True,timeout=30);assert p.returncode==0 and 'JobState=RUNNING'in p.stdout and'UserId=aclsylqidf('in p.stdout
 nodes=re.search(r'(?:^| )NodeList=([^ ]+)',p.stdout).group(1);v=subprocess.run(['scontrol','show','hostnames',nodes],capture_output=True,text=True,timeout=30);assert v.returncode==0 and'a14r2n20'in v.stdout.splitlines()
 wrapper=R/'incoming/Deep-paired-quality-r375/controller/resource_wave_host_alias_outer.sh';assert sha(wrapper)=='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d'
 intent=R/'control'/(label+'-intent.json')
 with intent.open('x')as f:json.dump(dict(source_manifest_SHA256=expected,publication_commit=publication,GPU=0,CPU_cores=4,host_mem_MiB=8192,hard_s=300,no_retry=True),f)
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 env.update(WAVE_INSIDE_R2323='GLM-value-full-fp32-leaves-r446/CPU_inside.sh',GLM446_SOURCE_SHA256=expected)
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:05:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'bash',str(wrapper),'probe']
 with(R/'logs'/(label+'-srun.log')).open('x')as log:process=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return dict(CPU_submitted=True,GPU=0,label=label,PID=process.pid,source_manifest_SHA256=expected,intent=str(intent),publication_commit=publication)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);args=a.parse_args();print(json.dumps(launch(Path(__file__).resolve().parent,args.manifest_sha,args.publication_commit)))
