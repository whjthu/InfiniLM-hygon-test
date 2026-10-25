import pathlib,json,hashlib,runpy,sys,os,signal
r=pathlib.Path('/refactor');m=json.loads((r/'incoming/Kimi-attnres-C9-w1-manifest-r8989.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha('/shared/Kimi-K3/config.json')==m['config_SHA256']
assert sha(m['allocation_source'])==m['allocation_source_SHA256']
for n,h in m['incoming_source_SHA256'].items():assert sha(r/'incoming'/n)==h,n
for n,h in m['existing_remote_SHA256'].items():assert sha(r/n)==h,n
assert os.environ.get('SLURM_GPUS_ON_NODE','1') in ['1','dcu:1']
initial_affinity=sorted(os.sched_getaffinity(0));assert len(initial_affinity)>=8
assert os.environ.get('SLURM_CPUS_PER_TASK')=='8'
assert os.environ.get('INFINILM_OWNER_CPU_WIDTH_R2392')=='1'
signal.alarm(175)
sys.argv=[str(r/'incoming/run_attnres_front_C9_r8984.py'),'--library',m['native'],'--sha256',m['native_SHA256'],'--cpu-receipt',str(r/m['cpu_receipt_path']),'--cpu-receipt-sha256',m['existing_remote_SHA256'][m['cpu_receipt_path']],'--device','0','--output',str(r/'results'/('kimi-attnres-C9-w1-r8989-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json'))]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
    output=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
    if output.exists():
        result=json.loads(output.read_text());result['resource_fix']={'prior_failed_step':'910374.938','prior_pending_label':'kimi-attnres-C9-r8987','initial_affinity':initial_affinity,'final_affinity':sorted(os.sched_getaffinity(0)),'SLURM_CPUS_PER_TASK':8,'owner_width':1};q=output.with_suffix('.tmp');q.write_text(json.dumps(result,indent=2));q.replace(output)
