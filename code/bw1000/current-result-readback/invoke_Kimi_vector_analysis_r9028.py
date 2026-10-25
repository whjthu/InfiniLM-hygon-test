import pathlib,base64,zlib,subprocess,json
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
# Analysis source is passed by the root stager before invocation.
p=r/'incoming/analyze_Kimi_crossover_vectors_r9028.py';assert p.exists()
z=subprocess.run(['srun','--jobid=910407','--overlap','--exact','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-vector-analysis-r9028','python3',str(p)],text=True,capture_output=True,timeout=170)
print(json.dumps({'CPU_only':True,'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr}));assert z.returncode==0
