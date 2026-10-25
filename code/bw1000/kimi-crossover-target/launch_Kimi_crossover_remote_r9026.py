import pathlib,subprocess,json
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
p=subprocess.run(['python3',str(r/'incoming/launch_Kimi_crossover_Model_r9026.py')],capture_output=True,text=True,timeout=30)
print(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}));assert p.returncode==0
