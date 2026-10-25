import pathlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';assert subprocess.run(['bash','-n',str(i/'GLM_567_cohort_CPU_r9057_inside.sh')]).returncode==0
exec(compile((i/'launch_GLM_567_cohort_CPU_r9057.py').read_bytes(),'launch_GLM_567_cohort_CPU_r9057.py','exec'))
