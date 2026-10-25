import pathlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';assert subprocess.run(['bash','-n',str(i/'GLM_fragment_CPU_r9048_inside.sh')]).returncode==0
exec(compile((i/'launch_GLM_fragment_CPU_r9048.py').read_bytes(),'launch_GLM_fragment_CPU_r9048.py','exec'))
