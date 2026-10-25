import pathlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';assert subprocess.run(['bash','-n',str(i/'GLM_M16_bridge_CPU_r9056_inside.sh')]).returncode==0
exec(compile((i/'launch_GLM_M16_bridge_CPU_r9056.py').read_bytes(),'launch_GLM_M16_bridge_CPU_r9056.py','exec'))
