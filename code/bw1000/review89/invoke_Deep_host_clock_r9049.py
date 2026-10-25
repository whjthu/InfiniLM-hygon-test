import pathlib,subprocess,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=subprocess.run(['python3',str(r/'incoming/launch_Deep_host_clock_component_r9049.py')],capture_output=True,text=True,timeout=20)
assert p.returncode==0,(p.stdout,p.stderr)
print(p.stdout)
