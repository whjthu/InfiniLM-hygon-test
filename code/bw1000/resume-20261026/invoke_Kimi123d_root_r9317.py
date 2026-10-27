import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-ctx2048-r123d/launch_CPU_r123d.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='611608e44b932bb8b272ec2fbbff8be79c81b916777cd02347054869e46578ff'
sys.argv=[str(p)];runpy.run_path(str(p),run_name='__main__')
