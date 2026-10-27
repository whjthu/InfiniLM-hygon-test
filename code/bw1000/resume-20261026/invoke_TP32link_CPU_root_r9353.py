import pathlib,hashlib,runpy
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-TP32-constructor-link-r120h/launch_CPU_r120h.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='25becf8d09a4fcd97e4b33074c4fe6282e48f75b850bbd42368a29edd799d981';runpy.run_path(str(p),run_name='__main__')
