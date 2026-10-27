import pathlib,hashlib,sys,runpy
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM-critical-Model-r133/launch_CPU_keeper_r133.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='bbe29a5354ead0f87de924c0d9c599ff5cef19df27198b6f7601be2ee8d815ed'
sys.argv=[str(p),'--manifest-sha','074aa459e4801b73a40393dd749d2a72911da91586cfdf3ef2d06e4ef3ffbed5','--go-file',str(p.parent/'root-GO-after-CPU455-r9353.json')];runpy.run_path(str(p),run_name='__main__')
