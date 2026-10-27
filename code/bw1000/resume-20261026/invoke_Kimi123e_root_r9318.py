import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-ctx2048-r123e/launch_CPU_r123e.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6352bed0d890f330926f07ed8530dde30e36a630f5db7eee7e86d9210254b7ac'
sys.argv=[str(p)];runpy.run_path(str(p),run_name='__main__')
