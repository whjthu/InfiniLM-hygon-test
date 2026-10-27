import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Deep-agent-P0-control-r143c/invoke_once_r143.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=="41b9eeee5282984be0a8cbca0f6b4d9b4f184cf494c2d485e194fe76fedaebb8"
sys.path.insert(0,str(p.parent));runpy.run_path(str(p),run_name="__main__")
