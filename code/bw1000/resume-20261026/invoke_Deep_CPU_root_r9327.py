import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Deep-prefix-native-r128/launch_CPU_r128.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='f5796906c6f69c7202ddf17c14b5c2cac44aad277498233fd0b9b51b440e8f43'
sys.argv=[str(p)];runpy.run_path(str(p),run_name='__main__')
