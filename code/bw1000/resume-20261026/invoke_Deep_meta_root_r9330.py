import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Deep-prefix-native-CPU-metadata-r128b/launch_CPU_r128b.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='c33e91377c48645058cc16fb2e792a1d5d985c30f92cc193b06bb9625b6a99e4'
sys.argv=[str(p)];runpy.run_path(str(p),run_name='__main__')
