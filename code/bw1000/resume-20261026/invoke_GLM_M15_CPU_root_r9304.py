import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM-M15-component-r121/launch_CPU_only_r121.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6d4c87f917f05065ed2ac83bcba8b8ca57a94b79e3269b918d247eadd9f6bbe5'
sys.argv=[str(p),'--parent','910407','--node','a14r2n20','--manifest-sha','d14a99ebc95f9d81318830b3cb964720a7562daf74723026e778f81011926646']
runpy.run_path(str(p),run_name='__main__')
