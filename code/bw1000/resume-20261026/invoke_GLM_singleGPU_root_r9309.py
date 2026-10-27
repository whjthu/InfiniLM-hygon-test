import pathlib,hashlib,runpy,sys
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'incoming/GLM-M15-controller-r123/launch_CPU_keeper_r123.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='f7534840daa83ca14a8ee7013765d576c276bdde1c4a24e715b70038859e2151'
sys.argv=[str(p),'--manifest-sha','4a4796d29e1d2d6c4551283b40a6427ae76d497656aec8ab8f7cf71762ae769c','--go-file',str(r/'incoming/GLM-M15-controller-r123/root-GO-after-publication.json')]
runpy.run_path(str(p),run_name='__main__')
