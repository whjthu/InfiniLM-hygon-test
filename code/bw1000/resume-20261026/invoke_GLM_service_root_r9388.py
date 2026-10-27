import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM-agent-service-r143b/launch_CPU_keeper_r143b.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='a7e262da64d193082d00b6382f94e81e38d1e4458d32994dd7850560187c243c'
sys.path.insert(0,str(p.parent));sys.argv=[str(p),"--manifest-sha",'39403de9502f544d613159331db67de1274d05b10852ac1729ff4267c2e7a21c',"--go-file",'/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM-agent-service-r143b/root-GO-r9388.json'];runpy.run_path(str(p),run_name="__main__")
