import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-agent-service-Model-r144c/launcher/launch_CPU_keeper_r144c.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='308885b6a44d430bbd12673725a59831e667478984e3d1875a729de7ed9a0575'
sys.path.insert(0,str(p.parent));sys.argv=[str(p),"--manifest-sha",'924dceef58c46007b77a8576eb0820617261435bdd1b9d1e22d2a9e7ab94648b',"--go-file",'/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-agent-service-Model-r144c/root-GO-r9388.json'];runpy.run_path(str(p),run_name="__main__")
