import pathlib,hashlib,runpy,sys
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/launch_CPU_r120c.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='327e589a1dedcbd52379113e228f068f00cc65c66b9947a08a03d26ba14908f3'
sys.argv=[str(p)];runpy.run_path(str(p),run_name='__main__')
