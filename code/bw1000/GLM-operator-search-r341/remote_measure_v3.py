import pathlib,hashlib,runpy,sys
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/GLM-operator-search-r341/repair_v3'
assert hashlib.sha256((p/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='6a42fb9db1d974667cb18201d555021065e19e1f3ebfe507e2f7f10dcad25823'
sys.argv=[str(p/'controller.py'),'d29a2db2d531e15349f8eff5bc5c278e8e26ee7e','measure'];runpy.run_path(str(p/'controller.py'),run_name='__main__')


