import pathlib,hashlib
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/dispatch_GLM_resident_exact_Model_r8866.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='d3da03487ab1eac39a9e18931089caf1230f762e692bfaf5fc6327442883b8c3'
exec(compile(p.read_text(),str(p),'exec'))
