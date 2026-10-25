import pathlib,hashlib
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/dispatch_GLM_resident_exact_Model_r8868.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='64bc6f34ffb6a17734ac3ef486379330d6e45496dc53f3a4f4bc5e7f069a6fb4'
exec(compile(p.read_text(),str(p),'exec'))
