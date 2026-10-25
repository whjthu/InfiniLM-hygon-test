import pathlib,hashlib
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/dispatch_GLM_resident_exact_Model_r8872.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='75a4858765fb8507299676aab171d1d7e446e4fd12a7a308916b89d365318a8e'
exec(compile(p.read_text(),str(p),'exec'))
