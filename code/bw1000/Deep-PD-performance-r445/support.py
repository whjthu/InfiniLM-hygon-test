"""Exact published root primitives shared by role, controller and CPU consumer."""
from pathlib import Path
import json,hashlib,importlib.util,sys

def load(directory):
 directory=Path(directory);deps=json.loads((directory/'DEPENDENCIES.json').read_bytes());parent=directory/'root_support';loaded={}
 for name,digest in deps['root_modules_SHA256'].items():
  path=parent/(name+'.py');assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
  if name in sys.modules:
   module=sys.modules[name];assert Path(module.__file__).resolve()==path.resolve()
  else:
   spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  loaded[name]=module
 return loaded
