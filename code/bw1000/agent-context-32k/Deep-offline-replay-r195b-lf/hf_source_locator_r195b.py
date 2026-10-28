"""Bounded installed-package AST resolution; no Transformers/Torch import."""
import ast,hashlib,pathlib
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def locate(root):
 root=pathlib.Path(root).resolve();trees={};evidence={};seen=set()
 def tree(path):
  path=path.resolve();assert path.is_relative_to(root)and path.suffix=='.py'and path.stat().st_size<=2*1024*1024,'HF source boundary exceeded'
  if path not in trees:
   assert len(trees)<40,'HF source file bound exceeded';trees[path]=ast.parse(path.read_bytes());evidence[path.relative_to(root).as_posix()]=sha(path)
   assert sum(p.stat().st_size for p in trees)<=16*1024*1024,'HF source byte bound exceeded'
  return trees[path]
 def imports(path,name):
  targets=[]
  for n in ast.walk(tree(path)):
   if isinstance(n,ast.ImportFrom):
    for a in n.names:
     if (a.asname or a.name)==name and a.name!='*':
      if n.level:
       base=path.parent
       for _ in range(n.level-1):base=base.parent
       p=base.joinpath(*(n.module or '').split('.')).with_suffix('.py')
      elif n.module and n.module.startswith('transformers.'):
       p=root.joinpath(*n.module.split('.')[1:]).with_suffix('.py')
      else:continue
      if p.is_file():targets.append((p,a.name))
  return targets
 def resolve(path,name):
  key=(path.resolve(),name)
  if key in seen:return []
  seen.add(key);t=tree(path);matches=[]
  for n in t.body:
   if isinstance(n,ast.ClassDef)and n.name==name:
    if any(isinstance(f,ast.FunctionDef)and f.name=='_decode'for f in n.body):matches.append((path,n.name))
    else:
     for base in n.bases:
      if isinstance(base,ast.Name):matches.extend(resolve(path,base.id))
   if isinstance(n,(ast.Assign,ast.AnnAssign)):
    targets=n.targets if isinstance(n,ast.Assign)else[n.target]
    if any(isinstance(x,ast.Name)and x.id==name for x in targets)and isinstance(n.value,ast.Name):matches.extend(resolve(path,n.value.id))
  for p,target in imports(path,name):matches.extend(resolve(p,target))
  return matches
 init=root/'__init__.py';candidates=resolve(init,'PreTrainedTokenizerFast')
 # Lazy export lists may omit executable aliases from __init__. Resolve only
 # direct tokenization-module definitions/aliases; never recurse package trees.
 if not candidates:
  paths=sorted(p for p in root.iterdir()if p.is_file()and p.suffix=='.py'and'tokeniz'in p.name);assert len(paths)<=32,'HF direct tokenization module bound exceeded'
  for p in paths:
   if any((isinstance(n,ast.ClassDef)and n.name=='PreTrainedTokenizerFast')or(isinstance(n,(ast.Assign,ast.AnnAssign))and any(isinstance(x,ast.Name)and x.id=='PreTrainedTokenizerFast'for x in(n.targets if isinstance(n,ast.Assign)else[n.target])))for n in tree(p).body):candidates.extend(resolve(p,'PreTrainedTokenizerFast'))
 candidates=sorted(set((p.resolve(),c)for p,c in candidates));assert len(candidates)==1,'HF exact fast-tokenizer alias/_decode implementation unresolved or ambiguous; stop before token replay'
 path,cls=candidates[0];return path,cls,{'resolved_class_SHA256':hashlib.sha256(cls.encode()).hexdigest(),'source_files_SHA256':evidence,'source_files_read':len(trees),'package_imported':False,'recursive_scan':False}
