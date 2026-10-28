"""Preserve project include spelling/identity in a private dependency mirror."""
import pathlib,hashlib,os,shlex
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def spelled_deps(output):
 return {str(pathlib.Path(os.path.normpath(x)).absolute())for x in shlex.split(output.replace('\\\n',' ').split(':',1)[1])}
class Mirror:
 def __init__(self,root,build,spelled,new_header,old_header_sha):
  self.project=pathlib.Path(root);self.root=pathlib.Path(build)/'project-mirror';self.root.mkdir();self.roots=list(dict.fromkeys([self.project.resolve(),pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014').resolve(),self.project]));self.back={};self.pins={};self.guard_new_sha=sha(new_header);self.original_sha={};copied={};aliases=[]
  for name in sorted(spelled):
   p=pathlib.Path(name);resolved=p.resolve()
   if self.relative(p)is None and self.relative(resolved)is None:continue
   assert p.is_file()and p.stat().st_size<16*1024**2,('unexpected project preprocessor input',str(p))
   original=self.logical(resolved);primary=self.target(original);alias=self.target(p)
   if original not in copied:
    primary.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes();guard=p.name=='native_sparse_cache_r22.hpp'or resolved.name=='native_sparse_cache_r22.hpp'
    if guard:assert hashlib.sha256(raw).hexdigest()==old_header_sha;raw=pathlib.Path(new_header).read_bytes()
    primary.write_bytes(raw);copied[original]=primary;self.back[str(primary.resolve())]=original;self.pins[str(primary.resolve())]=sha(primary);self.original_sha[original]=sha(p)
   assert sha(primary)==(self.guard_new_sha if p.name=='native_sparse_cache_r22.hpp'or resolved.name=='native_sparse_cache_r22.hpp'else sha(p))
   aliases.append((alias,primary,original))
  # Individual aliases point inside the private mirror, never at old headers.
  # This also retains #pragma-once identity for multiple old symlink spellings.
  alias_count=0
  for alias,primary,original in aliases:
   if alias==primary:continue
   alias.parent.mkdir(parents=True,exist_ok=True)
   if alias.exists()or alias.is_symlink():assert alias.resolve()==primary.resolve(),('ambiguous alias identity',str(alias),str(primary))
   else:alias.symlink_to(primary);alias_count+=1
  self.summary={'copied_project_files':len(copied),'private_file_aliases':alias_count,'copied_bytes':sum(p.stat().st_size for p in copied.values()),'only_changed_basename':'native_sparse_cache_r22.hpp','ordinary_project_bytes_unchanged':True,'project_dependencies_must_not_escape_private_mirror':True}
 def relative(self,p):
  p=pathlib.Path(os.path.normpath(str(p)))
  for root in self.roots:
   try:return p.relative_to(root)
   except ValueError:pass
  return None
 def logical(self,p):
  rel=self.relative(p);return str(self.project/rel)if rel is not None else str(pathlib.Path(p))
 def target(self,p):
  rel=self.relative(p);assert rel is not None,('not project path',str(p));out=self.root/rel;assert out.is_relative_to(self.root);return out
 def rewrite_flags(self,flags):
  out=[];i=0
  while i<len(flags):
   x=flags[i]
   if x in ['-I','-iquote','-isystem']:
    path=pathlib.Path(flags[i+1]);mapped=self.target(path)if self.relative(path)is not None else path
    if self.relative(path)is not None:mapped.mkdir(parents=True,exist_ok=True)
    out.extend([x,str(mapped)]);i+=2;continue
   for prefix in ['-I','-iquote','-isystem']:
    if x.startswith(prefix)and len(x)>len(prefix):
     path=pathlib.Path(x[len(prefix):])
     if self.relative(path)is not None:
      mapped=self.target(path);mapped.mkdir(parents=True,exist_ok=True);x=prefix+str(mapped)
     break
   out.append(x);i+=1
  return out
 def normalize_and_check(self,dependencies):
  normalized=set()
  for name in dependencies:
   p=pathlib.Path(name).resolve();key=str(p)
   if p.is_relative_to(self.root.resolve()):
    assert key in self.back,('unrecorded private dependency',key);assert sha(p)==self.pins[key];normalized.add(self.back[key])
   else:
    assert self.relative(p)is None,('project header escaped mirror back into original tree',key);normalized.add(key)
  return normalized
