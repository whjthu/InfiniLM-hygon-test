"""Replace exactly the obsolete five-node mapping audit in pinned legacy BASE."""
import ast,hashlib
OLD_BLOCK_AST_SHA256='f09d78bdc2391bc03f75242f76eddf5862417c3fcb8d43077f353dd3ff191db0'
def replace_mapping_audit(tree):
 hits=[]
 for parent in ast.walk(tree):
  body=getattr(parent,'body',None)
  if not isinstance(body,list):continue
  for i,node in enumerate(body):
   if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='loaded_paths'for t in node.targets):
    block=body[i:i+5];assert len(block)==5
    digest=hashlib.sha256(ast.dump(ast.Module(body=block,type_ignores=[])).encode()).hexdigest();assert digest==OLD_BLOCK_AST_SHA256,'Legacy binding block changed; refuse patch'
    hits.append((body,i,block))
 assert len(hits)==1,'Expected exactly one historical mapping block'
 body,i,old=hits[0]
 new=ast.parse("__common_proof_r300__=__latest_map_check_r300__(prebuild['actual_DTK_bindings'])\nreport['actual_new_runtime_paths']=__common_proof_r300__['paths']\nreport['actual_common_library_audit']=__common_proof_r300__").body
 body[i:i+5]=new
 return {'replaced_legacy_binding_blocks':1,'old_AST_SHA256':OLD_BLOCK_AST_SHA256,'new_exact_set_and_SHA_required':True}
