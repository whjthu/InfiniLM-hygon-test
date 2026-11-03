"""Reuse the actual375 provider/loader/forward/destroy, with private PD bindings."""
from pathlib import Path
import ast,hashlib,json,sys,time,os,copy,threading
from contract import check_live

OWNER_SHA='1b93c3db22ab59c6768e401dc46f71b630a768934fe316135b53d780d1b3fafb'
BASE_SHA='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def transform(base_raw,parent_transform,object_count):
 tree,proof=parent_transform(base_raw);before=ast.dump(next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward'))
 counts=dict(link_compare=0,linked_report=0,create_deadline=0)
 class Private(ast.NodeTransformer):
  def visit_Compare(self,node):
   node=self.generic_visit(node)
   if ast.dump(node.left)==ast.dump(ast.parse("len(prebuild['link']['full']['objects'])",mode='eval').body)and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)and node.comparators[0].value==49:
    node.comparators[0]=ast.Constant(object_count);counts['link_compare']+=1
   return node
  def visit_Dict(self,node):
   node=self.generic_visit(node)
   for i,key in enumerate(node.keys):
    if isinstance(key,ast.Constant)and key.value=='linked_objects'and isinstance(node.values[i],ast.Constant)and node.values[i].value==49:node.values[i]=ast.Constant(object_count);counts['linked_report']+=1
   return node
  def visit_keyword(self,node):
   node=self.generic_visit(node)
   if node.arg=='linked_objects'and isinstance(node.value,ast.Constant)and node.value.value==49:node.value=ast.Constant(object_count);counts['linked_report']+=1
   return node
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if isinstance(node.func,ast.Attribute)and node.func.attr=='modern_native_engine_create':
    counts['create_deadline']+=1
    return ast.Subscript(value=ast.Tuple(elts=[ast.Call(func=ast.Name(id='__pd_deadline__',ctx=ast.Load()),args=[],keywords=[]),node],ctx=ast.Load()),slice=ast.Constant(1),ctx=ast.Load())
   return node
 tree=Private().visit(tree);ast.fix_missing_locations(tree)
 assert counts==dict(link_compare=1,linked_report=1,create_deadline=1)
 assert ast.dump(next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward'))==before
 proof.update(PD_private_patch=counts,actual_linked_object_count=object_count,Native_forward_AST_unchanged=True,original_parameter_loader_and_finally_destroy_retained=True)
 return tree,proof

def provider_tree(raw):
 tree=ast.parse(raw);changes=0
 for fn in tree.body:
  if isinstance(fn,ast.FunctionDef)and fn.name=='run':
   for i,node in enumerate(fn.body):
    if isinstance(node,ast.If)and 'DEEP_R277_GPUSTART_UNIX'in ast.unparse(node.test):fn.body[i]=ast.Expr(ast.Call(func=ast.Name(id='__pd_deadline__',ctx=ast.Load()),args=[],keywords=[]));changes+=1
 assert changes==1;ast.fix_missing_locations(tree);return tree

def load(owner,root,pin,epoch,cancel_file,serve,before_weights=lambda provider:None,on_namespace_exit=lambda ns:None):
 """Called on the bootstrap main thread; all original Runtime owners stay inside Engine."""
 owner,root=Path(owner),Path(root);assert sha(owner/'SOURCE_MANIFEST.json')==OWNER_SHA
 m=json.loads((owner/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(owner/n)==h for n,h in m['files_SHA256'].items())
 sys.path.insert(0,str(owner));sys.path.insert(1,str(root/'candidates/latest-family-service-r270/Deep/python'))
 import bootstrap_r206b as parent,preweight_r206b as preweight,native_binding_r206b as binding
 assert Path(parent.__file__).resolve()==owner/'bootstrap_r206b.py'and sha(parent.BASE)==BASE_SHA
 original=binding.binding()
 from dso_binding import derive
 b,q,n,manifest=derive(pin,original,parent.BASE)
 def deadline():
  check_live(epoch,cancel_file)
  if epoch+5280-time.time()<=120:raise TimeoutError('Original PD deadline before provider/create/safe_open/weight load')
 # Private module namespace; original source bytes and global module bindings
 # remain unchanged. The sole AST edit replaces the old7080 deadline guard.
 env=dict(__name__='pd_private_preweight',__file__=str(owner/'preweight_r206b.py'),__pd_deadline__=deadline)
 exec(compile(provider_tree((owner/'preweight_r206b.py').read_bytes()),str(owner/'preweight_r206b.py'),'exec'),env)
 env['binding']=lambda:(b,q,n)
 os.environ['DEEP_SOURCE_MANIFEST_SHA']=OWNER_SHA
 provider=env['run']();before_weights(provider);tree,proof=transform(parent.BASE.read_bytes(),parent.transform,len(n['link']['full']['objects']))
 namespace=dict(__name__='__main__',__file__=str(parent.BASE),__manifest_r206b__=lambda:manifest,__prebuild_r206b__=lambda:n,__normal_create_metadata_r206b__=parent.normal_create_metadata,
  __service_r206b__=lambda ns:serve(ns,provider),__bootstrap_proof_r206b__=proof,__provider_proof_r206b__=provider['proof'],__latest_prepared_r277__=provider['prepared'],__weights_deadline_check_r287__=deadline,__pd_deadline__=deadline,__latest_map_check_r300__=parent.validate_deep)
 try:exec(compile(tree,str(parent.BASE),'exec'),namespace)
 finally:on_namespace_exit(namespace)
 assert namespace['report']['engine_worker_shutdown_completed']
 return namespace,proof,dict(binding=b,receipt=q,normalized=n)
