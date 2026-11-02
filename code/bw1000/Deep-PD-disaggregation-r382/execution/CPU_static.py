"""Real final receipt/objectgraph derivation and exact original loader AST, no dlopen."""
from pathlib import Path
import sys,os,json,ast,tempfile,hashlib,copy,time,importlib.util
from qualified_loader import transform,provider_tree,OWNER_SHA,BASE_SHA
from contract import check_live

def run():
 d=Path(__file__).resolve().parent;pin=json.loads((d/'NATIVE_PIN.json').read_bytes());assert pin['ready']is True
 owner=Path('/refactor/incoming/Deep-paired-quality-r375/owner');sys.path.insert(0,str(owner))
 assert hashlib.sha256((owner/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==OWNER_SHA
 import bootstrap_r206b as parent,native_binding_r206b as original
 from dso_binding import derive
 previous=original.binding();b,q,n,m=derive(pin,previous,parent.BASE)
 assert len(n['link']['full']['objects'])==52
 tree,proof=transform(parent.BASE.read_bytes(),parent.transform,52)
 assert proof['Native_forward_AST_unchanged']and proof['original_parameter_loader_and_finally_destroy_retained']
 base_tree=ast.parse(parent.BASE.read_bytes());path_node=next(n.value for n in base_tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='path'for t in n.targets))
 paths=[eval(compile(ast.Expression(path_node),'<actual BASE path>','eval'),dict(r=Path('/refactor'),replica=alias,tag='910376.999999'))for alias in ['n01','n04']]
 assert paths[0]!=paths[1]and all('910376.999999'in str(p)for p in paths)
 shell=(d/'GPU_inside.sh').read_text();assert '0) export DEEP_REPLICA_R9278=n01'in shell and '1) export DEEP_REPLICA_R9278=n04'in shell
 provider=provider_tree((owner/'preweight_r206b.py').read_bytes());ast.fix_missing_locations(provider)
 with tempfile.TemporaryDirectory(prefix='pd-static-')as tmp:
  cancel=Path(tmp)/'cancel';cancel.write_text('{}')
  def deadline():check_live(time.time(),cancel)
  env=dict(__name__='cpu_actual_provider_prefix',__file__=str(owner/'preweight_r206b.py'),__pd_deadline__=deadline)
  exec(compile(provider,str(owner/'preweight_r206b.py'),'exec'),env);env['binding']=lambda:(b,q,n);os.environ['DEEP_SOURCE_MANIFEST_SHA']=OWNER_SHA
  try:env['run']()
  except RuntimeError as e:assert 'canceled'in str(e)
  else:raise AssertionError('Canceled actual provider reached Torch')
 assert 'torch'not in sys.modules
 namespace_file=Path('/refactor/incoming/Deep-paired-quality-r375/controller/namespace_binding.py')
 spec=importlib.util.spec_from_file_location('pd_cpu_actual_namespace',namespace_file);namespace=importlib.util.module_from_spec(spec);spec.loader.exec_module(namespace)
 namespace_proof=namespace.verify_namespace(Path('/refactor'),namespace_file.parent)
 negative=[]
 for key in ['object_graph_receipt','CPU_receipt_SHA256']:
  bad=dict(pin);bad.pop(key)
  try:derive(bad,previous,parent.BASE)
  except KeyError:negative.append(key)
  else:raise AssertionError('Missing actualbinding field accepted')
 return dict(completed=True,passed=True,GPU=0,Native_dlopen=False,weights_opened=False,actual_new_Native_binding_derived=True,objects=52,library=q['library'],library_SHA256=q['library_SHA256'],object_graph_receipt_SHA256=pin['object_graph_receipt_SHA256'],loader_AST=proof,actual_provider_cancel_before_Torch=True,missing_field_negatives_before_load=negative,same_step_distinct_legacy_paths=list(map(str,paths)),actual_qualified_namespace_prefix=namespace_proof)
if __name__=='__main__':print(json.dumps(run()))
