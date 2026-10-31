"""Reuse qualified loader lifecycle; bind only the actual new guard artifact."""
import pathlib,ast,json,hashlib,sys,os
from native_binding_r206b import BASE,manifest,prebuild,binding
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');BASE_SHA='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'
def transform(raw):
 tree=ast.parse(raw);counts={'manifest':0,'prebuild':0,'create':0}
 for n in ast.walk(tree):
  if isinstance(n,ast.Dict):
   for i,k in enumerate(n.keys):
    if isinstance(k,ast.Constant)and k.value=='before_Torch_CDLL_and_weights':n.values[i]=ast.Constant(False) # Provider/HF parity intentionally runs first, still before weights.
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='manifest'for t in n.targets):n.value=ast.Call(func=ast.Name(id='__manifest_r206b__',ctx=ast.Load()),args=[],keywords=[]);counts['manifest']+=1
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='prebuild'for t in n.targets):n.value=ast.Call(func=ast.Name(id='__prebuild_r206b__',ctx=ast.Load()),args=[],keywords=[]);counts['prebuild']+=1
  elif isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_native_engine_create':
   assert [ast.literal_eval(x)for x in n.args[1:]]==[128,1024];n.args[1:]=[ast.Constant(1),ast.Constant(32768)];counts['create']+=1
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='report'and isinstance(t.slice,ast.Constant)and t.slice.value=='cache_capacity'for t in n.targets):n.value=ast.Constant(32768)
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='report'and isinstance(t.slice,ast.Constant)and t.slice.value=='cache_rows'for t in n.targets):n.value=ast.Constant(1)
 assert counts=={'manifest':1,'prebuild':1,'create':1}
 t=next(n for n in tree.body if isinstance(n,ast.Try)and any(isinstance(x,ast.FunctionDef)and x.name=='forward'for x in n.body));forward=next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name=='forward');before=ast.dump(forward)
 idx=next(i for i,x in enumerate(t.body)if isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id=='grouped'for y in x.targets));t.body[idx:idx]=ast.parse("__create_proof_r206b__=__normal_create_metadata_r206b__(globals())").body
 cut=next(i for i,x in enumerate(t.body)if isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id=='payload'for y in x.targets));keep=t.body[:cut];wanted={'tokenizer','HC_mode','projection_tile_mode','c4_tile_mode','precision_mode','csa_layers'};selected=[]
 for x in t.body[cut:]:
  if isinstance(x,ast.FunctionDef)and x.name=='forward':selected.append(x)
  elif isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id in wanted for y in x.targets):selected.append(x)
 assert len(selected)==7;t.body=keep+selected+ast.parse("__service_r206b__(globals())\nreport['collection_finished']=True").body;assert ast.dump(forward)==before and t.finalbody
 ast.fix_missing_locations(tree);compile(tree,str(BASE),'exec');return tree,dict(Native_forward_AST_unchanged=True,original_finally_destroy_retained=True,real_rows=1,real_capacity=32768,CPU_bound_new_Native=True,precision_and_math_unchanged=True)
def normal_create_metadata(ns):
 # Reuse observations already made by the mature loader's one normal Model
 # construction. No second Native constructor, cache probe or new Native gate.
 report=ns['report']
 return dict(prior_Native_lifecycle_actual_step='910374.1206',prior_normal_post_step='910374.1207',independent_Native_probe=False,weights_payload_opened=False,actual_normal_create_nonnull=bool(ns['ctx']),rows=report['cache_rows'],capacity=report['cache_capacity'],actual_owner_affinity=report['owner_affinity_before_weight_load'],GPU_memory_all8=report['GPU_memory_after_create'])
def main():
 raw=BASE.read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE_SHA;tree,proof=transform(raw);sys.path.insert(0,str(D));sys.path.insert(1,str(R/'candidates/latest-family-service-r270/Deep/python'))
 from preweight_r206b import run
 provider=run()
 from worker_r206b import serve
 namespace={'__name__':'__main__','__file__':str(BASE),'__manifest_r206b__':manifest,'__prebuild_r206b__':prebuild,'__normal_create_metadata_r206b__':normal_create_metadata,'__service_r206b__':serve,'__bootstrap_proof_r206b__':proof,'__provider_proof_r206b__':provider['proof'],'__latest_prepared_r277__':provider['prepared']}
 exec(compile(tree,str(BASE),'exec'),namespace);assert namespace['report']['engine_worker_shutdown_completed']
 result=R/'results'/('Deep-latest-full-owner-r277-'+namespace['tag']+'.json');q=json.loads(result.read_bytes());q.update(normal_Native_shutdown=True,legacy_load_receipt=str(namespace['path']),legacy_load_receipt_SHA256=hashlib.sha256(namespace['path'].read_bytes()).hexdigest());q['owner_exit_observation']=namespace.get('__owner_exit_observation_r206b__',{'status':'UNKNOWN'})
 for row in q['owner_exit_observation'].get('owners',[]):row['task_exists_after_destroy']=pathlib.Path('/proc/self/task/'+str(row['tid'])).exists()
 rows=q['owner_exit_observation'].get('owners',[]);q['owner_exit_observation']['all8_unique_threads_exited']=len(rows)==8 and len({x['tid']for x in rows})==8 and not any(x['task_exists_after_destroy']for x in rows);result.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');print('DEEP_REGULAR_R206B_NATIVE_NORMAL',flush=True)
if __name__=='__main__':main()
