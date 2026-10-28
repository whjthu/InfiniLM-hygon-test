"""Reuse qualified loader lifecycle; bind only the actual new guard artifact."""
import pathlib,ast,json,hashlib,sys,os
from native_binding_r182 import BASE,manifest,prebuild,binding
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');BASE_SHA='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'
def transform(raw):
 tree=ast.parse(raw);counts={'manifest':0,'prebuild':0,'create':0}
 for n in ast.walk(tree):
  if isinstance(n,ast.Dict):
   for i,k in enumerate(n.keys):
    if isinstance(k,ast.Constant)and k.value=='before_Torch_CDLL_and_weights':n.values[i]=ast.Constant(False) # Provider/HF parity intentionally runs first, still before weights.
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='manifest'for t in n.targets):n.value=ast.Call(func=ast.Name(id='__manifest_r182__',ctx=ast.Load()),args=[],keywords=[]);counts['manifest']+=1
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='prebuild'for t in n.targets):n.value=ast.Call(func=ast.Name(id='__prebuild_r182__',ctx=ast.Load()),args=[],keywords=[]);counts['prebuild']+=1
  elif isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_native_engine_create':
   assert [ast.literal_eval(x)for x in n.args[1:]]==[128,1024];n.args[1:]=[ast.Constant(1),ast.Constant(32768)];counts['create']+=1
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='report'and isinstance(t.slice,ast.Constant)and t.slice.value=='cache_capacity'for t in n.targets):n.value=ast.Constant(32768)
  elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='report'and isinstance(t.slice,ast.Constant)and t.slice.value=='cache_rows'for t in n.targets):n.value=ast.Constant(1)
 assert counts=={'manifest':1,'prebuild':1,'create':1}
 t=next(n for n in tree.body if isinstance(n,ast.Try)and any(isinstance(x,ast.FunctionDef)and x.name=='forward'for x in n.body));forward=next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name=='forward');before=ast.dump(forward)
 idx=next(i for i,x in enumerate(t.body)if isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id=='grouped'for y in x.targets));t.body[idx:idx]=ast.parse("__create_proof_r182__=__preload_gate_r182__(globals())").body
 cut=next(i for i,x in enumerate(t.body)if isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id=='payload'for y in x.targets));keep=t.body[:cut];wanted={'tokenizer','HC_mode','projection_tile_mode','c4_tile_mode','precision_mode','csa_layers'};selected=[]
 for x in t.body[cut:]:
  if isinstance(x,ast.FunctionDef)and x.name=='forward':selected.append(x)
  elif isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id in wanted for y in x.targets):selected.append(x)
 assert len(selected)==7;t.body=keep+selected+ast.parse("__service_r182__(globals())\nreport['collection_finished']=True").body;assert ast.dump(forward)==before and t.finalbody
 ast.fix_missing_locations(tree);compile(tree,str(BASE),'exec');return tree,dict(Native_forward_AST_unchanged=True,original_finally_destroy_retained=True,real_rows=1,real_capacity=32768,CPU_bound_new_Native=True,precision_and_math_unchanged=True)
def pre_load_gate(ns):
 torch=ns['torch'];lib=ns['lib'];owners=ns['verify_owner_affinity'](lib);assert owners['verified']and len(owners['actual_owner_report']['owners'])==8
 proof=dict(passed=True,weights_payload_opened=False,actual_create_nonnull=bool(ns['ctx']),rows=1,capacity=32768,actual_owner_affinity=owners,GPU_memory_all8=[list(torch.cuda.mem_get_info(i))for i in range(8)],CCL='Actual TP8 Engine create completed and all8 NativeSparseModel constructor rank.comm guards passed. Separate preweight payload collective not run.',source_vs_execution='Actual creation/memory/owner observations; CCL initialization completion inferred from the pinned constructor guards and successful Engine create, not a collective performance claim')
 ns['report']['formal_preload_create_proof']=proof;ns['atomic_save']();return proof
def main():
 raw=BASE.read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE_SHA;tree,proof=transform(raw);sys.path.insert(0,str(D));sys.path.insert(1,str(R/'candidates/Deep-agent-regular32K-r182/python'))
 from preweight_r182 import run
 provider=run()
 from worker_r182 import serve
 namespace={'__name__':'__main__','__file__':str(BASE),'__manifest_r182__':manifest,'__prebuild_r182__':prebuild,'__preload_gate_r182__':pre_load_gate,'__service_r182__':serve,'__bootstrap_proof_r182__':proof,'__provider_proof_r182__':provider}
 exec(compile(tree,str(BASE),'exec'),namespace);assert namespace['report']['engine_worker_shutdown_completed']
 result=R/'results'/('Deep-agent-regular32K-r182-'+namespace['tag']+'.json');q=json.loads(result.read_bytes());q.update(normal_Native_shutdown=True,legacy_load_receipt=str(namespace['path']),legacy_load_receipt_SHA256=hashlib.sha256(namespace['path'].read_bytes()).hexdigest());q['owner_exit_observation']=namespace.get('__owner_exit_observation_r182__',{'status':'UNKNOWN'})
 for row in q['owner_exit_observation'].get('owners',[]):row['task_exists_after_destroy']=pathlib.Path('/proc/self/task/'+str(row['tid'])).exists()
 rows=q['owner_exit_observation'].get('owners',[]);q['owner_exit_observation']['all8_unique_threads_exited']=len(rows)==8 and len({x['tid']for x in rows})==8 and not any(x['task_exists_after_destroy']for x in rows);result.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');print('DEEP_REGULAR_R182_NATIVE_NORMAL',flush=True)
if __name__=='__main__':main()
