"""Retain the qualified ac937 Native loader/forward/finally; replace only QA run.

The original source is SHA-pinned, read-only, and parsed without import on CPU.
No baseline source or shared Native library is edited.
"""
import ast, hashlib, pathlib, os, sys, json
D=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/refactor/incoming/Deep-C4-independent64-r9278/modern_Deep_independent64_r9278.py')
BASE_SHA='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'

def transform(raw):
 tree=ast.parse(raw);targets=[n for n in tree.body if isinstance(n,ast.Try)and any(isinstance(x,ast.FunctionDef)and x.name=='forward'for x in n.body)]
 assert len(targets)==1;t=targets[0];original=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='forward')
 cut=next(i for i,n in enumerate(t.body)if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='payload'for x in n.targets))
 keep=t.body[:cut];wanted={'tokenizer','HC_mode','projection_tile_mode','c4_tile_mode','precision_mode','csa_layers'}
 selected=[]
 for n in t.body[cut:]:
  if isinstance(n,ast.FunctionDef)and n.name=='forward':selected.append(n)
  elif isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id in wanted for x in n.targets):selected.append(n)
 assert len(selected)==7
 t.body=keep+selected+ast.parse("checkpoint('service_enter')\n__deep_service_callback__(globals())\nreport['collection_finished']=True\ncheckpoint('service_return')").body
 assert ast.dump(original)==ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='forward'))
 ast.fix_missing_locations(tree);compile(tree,str(BASE),'exec')
 return tree,{'qualified_source_SHA256':BASE_SHA,'Native_forward_AST_unchanged':True,'create_rows':128,'create_capacity':1024,'Native_math_changed':False,'original_finally_destroy_retained':len(t.finalbody)>0,'old_QA_run_removed':True}

def main():
 raw=BASE.read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE_SHA
 tree,proof=transform(raw)
 sys.path.insert(0,str(D));sys.path.insert(1,'/refactor/candidates/Deep-agent-p0-r143c/python')
 from preweight_import_r143 import run as preweight_import
 preweight_import()
 from deep_service_worker_r143 import serve
 namespace={'__name__':'__main__','__file__':str(BASE),'__deep_service_callback__':serve,'__bootstrap_proof__':proof}
 exec(compile(tree,str(BASE),'exec'),namespace)
 assert namespace['report']['engine_worker_shutdown_completed']
 p=namespace['r']/'results'/('deep-agent-p0-r143c-'+namespace['tag']+'.json');q=json.loads(p.read_bytes());q['normal_Native_shutdown']=True;q['legacy_load_receipt']=str(namespace['path']);q['legacy_load_receipt_SHA256']=hashlib.sha256(namespace['path'].read_bytes()).hexdigest();tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,ensure_ascii=False,indent=2));tmp.replace(p)
 print('DEEP_AGENT_P0_R143C_NATIVE_NORMAL',flush=True)

if __name__=='__main__':main()
