"""Private364 one-call binding; same marker/deadline/collect/capture/Native."""
import ast,copy,hashlib,inspect,textwrap
from context_policy_client_r388 import OFF,ON,WORKER_MF,bundle,client_type,sha

def settings_for_policy(settings,bound,mode=OFF):
 if mode==OFF:return settings
 if mode!=ON:raise ValueError('Explicit supported context policy required')
 if settings.identity['worker_manifest_SHA256']!=bound.policy.OLD_MANIFEST_SHA256 or settings.identity['target_Native_SHA256']!=bound.old.TARGET_NATIVE_SHA:raise ValueError('Exact old364 baseline identity required')
 value=copy.copy(settings);value.identity=dict(settings.identity,worker_manifest_SHA256=WORKER_MF)
 return value

def connect_policy(original_connect,settings,gate,control,original_client,save,*,bound,mode=OFF,**kwargs):
 if mode==OFF:return original_connect(settings,gate,control,original_client,save,**kwargs)
 chosen=settings_for_policy(settings,bound,mode);client=client_type(original_client,bound,mode)
 def save_policy(value):
  if 'draft_wire_source_binding'in value:
   value=dict(value,draft_wire_source_binding=dict(worker_manifest_SHA256=WORKER_MF,runtime_manifest_SHA256=bound.old.RUNTIME_SHA,target_Native_SHA256=bound.old.TARGET_NATIVE_SHA,protocol=bound.policy.PROTOCOL_ON,context_policy=ON,context_source_SHA256=bound.policy.CONTEXT_SHA256,worker_identity_scope='New388MF instart/listening marker; wire exactnonce+policy+384source+388MF+333runtime+327Native. ActualSlurm/CLI binding remainsfuturecontroller responsibility.'))
  save(value)
 return original_connect(chosen,gate,control,client,save_policy,**kwargs)

def bind_dependencies(base_bootstrap,root,mode=OFF):
 """Pass this returned class to original bootstrap.execute; no global patch."""
 if mode==OFF:return base_bootstrap.RealDependencies
 if mode!=ON:raise ValueError('Explicit supported context policy required')
 assert sha(base_bootstrap.RealDependencies.run_pair.__code__.co_filename)=='8d43dfa621a7bd23623449613ab8e2421ca28dcbb6347ae0b1608c8d270ce53a'
 assert sha(base_bootstrap.connect_draft.__code__.co_filename)=='8e762985e8f82932a147ab21ad526a4fe3f291d10844e849791186dfbfc1c617'
 bound=bundle(root)
 def connect_bound(settings,gate,control,original_client,save,**kwargs):
  return connect_policy(base_bootstrap.connect_draft,settings,gate,control,original_client,save,bound=bound,mode=ON,**kwargs)
 tree=ast.parse(textwrap.dedent(inspect.getsource(base_bootstrap.RealDependencies.run_pair)))
 calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='connect_draft'];assert len(calls)==1
 original=ast.dump(tree,include_attributes=False);calls[0].func.id='connect_draft_policy_r388'
 proof=dict(replacements=1,original_run_pair_AST_SHA256=hashlib.sha256(original.encode()).hexdigest(),private_run_pair_AST_SHA256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),only_change='connect_draft -> private388 connection binding')
 namespace=dict(vars(base_bootstrap),connect_draft_policy_r388=connect_bound)
 exec(compile(ast.fix_missing_locations(tree),'<private388-original364-run_pair>','exec'),namespace)
 cls=type('ContextPolicyTargetDependencies388',(base_bootstrap.RealDependencies,),{'run_pair':namespace['run_pair'],'context_policy_binding':proof})
 assert cls.__init__ is base_bootstrap.RealDependencies.__init__
 return cls

def execute_with_policy(base_bootstrap,directory,settings,gate,report,save,*,root,mode=OFF):
 """Production-callable entry; no allocator, launcher, retry or new request."""
 cls=bind_dependencies(base_bootstrap,root,mode)
 if mode==ON:report['draft_context_policy_adapter']=dict(context_policy=ON,worker_manifest_SHA256=WORKER_MF,private_target_binding=cls.context_policy_binding,device_and_Model_qualification=False)
 return base_bootstrap.execute(settings,gate,cls(directory),report,save)
