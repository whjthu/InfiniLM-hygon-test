"""One private constructor call replacement; original close/loader/guard unchanged."""
from pathlib import Path
import ast,hashlib,importlib.util,inspect,textwrap
from policy import OFF,ON,CONTEXT_SHA256,checked

def private_runtime_type(old,context_factory):
 raw=textwrap.dedent(inspect.getsource(old.Runtime.__init__));tree=ast.parse(raw)
 calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='LongCommittedFeatureContext']
 assert len(calls)==1 and ast.dump(calls[0].keywords[0])==ast.dump(ast.keyword(arg='enabled',value=ast.Constant(True)))
 original=ast.dump(tree,include_attributes=False);calls[0].func.id='context_factory_r388'
 proof=dict(original_constructor_AST_SHA256=hashlib.sha256(original.encode()).hexdigest(),private_constructor_AST_SHA256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),replacements=1,only_changed_call='LongCommittedFeatureContext -> context_factory_r388')
 namespace=dict(vars(old),context_factory_r388=context_factory)
 exec(compile(ast.fix_missing_locations(tree),'<private-r388-original-constructor>','exec'),namespace)
 cls=type('PolicyRuntime_r388',(old.Runtime,),{'__init__':namespace['__init__']})
 assert cls.close is old.Runtime.close
 return cls,proof

class RuntimeFactory:
 def __init__(self,old,root,runtime,incoming,report,mode=OFF):
  self.old,self.root,self.runtime,self.incoming,self.report=old,Path(root),runtime,incoming,report
  self.mode=checked(mode);self.scope=None;self.called=False
 def load_context(self):
  path=self.root/'incoming/Kimi-raw-history-elision-r384/context.py'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==CONTEXT_SHA256
  spec=importlib.util.spec_from_file_location('projected_only_r384_for388',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module.ProjectedOnlyContext
 def __call__(self):
  assert not self.called,'No runtime reload/retry within a connection';self.called=True
  # Called only after exact authenticated hello. Source import is also before
  # original Runtime touches the head/checkpoint, without patching shared code.
  if self.mode==ON:
   context_cls=self.load_context()
   def context_factory(model,weight_tag,*,enabled):return context_cls(model,weight_tag,enabled=enabled,discard_raw=True)
   cls,proof=private_runtime_type(self.old,context_factory);self.report['private_constructor']=proof
  else:cls=self.old.Runtime
  import torch
  self.scope=torch.inference_mode();self.scope.__enter__()
  value=None
  try:
   value=cls(self.root,self.runtime,self.incoming,self.report)
   c=value.session.context
   if self.mode==ON:assert isinstance(c,context_cls)and c.discard_raw is True
   self.report.update(context_policy_actual=self.mode,context_source_SHA256=CONTEXT_SHA256 if self.mode==ON else None,context_class=type(c).__module__+'.'+type(c).__qualname__,original_runtime_close_unchanged=(cls.close is self.old.Runtime.close))
   return value
  except BaseException:
   try:
    if value is not None:value.close()
   finally:self.close_scope()
   raise
 def close_scope(self):
  if self.scope is not None:
   scope=self.scope;self.scope=None;scope.__exit__(None,None,None)
