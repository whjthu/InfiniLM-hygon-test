"""Raw JSON budget contract shared by CPU, pre-SSE sampling and enqueue."""
DEFAULT_OUTPUT=4096
CONTEXT=32768
BUDGET_KEYS=('max_tokens','max_completion_tokens','max_new_tokens')
def raw_budget(data):
 supplied=[];nested=data.get('sampling_params',{})
 if nested is None:nested={}
 if not isinstance(nested,dict):raise ValueError('sampling_params must be an object')
 for label,obj in [('top',data),('sampling_params',nested)]:
  for key in BUDGET_KEYS:
   if key in obj:
    value=obj[key]
    if type(value)is not int or value<1:raise ValueError('Every raw output budget must be a positive JSON integer; no bool, float, string or null coercion')
    supplied.append((label+'.'+key,value))
 if supplied and len({x[1]for x in supplied})!=1:raise ValueError('Conflicting output budget fields')
 budget=supplied[0][1]if supplied else DEFAULT_OUTPUT
 if budget>CONTEXT:raise ValueError('Output budget exceeds actual context')
 return budget,dict(supplied=supplied,selected=budget,coercion=False,clamp=False,default_applied=not supplied)
def check_sampling(data,params):
 budget,proof=raw_budget(data)
 if type(params.max_tokens)is not int or params.max_tokens!=budget:raise ValueError('Sampling budget must equal the original integer budget')
 return budget,proof
def install_pre_sse_guard(service):
 original=service._build_sampling_params
 def strict(data):
  budget,_=raw_budget(data)
  # A validated integer alias/default is forwarded without int() conversion.
  checked=dict(data);checked['max_tokens']=budget
  params=original(checked);check_sampling(data,params);return params
 service._build_sampling_params=strict
 return {'pre_SSE_raw_integer_guard':True,'enqueue_raw_integer_guard_required':True,'default_output_budget':4096,'actual_context':32768,'no_clamp':True}
def CPU_contract():
 from types import SimpleNamespace
 for value in [4096,8192,1]:
  for key in BUDGET_KEYS:
   d={key:value};assert raw_budget(d)[0]==value;assert check_sampling(d,SimpleNamespace(max_tokens=value))[0]==value
 for value in [True,False,4096.0,4096.5,'4096',None,0,-1]:
  for d in [{'max_tokens':value},{'sampling_params':{'max_tokens':value}}]:
   try:raw_budget(d)
   except ValueError:pass
   else:raise AssertionError(('invalid raw budget admitted',type(value).__name__))
 assert raw_budget({})[0]==4096
 class Service:
  def __init__(self):self.calls=0
  def _build_sampling_params(self,data):self.calls+=1;return SimpleNamespace(max_tokens=data['max_tokens'])
 service=Service();install_pre_sse_guard(service)
 for value in [True,4096.0,'4096']:
  try:service._build_sampling_params({'max_tokens':value})
  except ValueError:pass
  else:raise AssertionError('Pre-SSE raw validation failed')
 assert service.calls==0
 assert service._build_sampling_params({'max_tokens':8192}).max_tokens==8192 and service.calls==1
 return {'passed':True,'scope':'Pure CPU raw-budget/early-sampling contract, not real HTTP or Model qualification','4096_and8192_forwarded_unchanged':True,'float_bool_string_rejected_before_sampling':True}
