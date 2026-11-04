"""One engine-fixed explicit experimental profile; no request mode control."""
NAME='deep-hc0-fullpre-c1'
def mode(config):
 value=config.get('experimental_profile')
 if value is None:return 0
 if type(value)is not str or value!=NAME:raise ValueError('Unsupported explicit engine experimental_profile')
 return 1
def reject_request_control(data):
 for obj in [data,data.get('sampling_params')or{},data.get('chat_template_kwargs')or{}]:
  if not isinstance(obj,dict):raise ValueError('Public request and parameter fields must be objects')
  for key in obj:
   if key in {'experimental_profile','engine_profile','profile','fullpre','fullpre_mode','HC_mode','r366','r366_fixed','rows','context','TP','PP'}or str(key).startswith(('r449_','r450_','r454_')):
    raise ValueError('Engine profile/cache/parallel policy is not a public request control: '+str(key))
