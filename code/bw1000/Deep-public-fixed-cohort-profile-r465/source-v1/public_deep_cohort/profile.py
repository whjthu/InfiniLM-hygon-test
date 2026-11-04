"""Explicit OFF-by-default closed cohorts, not an online batch scheduler."""
import ctypes as C,re
NAME=PROFILE='deep-hc0-fixed-cohort-c4'
def validate(config):
 if config.get('experimental_profile')!=NAME:raise ValueError('Fixed cohort profile requires explicit startup opt-in')
 assert(config['rows'],config['context'],config['HC_mode'],config['r366'],config['fullpre'])==(4,32768,0,0,0)
 assert config['profile']==NAME and config['capacity_per_row']==32768
 assert type(config['cohort_assembly_seconds'])in(int,float)and 0<config['cohort_assembly_seconds']<=30
 return NAME
def ticket(data):
 v=data.get('native_fixed_cohort')
 if type(v)is not dict or set(v)!={'id','mode','row','size'}:raise ValueError('Explicit four-request cohort ticket required')
 if type(v['id'])is not str or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',v['id']):raise ValueError('Invalid cohort identity')
 if v['mode']not in('serial','cohort')or type(v['row'])is not int or v['row']not in range(4)or type(v['size'])is not int or v['size']!=4:raise ValueError('Only a closed four-request cohort is supported')
 return dict(v)
def reject_request_control(data):
 for obj in[data,data.get('sampling_params')or{},data.get('chat_template_kwargs')or{}]:
  if not isinstance(obj,dict):raise ValueError('Request parameters must be objects')
  if any(k in{'experimental_profile','engine_profile','profile','fullpre','fullpre_mode','HC_mode','r366','r366_fixed','rows','context','TP','PP'}for k in obj):raise ValueError('Native math/cache settings are startup-only')
 if data.get('chat_template_kwargs',{}).get('enable_thinking')is not False:raise ValueError('Explicit thinkingfalse required')
 if type(data.get('max_tokens',4096))is not int or data.get('max_tokens',4096)!=4096:raise ValueError('Original4096 budget required')
 ticket(data)
def check_modes(lib):
 for name in['deep_hc_canonical_get_r7130','deep_hc_model_epilogue_get_r366','deep_hc_fullpre_enabled_r377','modern_deep_c4_tile_get_r5920','modern_deep_layout_get_r7052','deep_moe_tail_get_r7211']:
  f=getattr(lib,name);f.argtypes=[];f.restype=C.c_int;assert f()==0,('Frozen mode must remainOFF',name)
 return dict(HC=0,r366=0,fullpre=0,c4_tile=0,layout=0,tail_SUM=0)
