"""CPU-only pre-engine requirement. It never asserts actual worker affinity."""
import os,json,sys
KEY='INFINILM_OWNER_CPU_WIDTH_R2392'
def validate(env):
 if KEY not in env:raise ValueError('Missing explicit '+KEY+' before any Engine or model load')
 raw=env[KEY]
 if raw not in ('0','1','4','8'):raise ValueError('Unsupported explicit owner CPU width: '+repr(raw))
 return {'CPU_preflight_passed':True,'env_key':KEY,'requested_width':int(raw),'actual_affinity_verified':False,'actual_gate':'Read actual worker owner report under configured width after Engine; missing terminal report remains unverified, never fabricated'}
if __name__=='__main__':
 if '--selftest' in sys.argv:
  rejected=0
  for e in [{},{KEY+' ':'4'},{KEY:'2'},{KEY:'4.0'},{KEY:''}]:
   try:validate(e)
   except ValueError:rejected+=1
  assert rejected==5
  assert [validate({KEY:s})['requested_width'] for s in ['0','1','4','8']]==[0,1,4,8]
  print(json.dumps({'passed':True,'negative_cases':5,'positive_cases':4,'scope':'CPU config preflight, no Engine/GPU or historicalaffinity reconstruction'}))
 else:print(json.dumps(validate(os.environ)))
