import pathlib,json,os,traceback,numpy as np
from Kimi_corrected_WFP4_real_r8970 import run
root=pathlib.Path('/refactor');label='Kimi-corrected-WFP4-real-r8970';handle=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=root/'results'/(label+'-'+handle+'.json')
try:
 run(root/'incoming/Kimi-corrected-WFP4-real-manifest-r8970.json',out)
 q=json.loads(out.read_bytes());s=q['samples'];pairs=[]
 for n in range(8):
  a=next(x for x in s if x['pair']==n and not x['candidate']);b=next(x for x in s if x['pair']==n and x['candidate']);pairs.append({'pair':n,'event_gain_fraction':1-b['compound_provider_event_ms']/a['compound_provider_event_ms'],'host_gain_fraction':1-b['host_complete_ms']/a['host_complete_ms']})
 summary={'pairs':pairs}
 for key in ['compound_provider_event_ms','host_complete_ms']:
  old=np.array([x[key]for x in s if not x['candidate']]);new=np.array([x[key]for x in s if x['candidate']]);summary[key]={'current_mean':float(old.mean()),'candidate_mean':float(new.mean()),'mean_gain_fraction':float(1-new.mean()/old.mean())}
 summary['stable_5percent_screen']=all(x['event_gain_fraction']>=.05 and x['host_gain_fraction']>=.05 for x in pairs);summary['decision']='REVIEW_NUMERICS_AND_COMPONENT_ONLY'if summary['stable_5percent_screen']else'STOP_NO_STABLE_5_PERCENT';q['timing_summary']=summary;out.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'completed':True,'normal_owner_close':q['normal_owner_close'],'timing_summary':summary}))
except BaseException as e:
 q=json.loads(out.read_bytes())if out.exists()else{};q.update(completed=False,error=repr(e),traceback=traceback.format_exc());out.write_bytes((json.dumps(q,indent=2)+'\n').encode());raise
