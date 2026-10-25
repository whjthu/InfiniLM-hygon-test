import pathlib,json,hashlib,ast
r=pathlib.Path.cwd();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(n,s):
 if n.endswith('.py'):ast.parse(s)
 (r/n).write_bytes(s.encode('utf8'))
files=['modern_Deep_latent_Model_r8942.py','Deep_latent_Model_protocol_r8942.py','Deep_latent_Model_r8942_inside.sh','verify_Deep_latent_Model_r8942.py','launch_Deep_latent_Model_r8942.py','check_Deep_latent_Model_actual_r8942.py','check_Deep_latent_Model_consumer_r8942.py','check_Deep_latent_Model_error_r8942.py','stage_Deep_latent_Model_runner_r8942.py']
for n in files:
 s=(r/n).read_text().replace('r8942','r8943')
 if n.startswith('modern_'):
  s='import time\npacket_started=time.perf_counter()\n'+s
  s=s.replace('A/C/C/A full128:', 'A/C/C full128 +optionalbudgetA3:')
 if n.startswith('Deep_latent_Model_protocol'):
  s=s.replace('A/C/C/A full128','A/C/C full128 +optionalA3onlyifremaining>=400s').replace('one load accepted A/C/C/A','one load accepted A/C/C plusbudgetoptionalA')
  s=s.replace("[(0,'A0'),(1,'C1'),(1,'C2'),(0,'A3')]", "[(0,'A0'),(1,'C1'),(1,'C2')]")
  point=" policy(0);select(0);g['run_chat']('after_restore')"
  assert s.count(point)==1
  s=s.replace(point,""" remaining=1740-(time.perf_counter()-g['packet_started']);report['optional_final_A_budget']={'remaining_inner_seconds':remaining,'threshold_seconds':400,'status':'RUNNING'if remaining>=400 else'SKIPPED_BUDGET'};save()
 if remaining>=400:
  entry=one(0,'A3')
  if not entry['complete']:save();raise RuntimeError('A3_full128_quality_failed_STOP')
  report['optional_final_A_budget']['status']='COMPLETED'
"""+point)
  s=s.replace("outcome='FULL_ACCA_COMPLETE_PENDING_TERMINAL'", "outcome='FULL_ACC_OPTIONAL_A_COMPLETE_PENDING_TERMINAL'").replace('fourfull128 frozen814/freshA','threeorfourfull128 frozen814/freshA').replace('Ordinary unmarkedModel A/C/C/A;', 'Ordinary unmarkedModel A/C/C plusoptionalA3; noendpointdriftclaimifskipped;')
 if n.startswith('verify_'):
  s=s.replace("len(trials)==4", "len(trials)in[3,4]")
  s=s.replace("[c['latent_arm']for c in trials]==[0,1,1,0]and[c['label']for c in trials]==['A0','C1','C2','A3']", "[c['latent_arm']for c in trials]==[0,1,1,0][:len(trials)]and[c['label']for c in trials]==['A0','C1','C2','A3'][:len(trials)]")
  s=s.replace("assert q['owner_event_windows']==[]", "assert q['optional_final_A_budget']['threshold_seconds']==400 and q['optional_final_A_budget']['status']==('COMPLETED'if len(trials)==4 else'SKIPPED_BUDGET')\nassert q['owner_event_windows']==[]")
 if n.startswith('check_Deep_latent_Model_actual'):
  s=s.replace("[(0,'A0'),(1,'C1'),(1,'C2'),(0,'A3')]", "[(0,'A0'),(1,'C1'),(1,'C2')]")
  s=s.replace("assert 'shutdown_entered'in es", "assert 'remaining>=400'in ps and 'SKIPPED_BUDGET'in ps and 'packet_started=time.perf_counter()'in es\nassert 'shutdown_entered'in es")
 if n.startswith('check_Deep_latent_Model_consumer'):
  s=s.replace('def execute(negative=None):', 'def execute(negative=None,budget_skip=False):')
  s=s.replace('g=dict(checkpoint=checkpoint', "g=dict(packet_started=matrix.time.perf_counter()-(1400 if budget_skip else 0),checkpoint=checkpoint")
  s=s.replace('lib.calls==540', 'lib.calls==(405 if budget_skip else 540)').replace("len(report['owner_trials_r8943'])==4", "len(report['owner_trials_r8943'])==(3 if budget_skip else 4)")
  s=s.replace("==32\n", "==(24 if budget_skip else 32)\n")
  s=s.replace("==[0,1,1,0]\n", "==[0,1,1,0][:3 if budget_skip else 4]\n")
  s=s.replace('lib.pre==32 and lib.dec==508', 'lib.pre==(24 if budget_skip else 32) and lib.dec==(381 if budget_skip else 508)')
  s=s.replace("CAPI_calls=540,CAPI_calls_per_arm", "CAPI_calls=lib.calls,CAPI_calls_per_arm").replace("plan=['A0','C1','C2','A3']", "plan=['A0','C1','C2','A3'][:3 if budget_skip else 4]")
  s=s.replace('positive=execute(),', 'positive=execute(),positive_budget_skip=execute(budget_skip=True),')
 write(n.replace('r8942','r8943'),s)
m=json.loads((r/'Deep-latent-Model-manifest-r8942.json').read_text());m['scope']=m['scope'].replace('A/C/C/A full128','A/C/C mandatoryfull128;optionalA3 onlyremaininginner>=400s');m['geometry']['plan']=['A0','C1','C2'];m['geometry']['optional_final_A']={'label':'A3','minimum_inner_remaining_seconds':400,'else':'SKIPPED_BUDGET','full_output_tokens':128};m['incoming_source_SHA256']={n.replace('r8942','r8943'):sha(r/n.replace('r8942','r8943'))for n in m['incoming_source_SHA256']};m['budget']['mandatory_three_zero_gain_seconds']=545.4753086329438+3*241.84949570009485+150.70509766696136;m['budget']['mandatory_three_zero_gain_margin_seconds']=1740-m['budget']['mandatory_three_zero_gain_seconds'];m['budget']['scope']='FixedA/C/C full128mandatory. OptionalfinalA3 onlyif1740-entryelapsed>=400s, elseSKIPPED_BUDGET. Nooutputshortening/extraQA/extension/retry;remainingclockincludesimports/load. PlanningfactorNOTModelgain.';m['quality']=m['quality'].replace('Eachof4full128','Eachof3or4full128');m['CPU_consumer_contract']={'path':'Deep-latent-Model-consumer-CPU-r8943.json','SHA256':'PENDING'};m['CPU_exception_contract']={'path':'Deep-latent-Model-error-CPU-r8943.json','SHA256':'PENDING'};(r/'Deep-latent-Model-manifest-r8943.json').write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
print('r8943 final budget packet prepared, noGPU')
