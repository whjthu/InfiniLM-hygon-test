def make_commands(data,admitted):
 main=data['GSM64'];mmlu=data['MMLU32'];commands=[]
 def add(label,requests,enabled,category,C=None,W=None,cap=512,**extra):
  chosen=[x for x in requests if x['id']in admitted];c=len(chosen)
  if not c:return
  commands.append({'schema':1,'action':'case','id':label,'C':c,'W':W or(8 if c==1 else 4),'strategy':'state','I192':True,'adaptive_width':True,'epsilon':0,'W2pair':enabled,'draft_projected_cache':False,'cap':cap,'case_budget_s':180 if c==1 else 480,'category':category,'planned_request_ids':[x['id']for x in requests],'requests':chosen,**extra})
 for enabled in [False,True]:add('warm-'+str(int(enabled)),main[:16],enabled,'warm',cap=8)
 for i,q in enumerate(main[:8]):
  for enabled in ([False,True]if i%2==0 else[True,False]):add('screen-%02d-%s'%(i,int(enabled)),[q],enabled,'screen')
 for group in range(4):
  for enabled in ([False,True]if group%2==0 else[True,False]):add('GSM64-C16-g%d-%d'%(group,int(enabled)),main[group*16:(group+1)*16],enabled,'main-GSM64')
 for rep in range(2):
  for enabled in ([False,True]if rep%2==0 else[True,False]):
   groupid='service32-r%d-%d'%(rep,int(enabled))
   for part in range(2):add(groupid+'-part'+str(part),main[part*16:(part+1)*16],enabled,'frontend-C32',service_id=groupid,service_part=part)
 for group in range(2):
  for enabled in ([False,True]if group%2==0 else[True,False]):add('MMLU32-C16-g%d-%d'%(group,int(enabled)),mmlu[group*16:(group+1)*16],enabled,'MMLU32')
 return commands
