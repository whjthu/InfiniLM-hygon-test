"""Small CAPI adapter; drains bounded native journal before finalID gates."""
import ctypes,json,torch,threading
class Layer77Policy:
 def __init__(self,lib,stage,rank,control,check,report,save):
  self.owner_thread=threading.get_ident()
  self.lib,self.stage,self.rank,self.control,self.check,self.report,self.save=lib,stage,rank,control,check,report,save
  lib.modern_glm_cross_control_r5775.argtypes=[ctypes.c_int]*4
  lib.modern_glm_cross_journal_r5775.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_cross_journal_r5775.restype=ctypes.c_char_p
 def collect(self,label,step):
  assert threading.get_ident()==self.owner_thread
  raw=self.lib.modern_glm_cross_journal_r5775(self.rank,1);assert raw,'native journal failed';q=json.loads(raw)
  expected=list(range((0,26,54)[self.stage],(26,54,78)[self.stage]));assert sorted(i['layer']for i in q['identities'])==expected
  assert all(i['stage']==self.stage and i['rank']==self.rank for i in q['identities'])
  for event in q['events']:
   for key in ['joined_copy_equal','shadow_copy_equal']:
    if key in event:assert event[key],('semanticcopyfailure-stopcandidate',event)
   if 'shadow_old_new'in event:assert event['shadow_old_new']['finite']
  if q['events']:self.report.setdefault('layer77_native_journals',[]).append({'phase':label,'round':step,**q});self.save()
  return q
 def switch(self,label,mode,diagnostic,shadow=-1):
  self.collect(label+'-previous-drained',-1)
  rc=self.lib.modern_glm_cross_control_r5775(self.rank,mode,int(diagnostic),shadow)
  vote=torch.tensor([float(rc==0),float(mode),float(diagnostic),float(shadow+1)],device='cuda');self.control(vote)
  assert vote.tolist()==[24.,float(mode*24),float(int(diagnostic)*24),float((shadow+1)*24)],('all16 immutable exactphase policy',rc,vote.tolist())
  self.report.setdefault('layer77_phase_policies',{})[label]={'mode':mode,'diagnostic':bool(diagnostic),'shadow':shadow,'candidate_global_allowlist':[77]if mode==2 else list(range(78))if mode==1 else[],'resident_mask_separate':True};self.save()
