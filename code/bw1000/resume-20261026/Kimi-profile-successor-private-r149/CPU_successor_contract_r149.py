"""No process/Slurm/HTTP/GPU: exact cleanup and claim admission contract."""
import copy
import json
from successor_after_P0_r149 import predecessor_cleanup,validate_claim,GPU,KEEPER,RUN,NODES,P0_MANIFEST,PROFILE_MANIFEST
rows={GPU:[GPU,'COMPLETED','0:0'],KEEPER:[KEEPER,'COMPLETED','0:0']}
watch={'actual_handle':GPU,'CPU_actual_step':KEEPER,'post_empty':True,'terminal':[GPU,'name','COMPLETED','0:0'],'post_KFD':{'terminal':['910380.999','post','COMPLETED','0:0'],'rows':[{'node':n,'passed':True,'kfd_pids':[]}for n in NODES]}}
reports=[{'node':i,'run':RUN,'manifest_SHA256':P0_MANIFEST,'all8_owned_workers_shutdown_completed':True,'agent_HTTP_normal_shutdown':True}for i in range(4)]
identity={'actual_step':GPU,'cards':32,'manifest_SHA256':P0_MANIFEST}
assert predecessor_cleanup(watch,reports,identity,rows)
active=copy.deepcopy(rows);active[GPU][1]='RUNNING';assert predecessor_cleanup(watch,reports,identity,active) is False
dirty=copy.deepcopy(watch);dirty['post_empty']=False;assert predecessor_cleanup(dirty,reports,identity,rows) is False
count=3
for kind in ['wrong_handle','dirty_kfd','shutdown_missing','wrong_identity']:
 w=copy.deepcopy(watch);r=copy.deepcopy(reports);i=copy.deepcopy(identity)
 if kind=='wrong_handle':w['actual_handle']='910380.432'
 if kind=='dirty_kfd':w['post_KFD']['rows'][0]['kfd_pids']=['1']
 if kind=='shutdown_missing':r[0]['all8_owned_workers_shutdown_completed']=False
 if kind=='wrong_identity':i['actual_step']='910380.432'
 try:predecessor_cleanup(w,r,i,rows)
 except AssertionError:pass
 else:raise AssertionError('invalid cleanup accepted '+kind)
 count+=1
go={'publication_branch':'bw1000-experiments','publication_commit':'a'*40,'claim_export_is_current_branch_head':True,'GPU_successor_authorized':True,'source_published':True}
claim={'work_item':'kimi-current-c8-verify-profile','state':'P0_RUNNING_PROFILE_SUCCESSOR_ARMED','new_source_manifest_SHA256':PROFILE_MANIFEST,'resources':{'target_parent':'910380','target_nodes':NODES,'target_cards':32,'draft_parent':'910371','draft_node':'a02r3n03','draft_cards':1},'handoff_gate':{'predecessor_GPU':GPU,'predecessor_keeper':KEEPER,'successor':'Kimi-sparse-profile-r136b','order':'P0_TERMINAL_CLEANUP_THEN_PROFILE'}}
validate_claim(go,claim);count+=1
bad=copy.deepcopy(claim);bad['state']='DEFERRED_FOR_P0'
try:validate_claim(go,bad)
except AssertionError:pass
else:raise AssertionError('stale deferred claim accepted')
count+=1
print(json.dumps({'passed':True,'CPU_only':True,'Slurm':False,'GPU':False,'Native':False,'cases':count}))
