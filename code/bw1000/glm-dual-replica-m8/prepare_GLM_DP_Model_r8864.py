import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent;old=r/'GLM-exact-budget-Model-private-r8855';d=r/'GLM-DP-Model-private-r8864';d.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent=json.loads((old/'GLM-CROSSREQUEST-Model-source-manifest-r8855.json').read_bytes())
names=list(parent['source_SHA256'])+['GLM-transient-Model-observer-manifest-r8855.json','dispatch_GLM_resident_exact_Model_r8855.py']
for name in names:
 raw=(old/name).read_bytes()
 if 'r8855'in name:raw=raw.replace(b'r8855',b'r8864')
 (d/name.replace('r8855','r8864')).write_bytes(raw)
def edit(name,fn):
 p=d/name;s=p.read_text();p.write_bytes(fn(s).encode())
def probe(s):
 s=s.replace("'ROWS32 TP8PP2/B16thenB32M8", "'DP two independent ROWS32 TP8PP2/B16M8")
 start=s.index('    endpoint=root/');end=s.index('    waiting=time.monotonic()',start)
 s=s[:start]+"    replica=os.environ['GLM_DP_REPLICA_R8864'];assert replica in('A','B')\n    endpoint=pathlib.Path(os.environ['GLM_DP_ENDPOINT_R8864'])\n"+s[end:]
 s=s.replace("e=json.loads(endpoint.read_text());assert e['participants']==16 and e['emitter']==8", "e=json.loads(endpoint.read_text());assert e['participants']==16 and e['emitter']==8 and e['replica']==replica and e['common_collector']\n    report['DP_common_clock_endpoint']={k:e[k]for k in ['host','port','replica','common_run']}\n    report['DP_replica']=replica;report['DP_global_rows']=list(range(0 if replica=='A'else 16,16 if replica=='A'else 32))")
 s=s.replace("participant,timeout=190)","participant,timeout=3600)")
 needle="        start = time.perf_counter(); step = 0";assert s.count(needle)==1
 s=s.replace(needle,"        if participant==emitter:feedback.request('dp_service_start',case_name,phase=dp_phase)\n"+needle)
 needle="            if continuous_policy and step==1:";assert s.count(needle)==1
 s=s.replace(needle,needle+"\n                if participant==emitter:feedback.request('dp_prefill_complete',case_name,phase=dp_phase)")
 needle="                            for row in group:tail_ready[row].append(ready_stamp)"
 # The deeper-indented occurrence is decode; only instrument the later prefill loop.
 prefill="                        for row in group:tail_ready[row].append(ready_stamp)"
 assert s.count(prefill)==2
 idx=s.rindex(prefill);s=s[:idx]+s[idx:].replace(prefill,prefill+"\n                        feedback.request('dp_prefill_rows',case_name,phase=dp_phase,rows=group,ids=chosen_cpu,token_index=0)",1)
 needle="historical_i512=json.loads(frozen477.read_text());assert historical_i512['input_ids']==i512_inputs";assert needle in s
 s=s.replace(needle,needle+"\n    global_rows=list(range(0 if replica=='A'else 16,16 if replica=='A'else 32))\n    i512_inputs=[i512_inputs[i]for i in global_rows]\n    historical_i512={**historical_i512,**{k:[historical_i512[k][i]for i in global_rows]for k in ['input_ids','generated_ids','stopped_at_EOS']}}")
 s=s.replace("report['prefill_factor_scope']='Same-load B16/B32 schedulingonly ordinaryfull128, O8fixed/VON/TOFF/fixedactualadmittedmask32rowbanks/crossOFF/sharedON; optionalNO byactualbudget; no windowprofile.'", "report['prefill_factor_scope']='One load perreplica, B16M8 mapped disjoint32globalrequests: ownisolated thenjoint; commonCPUclock/ports/nonces/communicators; prefillchunk8/VON/PVOFF/TOFF/prefixON; no optionalarm/profile.'")
 s=s.replace("labels=['perf-I512-B16-N-r0','perf-I512-B32-N-r0']", "labels=[f'perf-I512-DP-{replica}-{p}-N-r0'for p in('iso'+replica,'joint')]")
 old="        for label in labels:\n            prompts=i512_inputs[:16] if '-B16-'in label else i512_inputs\n            generate(label,prompts,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)"
 new="        for dp_phase in('isoA','isoB','joint'):\n            ticket=feedback.request('dp_gate','DP-control',phase=dp_phase)\n            if ticket['run']:\n                label=f'perf-I512-DP-{replica}-{dp_phase}-N-r0'\n                generate(label,i512_inputs,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)\n            feedback.request('dp_done','DP-control',phase=dp_phase,healthy=True)"
 assert old in s;s=s.replace(old,new)
 return s
edit('glm_prefill_micro_probe_r8864.py',probe)
edit('modern_glm_prefill_micro_r8864.py',lambda s:s.replace('=2600-(','=3500-('))
# Qualified old CPU207 provider is used unchanged; DP is not bundled with M16.
edit('GLM_transient_preflight_r8864.py',lambda s:s.replace("assert (m['TP']", "assert (m['TP']"))
for name in ['glm_DP_common_clock_r8864.py','run_GLM_DP_node_r8864.py']: (d/name).write_bytes((r/name).read_bytes())
# Collector reuses this exact qualified Python hub, never overwrites its shared copy.
(d/'glm_continuous_feedback_r8855.py').write_bytes((old/'glm_continuous_feedback_r8855.py').read_bytes())
worker=(old/'glm_prefill_micro_Model_r8855_inside.sh').read_text().replace('r8855','r8864').replace('export GLM_STAGE_R5040="$SLURM_PROCID"\n','').replace('timeout --kill-after=20s 2600 python','python')
(d/'GLM_DP_worker_inside_r8864.sh').write_bytes(worker.encode())
(d/'glm_prefill_micro_Model_r8864_inside.sh').write_bytes(b'#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\ntimeout --kill-after=20s 3500 python -u /refactor/incoming/run_GLM_DP_node_r8864.py\n')
m=dict(parent);m.update(TP=8,PP=2,active=32,constructor_rows=32,DP_replicas=2,DP_total_service_cards=32,DP_nodes_by_slot=['a02r3n03','a02r3n07','a02r3n15','a02r3n16'],DP_global_rows={'A':list(range(16)),'B':list(range(16,32))},DP_phases=['isoA','isoB','joint'],scope='Two independent world16 TP8PP2 B16M8 replicas, fixedconstructor32/exactbanks/currentqualified561DSO, disjoint32distinct477 I512O128. One load perreplica; isolatedA/isolatedB/joint, each2full128arms. Shared CPU clock collector only; originalemitter metrics separately. 910371 n03,n07,n15,n16/c64/240000Meach/60minhard3500inner; rootallocation/ModelGO pending.',root_GO='PENDING_ROOT_DP_ALLOCATION_AND_FINAL_PACKAGE_GO',diagnostic_budget='60minhard3500inner proposed:~450sstartup+max~1000sload(single561, storagecontentionUNKNOWN)+3sequentialphasewalls~3*143s+QA/setup/cleanupmargin; eachreplica2full128arms/1load; no extraarm/retry/extension.',math='Unchanged qualified561DSO68acfe/currentM8/vector/prefix/exactbank, no newTU/native/SDK compile; M16 absent.',scope_hit='ReplicaA original477globalrows0..15, B16..31, each16distinct512; originalfull128gold/EOS/QA/32normal/audit/post. DP commonobserver packet arrival clocks, no sequentialTPS sum.',perf_labels={'A':['perf-I512-DP-A-isoA-N-r0','perf-I512-DP-A-joint-N-r0'],'B':['perf-I512-DP-B-isoB-N-r0','perf-I512-DP-B-joint-N-r0']},maximum_loads=2,maximum_loads_per_replica=1,maximum_full128_arms=4,maximum_full128_arms_per_replica=2,fixed_ordinary_arm_count=4)
m['input_provenance']=dict(parent['input_provenance']);m['input_provenance']['B16']='A first16/B last16; disjointtotal32; explicitglobalrowmaps; no newgold';m['mechanical_revision']={'provider':'qualified561 unchanged','control':'One existingfeedback hub perreplica hostedinoneCPUprocess; pureCPU phasegate/done, prefillrows/complete/service-start observations; no GPU synchronization added.'}
m['source_SHA256']={n.replace('r8855','r8864'):sha(d/n.replace('r8855','r8864'))for n in parent['source_SHA256']}
for n in ['glm_DP_common_clock_r8864.py','run_GLM_DP_node_r8864.py','GLM_DP_worker_inside_r8864.sh','glm_continuous_feedback_r8855.py']:m['source_SHA256'][n]=sha(d/n)
mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8864.json';mp.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode())
for p in d.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
print(json.dumps({'private_directory':d.name,'CPU_only':True,'GPU':False,'phase':'probe/node/commonclock ready; launcher/audit/watch still pending','source_count':len(m['source_SHA256'])}))
