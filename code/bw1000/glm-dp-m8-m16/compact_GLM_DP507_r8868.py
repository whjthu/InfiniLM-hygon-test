"""Only original DP507 saved terminal evidence, no GPU/process step."""
import pathlib,json,hashlib,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();wp=r/'results/GLM-transient-Model-watch-r8868.json';w=json.loads(wp.read_bytes());assert w['state'].startswith('terminal_')and w.get('postKFD_passed');run=w['run'];ap=r/'results/GLM-transient-Model-terminal-audit-r8868.json';audit=json.loads(ap.read_bytes())if ap.exists()else None
gold=json.loads((r/'incoming/GLM477-frozen-gold-r8844.json').read_bytes());owners=[];tails={};arms={};allgold=True
def summary(v):return{'min':min(v),'mean':statistics.mean(v),'max':max(v)}if v else None
def pct(v,p):
 if not v:return None
 v=sorted(v);z=(len(v)-1)*p;i=int(z);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(z-i)
for a in('A','B'):
 selected=list(range(16))if a=='A'else list(range(16,32))
 for stage in(0,1):
  for rank in range(8):
   p=r/'results'/f'glm-transient-Model-r8868-{run}-{a}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes());cases=q.get('multi_batch_cases',[]);good=len(cases)==2 and all(c['generated_ids']==[gold['generated_ids'][i]for i in selected]and c['stopped_at_EOS']==[gold['stopped_at_EOS'][i]for i in selected]for c in cases);allgold&=good
   owners.append({'replica':a,'stage':stage,'rank':rank,'SHA256':sha(p),'normal':bool(q.get('completed')and q.get('worker_shutdown_completed')and q.get('post_close_Torch_liveness')),'botharms_fullgold_EOS':good,'load_s':q.get('actual_owner_weight_load_wall_seconds')})
   if stage==1 and rank==0:tails[a]=q
 arms[a]=[]
 for c in tails[a].get('multi_batch_cases',[]):
  t=c['tail_ready_metrics'];gaps=sum(t['ITL_seconds_per_request'],[]);arms[a].append({'phase':c['phase'],'B':c['batch'],'M':c['micro'],'prefill_s':c['tail_prefill_CAPI_return_seconds'],'prefill_input_TPS':c['prefill_input_tokens_per_service_second'],'driver_E2E_s':c['elapsed_seconds'],'emitter_decode_TPS':t['aggregate_useful_decode_tokens_s'],'emitter_decode_TPS_per16cards':t['decode_tokens_s_per_card'],'driver_E2E_useful_TPS':t['secondary_phase_E2E_tokens_s'],'actual_user_firstlast_TPS':summary(t['request_useful_decode_tokens_s']),'actual_user_TTFT_s':summary(t['TTFT_seconds_per_request']),'ITL_P95_s':pct(gaps,.95),'ITL_P99_s':pct(gaps,.99),'max_gap_s':max(gaps),'scope':'Localemitter clock perworld16; formaljointperformance comesfrom commonCPUcollector'})
clock={k:{kk:vv for kk,vv in v.items()if kk!='per_request'}for k,v in audit['common_clock']['metrics'].items()}if audit else None
mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8868.json';m=json.loads(mp.read_bytes());normal=sum(x['normal']for x in owners)
out={'hardware':'BW1000','level':'wholeModel_DP','status':'QUALIFIED'if w.get('qualified')and normal==32 and allgold else'REJECTED_OR_INCOMPLETE','qualified':bool(w.get('qualified')and normal==32 and allgold),'producer':w['producer'],'run':run,'Slurm':w['jobs'],'scope':'Two independent TP8PP2/world16 B16replicas, fixedsame32distinct477 split16+16,I512O128;jointM8 thenjointM16 with2arms/oneweightloadperreplica;total32cards;n03/n07/n15/n16;60min3500;qualified567native68bdbd/PVtile2quantreuseOFF','sourceclaim_commit':'797da108ffd04c50423b4813e990d68f93a90142','manifest_SHA256':sha(mp),'entry_SHA256':sha(r/'incoming/dispatch_GLM_resident_exact_Model_r8868.py'),'DSO_SHA256':m['model_SHA256'],'all32normal':normal,'allowner_botharms_fullmapped477IDs_EOS':allgold,'owner_report_path_template':'results/glm-transient-Model-r8868-{run}-{replica}-stage{stage}-rank{rank}.json','owner_reports':owners,'local_emitter_arms':arms,'common_CPU_collector_clock':clock,'clock_source':{k:v for k,v in audit['common_clock'].items()if k!='metrics'}if audit else None,'communicator_ID_fingerprints':audit.get('communicator_ID_fingerprints')if audit else None,'independent_comms':audit.get('independent_replica_ID_sets')if audit else None,'watch':{'file':wp.name,'SHA256':sha(wp),'state':w['state'],'audit_returncode':w.get('audit_returncode'),'audit_SHA256':w.get('audit_SHA256'),'audit_stderr_tail':w.get('audit_stderr_tail'),'postKFD_passed':w.get('postKFD_passed'),'postKFD':w.get('postKFD')},'historical_PP4_reference':{'producer':'910371.381','decode_TPS':70.149917,'TPS_per_card':2.19218490625,'scope':'Olderprovider/B128M16/tworepeatedfamilies;descriptive only,notmatchedfreshPP4control'},'no_extra_load_arm_retry_extension':True,'comparison':'BothjointM8/M16 actualoverlap and commonarrivalclock;fixedtotal32distinct,noisolatedarms. Constantperuser512/128,nonEOS/padaccounting unchanged. Peruser(n-1)/(last-first),notaggregate/B.'}
columns=['replica','stage','rank','SHA256','normal','botharms_fullgold_EOS','load_s'];out['owner_reports']={'columns':columns,'rows':[[x[k]for k in columns]for x in owners]}
out.pop('historical_PP4_reference')
for cases in out['local_emitter_arms'].values():
 for c in cases:c.pop('scope',None)
for p in('jointM8','jointM16'):out['common_CPU_collector_clock'][p].pop('scope',None)
out['clock_scope']='Samecollector successfulpacketarrivals incltransport; localemitterseparate; actualnonEOS(n-1)/(last-first),noaggregate/B;noisolatedsum/52.70x2/64requests.'
out['status']='QUALIFIED_NEGATIVE_M16'if out['qualified']and clock['comparison']['M16_over_M8_decode_ratio']<1 else out['status']
assert len(json.dumps(out,ensure_ascii=False,indent=2).encode())<16000
print(json.dumps(out))
