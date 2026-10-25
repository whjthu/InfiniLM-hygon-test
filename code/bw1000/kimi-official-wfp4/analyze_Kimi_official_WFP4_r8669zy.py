"""CPU consumer of completed component outputs; raw tensors stay on the cluster."""
import pathlib,json,hashlib,sys,re,numpy as np,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';handle=sys.argv[1];assert handle.startswith('910374.')and handle.split('.')[1].isdigit();label='Kimi-official-WFP4-r8669zy';m=json.loads((r/'incoming/Kimi-official-WFP4-manifest-r8669zy.json').read_text());plan=[0,1,0,1,1,0,1,0,0,1];out={'scope':'ActualfullMoEL1 component only; not Modeltokens/TPS','actual_handle':handle,'qualified':False,'ranks':[],'metrics':[]};canonical={}
try:
 for node in range(2):
  p=r/'results'/f'{label}-{handle}-node{node}.json';q=json.loads(p.read_text());assert q['completed']and q['native']['all8normal_before_communicators']and q['normal_CUDA_cleanup'];folder=pathlib.Path(q['raw_directory'].replace('/refactor/',str(r)+'/'))
  for row in q['native']['reports']:
   rank=row['rank'];globalrank=node*8+rank;assert row['global_tp_rank']==globalrank and row['actual_tp_size']==16 and row['normal_Runtime_close']and row['module_closed'];keys=row['weight_binding'];assert [z['local_experts']for z in keys]==[896,896]and [z['expert_TP_size']for z in keys]==[16,16]and keys[1]['global_expert_start']==0 and [z['W13_repacked']for z in keys]==[False,True]
   for kind,binding in enumerate(keys):
    parsed=[re.fullmatch(r'experts\.(\d+)\.(w[123]\.weight_(?:packed|scale))',k)for k in binding['keys']if k.startswith('experts.')]
    actual={(int(z.group(1)),z.group(2))for z in parsed if z};start=0;E=896
    expected={(e,w+'.weight_'+suffix)for e in range(start,start+E)for w in ['w1','w2','w3']for suffix in ['packed','scale']}
    assert all(parsed)and actual==expected,('actualFQNcoverage',globalrank,kind)
   
   expected_hits=0
   for arm_record in row['arms']:
    aidx=arm_record['arm'];expected_hits+=plan[aidx];assert arm_record['I192_counter_delta']==[1,16] and arm_record['official_W13_counter_after']==[expected_hits,16*expected_hits]
   arrays=[]
   for arm in range(10):
    raw=(folder/f'rank{rank}-arm{arm}.bin').read_bytes();assert len(raw)==229376;bits=np.frombuffer(raw,np.uint16);f=(bits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();digest=hashlib.sha256(raw).hexdigest()
    if arm not in canonical:canonical[arm]=digest
    assert canonical[arm]==digest,('TP16rankagreement',globalrank,arm)
    arrays.append((raw,f))
   repeat=all(arrays[i][0]==arrays[plan[i]][0]for i in range(10));assert repeat,('ownrepeat',globalrank)
   a=arrays[0][1].astype(np.float64);b=arrays[1][1].astype(np.float64);err=np.abs(a-b);den=np.sqrt(np.mean(a*a));relative=float(np.sqrt(np.mean((a-b)**2))/max(den,1e-30));maxnorm=float(np.max(err)/max(np.max(np.abs(a)),1e-30));screen=relative<=m['predeclared_numerics']['gross_error_screen']['relativeRMS_max']and maxnorm<=m['predeclared_numerics']['gross_error_screen']['maxabs_over_baseline_maxabs_max']
   out['ranks'].append({'globalrank':globalrank,'finite':True,'ownrepeat_exact':repeat,'maxabs':float(np.max(err)),'RMSE':float(np.sqrt(np.mean((a-b)**2))),'relativeRMS':relative,'maxabs_over_baseline_maxabs':maxnorm,'abs_error_p50_p95_p99':np.quantile(err,[.5,.95,.99]).tolist(),'bitdifferences_diagnostic_only':int(np.count_nonzero(np.frombuffer(arrays[0][0],np.uint16)!=np.frombuffer(arrays[1][0],np.uint16))),'gross_error_screen_pass':screen})
   assert all(z['participants_sum']==16 for z in row['barriers']);out['metrics'].append({'globalrank':globalrank,'ordinary':[{'arm':z['arm'],'official_W13':z['candidate_ON'],'full_MoE_complete_wall_ms':z['full_MoE_complete_wall_ms']}for z in row['arms']if z['kind']=='ordinary']})
 out['all16finite_rankagreement_ownrepeat']=True;out['gross_error_screen_pass']=all(z['gross_error_screen_pass']for z in out['ranks']);out['qualified']=out['gross_error_screen_pass'];out['numeric_scope']='Screen predeclaredbeforeGPU, not a mathematicalerrorbound or tokenacceptance. Crossvariant bitdiff not rejection.'
except BaseException as e:out['error']=repr(e)
finally:
 p=r/'results'/f'{label}-numeric-audit-{handle}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:out.get(k)for k in ['actual_handle','qualified','gross_error_screen_pass','error']}))
if not out['qualified']:raise SystemExit(1)
