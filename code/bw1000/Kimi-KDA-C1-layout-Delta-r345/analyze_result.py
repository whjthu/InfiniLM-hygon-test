from pathlib import Path
import json,hashlib,statistics,math
R=Path(__file__).resolve().parents[3];D=R/'results/bw1000/2026-11-02';med=statistics.median;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted(D.glob('Kimi345-C1-layout-Delta-910371.635-node*.json'));assert len(files)==2
reports=[json.loads(p.read_bytes())for p in files];owners=[w for r in reports for w in r['workers']];assert sorted(w['rank']for w in owners)==list(range(16))
for r in reports:assert r['completed']and r['normal_close']and r['C']==r['T']==1 and r['pool']==3 and r['Math']==0
bykey={};numerics={name:0. for name in ['output','conv','state']};same_card=[];first_metadata=[]
for w in owners:
 assert w['completed']and w['normal_close']and w['fused_counts']==[160,0]and len(w['samples'])==320
 assert w['first_ON_dispatch']['counts']==[1,0]and w['first_ON_dispatch']['metadata']['eligible']
 first_metadata.append({'rank':w['rank'],'data':w['first_ON_dispatch']})
 for s in w['samples']:
  assert s['profile']is None and s['source_rows_unchanged']
  assert s['warmup']==(s['trial']<2)
  assert all(math.isfinite(v)for n in s['numerical'].values()for v in n.values())
  for name in numerics:numerics[name]=max(numerics[name],s['numerical'][name]['max_abs'])
  bykey.setdefault((s['block'],s['trial']),[]).append((w['rank'],s))
 ratios=[]
 for n in range(8):
  off=[s['API_wall_ms']for s in w['samples']if not s['warmup']and s['round']==n and s['arm']=='OFF'];on=[s['API_wall_ms']for s in w['samples']if not s['warmup']and s['round']==n and s['arm']=='ON'];assert len(off)==len(on)==16;ratios.append(med(on)/med(off))
 same_card.append({'rank':w['rank'],'paired_round_ratios':ratios,'paired_median_ratio':med(ratios)})
max16=[]
for (block,trial),group in sorted(bykey.items()):
 assert len(group)==16
 rank,s=max(group,key=lambda v:v[1]['API_wall_ms']);assert len({(v['arm'],v['round'],v['warmup'])for _,v in group})==1
 max16.append({'block':block,'trial':trial,'round':s['round'],'arm':s['arm'],'warmup':s['warmup'],'max16_APIwall_ms':s['API_wall_ms'],'critical_rank':rank})
rounds=[]
for n in range(8):
 off=[s for s in max16 if s['round']==n and s['arm']=='OFF'and not s['warmup']];on=[s for s in max16 if s['round']==n and s['arm']=='ON'and not s['warmup']];assert len(off)==len(on)==16
 blocks=sorted({s['block']for s in off});bracket=[med(s['max16_APIwall_ms']for s in off if s['block']==b)for b in blocks];a=med(s['max16_APIwall_ms']for s in off);b=med(s['max16_APIwall_ms']for s in on)
 rounds.append({'round':n,'OFF_median_ms':a,'ON_median_ms':b,'ON_over_OFF':b/a,'OFF_block_medians_ms':bracket,'OFF_bracket_relative_spread':abs(bracket[1]-bracket[0])/med(bracket)})
off=med(s['max16_APIwall_ms']for s in max16 if not s['warmup']and s['arm']=='OFF');on=med(s['max16_APIwall_ms']for s in max16 if not s['warmup']and s['arm']=='ON')
post=json.loads((D/'Kimi345-GPU635-post.json').read_bytes());proc=json.loads((D/'Kimi345-process-post.json').read_bytes());assert all(v['KFD_empty']and v['control30040_bindable']for v in post['KFD_and_port_post']);assert all(not v['own_driver_pids']for v in proc['process_post'])
result={'actual_GPU':'910371.635 COMPLETED0/76s','source_commit':'fac30e0f4c7595272284b947ea2c1bf71e77b526','owners':16,'all_normal_close':True,'all_first_ON_chosen1_fallback0':True,'all_final_chosen160_fallback0':True,'profile_events':'OFF throughout; no new GPUevent measurement','scope':'ActualTP16 fullDelta API including inputupload/forward/streamdrain/outputcopy. Historicalteacher512 checkpoint/state C1T1 pool3 Math0; notlive32K,model,service orquality benchmark','raw_SHA256':{p.name:sha(p)for p in files},'analyzer_SHA256':sha(Path(__file__)),'numerical_max_abs':numerics,'max16_OFF_median_ms':off,'max16_ON_median_ms':on,'aggregate_ratio':on/off,'paired_round_median_ratio':med(r['ON_over_OFF']for r in rounds),'max_OFF_bracket_spread':max(r['OFF_bracket_relative_spread']for r in rounds),'all_OFF_brackets_within3pct':all(r['OFF_bracket_relative_spread']<=.03 for r in rounds),'rounds':rounds,'same_card':same_card,'max16_samples':max16,'first_ON_metadata':first_metadata,'post':post['KFD_and_port_post'],'process_post':proc['process_post'],'Model_gain':False,'GPU_replayed':False}
target=D/'Kimi345-GPU635-paired-fullDelta-analysis.json';assert not target.exists();target.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:result[k]for k in ['actual_GPU','numerical_max_abs','max16_OFF_median_ms','max16_ON_median_ms','aggregate_ratio','paired_round_median_ratio','max_OFF_bracket_spread','all_OFF_brackets_within3pct','rounds']},indent=2))
