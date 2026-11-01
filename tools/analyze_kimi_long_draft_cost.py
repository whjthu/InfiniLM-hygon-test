"""Independently consume the exact draft-only601 result; never infer acceptance or speedup."""
import json,hashlib,math,statistics,sys
from pathlib import Path

def need(value,message):
 if not value:raise ValueError(message)
def seconds(value):
 need(type(value)in[int,float]and math.isfinite(value)and value>0,'Invalid wall time');return value
def analyze(observation):
 k=observation['runs']['Kimi'];w=k['watch'];r=w['report'];rows={x[0]:x for x in k['accounts']}
 for step,label,node in [('910371.601','Kimi-C1-long-draft-cost-r324','a02r3n15'),('910371.602','Kimi-C1-long-draft-cost-r324-post','a02r3n15'),('910407.638','Kimi-C1-long-draft-cost-r324-keeper','a14r2n20')]:
  need(step in rows and rows[step][1:4]==[label,'COMPLETED','0:0']and rows[step][5]==node,'Exact GPU/post/keeper not completed normally')
 for step,limit in [('910371.601',600),('910407.638',900)]:
  parts=rows[step][4].split(':');need(len(parts)==3 and all(x.isdigit()for x in parts),'Invalid elapsed time');h,m,s=map(int,parts);need(h*3600+m*60+s<=limit,'Runtime scope exceeded')
 need(w['completed']is True and w['actual_GPU']=='910371.601'and w['CPU_keeper_actual']=='910407.638','Wrong actual handles')
 need(w['post_empty']is True and w['post']['step']=='910371.602'and w['post']['node']=='a02r3n15'and w['post']['passed']is True and w['post']['kfd_pids']==[],'Post KFD cleanup unproved')
 need(hashlib.sha256((json.dumps(r,indent=2)+'\n').encode()).hexdigest()==w['report_SHA256'],'Report byte hash mismatch')
 need(r['completed']is True and r['passed']is True and r['normal_close']is True and r['cleanup_errors']==[],'Incomplete/abnormal component')
 need(r['actual_step']=='910371.601'and r['GPU_count']==1 and r['target_cards']==0 and r['HTTP']==0,'Wrong runtime scope')
 need(r['source_manifest_SHA256']=='88ea8f6614b0f37cd82c456685be4bbb46c2164f38d2ec31092082de912a13cf'and r['consumer_manifest_SHA256']=='ac3e886441cfbabecffdd8bfb81a31a0d36f72a8fd6e32bfd6aac89c6c1e2ca2','Wrong source closure')
 need(r['profile']is False and r['synthetic_expanded_short_features']is True and r['target_acceptance_qualified']is False and r['Model_speedup_claim']is False,'Qualification scope mismatch')
 need(r['loader_receipt']['full_SHA256']=='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2'and r['loader_receipt']['BF16_tensors_loaded']==87,'Unbound trained checkpoint')
 need([x['N']for x in r['cases']]==[1024,8192,32760],'Missing context point')
 summary=[]
 for x in r['cases']:
  N=x['N'];need(x['prompt_tokens']==N-8 and x['committed_decode_tokens']==8 and x['normal_retired_closed']is True,'Uncommitted/missing lifecycle')
  need(x['payload_bytes']=={'raw':N*35840*2,'projected':N*7168*2},'Feature payload geometry differs')
  need(len(x['warmups'])==2 and all(seconds(v)for v in x['warmups']),'Warmups incomplete')
  need(len(x['samples'])==8 and [v['index']for v in x['samples']]==list(range(8))and all(v['proposal_count']==7 for v in x['samples']),'Measured query coverage incomplete')
  values=[seconds(v['complete_query_D2H_wall_s'])for v in x['samples']];median=statistics.median(values)
  need(math.isclose(median,x['query_median_s'],rel_tol=1e-12),'Stored median incorrect')
  need(sum(v['rows']for v in x['append_chunks'])==N-8 and all(1<=v['rows']<=512 for v in x['append_chunks']),'Prompt append geometry differs')
  prompt=sum(seconds(v['H2D_projection_copy_fence_wall_s'])for v in x['append_chunks']);need(math.isclose(prompt,x['prompt_H2D_append_wall_s'],rel_tol=1e-12),'Stored prompt cost incorrect')
  for key in ['query_memory_before_bytes','query_peak_allocated_bytes','query_peak_reserved_bytes','init_peak_allocated_bytes']:need(type(x[key])is int and 0<x[key]<=r['total_bytes'],'Invalid peak memory')
  need(x['query_memory_before_bytes']<=x['query_peak_allocated_bytes']<=x['query_peak_reserved_bytes'],'Memory counters inconsistent')
  summary.append({'context_tokens':N,'prompt_H2D_append_s':prompt,'committed8_H2D_append_s':seconds(x['committed8_H2D_append_wall_s']),'query_median_ms':median*1000,'query_min_ms':min(values)*1000,'query_max_ms':max(values)*1000,'query_peak_allocated_GiB':x['query_peak_allocated_bytes']/2**30,'query_peak_reserved_GiB':x['query_peak_reserved_bytes']/2**30,'raw_projected_payload_GiB':sum(x['payload_bytes'].values())/2**30})
 need(r['cases'][0]['short_control']['hidden_relativeRMS_diagnostic']==0. and r['cases'][0]['short_control']['proposal_equal_diagnostic']is True,'Short old/new control differs')
 need(r['cases'][-1]['32761_AR_tail_no_query']is True,'Context tail fallback absent')
 return {'completed':True,'passed':True,'actual_GPU':'910371.601 COMPLETED0/54s','CPU_keeper':'910407.638 COMPLETED0/64s','post':'910371.602 COMPLETED0 KFDempty','report_SHA256':w['report_SHA256'],'samples_per_context':8,'warmups_per_context':2,'context_costs':summary,'load_and_full_checkpoint_hash_wall_s':seconds(r['load_and_full_checkpoint_hash_wall_s']),'query_clock':'Synchronized host wall including five layers, fullcontextcopies, head, selector, proposal D2H; profile OFF','append_clock':'Synchronized fixture H2D, projection/norm, raw/projected copies and fence; CPU fixture generation excluded','cold_initial_append_s':r['cases'][0]['append_chunks'][0]['H2D_projection_copy_fence_wall_s'],'cold_note':'First512append2.64s retained as first-call cold observation. Context points are not matched independent cold starts; no causal kernel attribution.','decision':'COST_DOES_NOT_EXCLUDE_CONDITIONAL_CAPTURE_AND_ACCEPTANCE_CARD','target_acceptance_qualified':False,'Model_speedup_claim':False,'limits':['Periodic synthetic shortfeatures, no true32Ktargetactivations or quality evidence.','Only8queries/context; no throughput scaling or online service qualification.','Target capture/feature transport/verify/replay/accepted useful tokens/emit costs remain unmeasured.','Allocated/reserved memory include model/head; not an incremental production allocation guarantee.'],'next':'Reuse305HOSTclosure; inspect isolated final link and Pythonconsumer CPU closure first. Then prepare one32target+1draft residency for actualdevicecapture and conditional shortest4requestpair. No automatic target reload.'}

if __name__=='__main__':print(json.dumps(analyze(json.loads(Path(sys.argv[1]).read_bytes())),indent=2))
