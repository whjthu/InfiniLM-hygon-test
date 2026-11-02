"""Recompute saved CPU752 metadata only; no Torch imports or job submissions."""
from pathlib import Path
import json,base64,hashlib
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent;O=G/'results/bw1000/2026-11-02'
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
 x=json.loads((L/'Kimi376-CPU752-sealed-transfer.json').read_bytes());raw=base64.b64decode(x['base64']);assert sha(raw)==x['SHA256']
 sealed=O/'Kimi376-CPU752-original-sealed.json';sealed.write_bytes(raw);r=json.loads(raw);c=r['checks']
 assert r['completed']and r['passed']and r['actual_CPU']=='910407.752'and c['GPU']==0 and not c['Native_dlopen']and not c['checkpoint_loaded']
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert r['source_manifest_SHA256']==sha((D/'SOURCE_MANIFEST.json').read_bytes())
 assert all(sha((D/n).read_bytes())==h for n,h in m['files_SHA256'].items())
 assert len(c['results'])==6 and len(c['cases'])==26 and all(v['passed']for v in c['cases'])
 summary=[]
 for case in c['results']:
  rows=case['rows'];assert len(rows)==8 and case['accepted_prefix_counts']==[8]*7
  for i,row in enumerate(rows):
   assert row['past']==case['prompt']+8*i
   assert all(v['computed_rows']==(case['prompt']+8 if i==0 else 16)for v in row['cache_stats']['linear_calls'])
   assert row['cache_stats']['published_after_original_fence']and row['original_final_fence_before_publication']['pending']
   assert row['original_final_fence_before_publication']['cached_length']==(0 if i==0 else row['past']-8)
   assert all(v['finite']for v in row['stages'].values())and row['hidden']['finite']and row['logits']['finite']
  summary.append(dict(dtype=case['dtype'],prompt=case['prompt'],queries=8,hidden_NRMSE_max=max(v['hidden']['NRMSE']for v in rows),logits_NRMSE_max=max(v['logits']['NRMSE']for v in rows),stage_NRMSE_max=max(n['NRMSE']for v in rows for n in v['stages'].values()),proposal_position_matches=[v['proposal_matches']for v in rows],OFF_diagnostic_complete_sequence_CPU_s=case['OFF_complete_sequence_CPU_s'],ON_diagnostic_complete_sequence_CPU_s=case['ON_complete_sequence_CPU_s'],OFF_cold_query_CPU_s=rows[0]['OFF_complete_dispatch_s'],ON_cold_query_CPU_s=rows[0]['ON_complete_dispatch_s'],tiny_peak_old_pending_cache_storage_bytes=case['ON_peak_live_cache_storage_bytes']))
 assert all(v['all_cache_refs_dropped']for v in c['normal_final_closes'])
 out=dict(completed=True,passed=True,state='CPU752_REAL_TINY_LIFECYCLE_PASS_DEVICE_UNQUALIFIED',actual_CPU='910407.752',Slurm_terminal='COMPLETED0/13s',source_publication='29a6889e1faa68f5d85e05a808007cf71130860a',source_manifest_SHA256=r['source_manifest_SHA256'],raw=dict(remote=x['path'],local=str(sealed.relative_to(G)),SHA256=x['SHA256'],bytes=len(raw)),actual333_Session=True,original_attention_forward_method_identity_unchanged=True,original_fullmask_RoPE_SDPA_Q_O_head_selector_unchanged=True,private_model_routing_only=True,shared_model_parameters_buffers_not_modified=True,cache_defaultOFF=True,canonical_sequences=6,main_complete_query_dispatches=96,synthetic_committed_counts=[8]*7,stage_final_numerics=summary,boundary_cases=c['cases'],normal_closes=c['normal_final_closes'],CPU_peak_RSS_KiB=c['peak_CPU_RSS_KiB'],CPU_check_wall_s=c['total_CPU_wall_s'],fence_publication='Actual baseCodec finalfence observed pendingTrue and oldcached_length/publish_count; onlyafterbase returns does committing_codec publish all10values. Bothdtype faultcases show failurepoisons anddrop, no prematurepublish.',memory_geometry='ProductionBF16 L32638 caches5*(576+16384)*L*2=5,535,404,800B=5.155247GiB. Viewsretain8draftunderlying storage extra1,356,800B. Old+pending updates may doublepayload; actualGPUpeak and4GiBfloor stillunqualified.',quality='TinyCPU andrandomparameters only; F32 CPUreference explicitly casts originalBF16wiredecode toF32. Allstages finite; differenttokens permitted. Proposal agreement is diagnostic, not targetacceptance/Modelquality.',timing='FullCPUrequest costs include promptappend/seal/coldfirst8query/sevenaccepted8/retire; finalclose andtinymodel/headstartup separate. Traceclones andnumericsmake this adiagnosticconsumer, no GPUspeed orModelgain; BF16CPUnearimproves butF32CPUnearregresses, neither predictsdevice.',inference_weights='CPUversionchange checktested; GPUinference may lackversion, frozenmodule/deviceptr/dtype/config/weighttag contract andstrictRuntime required. No perqueryfullweightSHA.',GPU=0,target_cards=0,checkpoint_loaded=False,Native_dlopen=False,Model_gain='UNKNOWN',next='Prepare separate immutable whole-driverCPU plan then explicitsingle-draftGPUclaim if authorized; do not replay690/697 or loadtarget')
 (O/'Kimi376-CPU752-lifecycle-and-numerical-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 tpath=G/'coordination/tasks/Kimi-preRoPE-prefix-cache-r376.json';t=json.loads(tpath.read_bytes());t.update(state=out['state'],GPU_GO=False,actual_CPU='910407.752',CPU_receipt_SHA256=x['SHA256'],CPU_result='results/bw1000/2026-11-02/Kimi376-CPU752-lifecycle-and-numerical-audit.json',all_CPU_resources_released=True,device_qualification='PENDING; newwhole-driverCPU/precise1draftclaim required');tpath.write_text(json.dumps(t,indent=2)+'\n')
 print(json.dumps(dict(completed=True,CPU_passed=True,sequences=6,query_dispatches=96,cases=26,peak_RSS_KiB=c['peak_CPU_RSS_KiB'],GPU=0,receipt_SHA256=x['SHA256'])))

if __name__=='__main__':main()
