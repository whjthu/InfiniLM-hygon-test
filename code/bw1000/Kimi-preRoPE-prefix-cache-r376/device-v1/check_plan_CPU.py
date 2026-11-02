"""Exact shared18sequence plan on pinned actualtiny BF16/333 consumers, no strictload/GPU."""
from pathlib import Path
import sys,json,copy,time,resource

def main():
 D=Path(__file__).resolve().parent;B=D.parent;R=Path('/refactor');inc=R/'incoming'
 sys.path[:0]=[str(D),str(B),str(inc/'Kimi-long-draft-RPC-CPU-r333'),str(inc)]
 import torch
 from dflash2_creator_factory_r9007 import make_factory
 from long_rpc import TorchFeatureCodec
 from sequence_consumer import plan,run_sequence,run_plan,ALERT,INVESTIGATE
 fences={'count':0}
 class CountCodec(TorchFeatureCodec):
  def fence(self):super().fence();fences['count']+=1
 torch.set_num_threads(4);torch.manual_seed(37601);start=time.perf_counter()
 cfg=json.loads((inc/'config.json').read_bytes());cfg.update(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4);cfg['dflash_config']=dict(cfg['dflash_config'],selector_rank=8,selector_top_k=4)
 model=make_factory(cfg,inc)().to(torch.bfloat16).eval();head=torch.nn.Linear(32,163840,bias=False,dtype=torch.bfloat16).eval()
 weights={k:v.clone()for k,v in model.state_dict().items()};head_before=head.weight.clone();modules=[(n,id(v))for n,v in model.named_modules()]
 torch.manual_seed(324);base=(torch.randn(1,512,160,dtype=torch.bfloat16)*.1).contiguous()
 def body(past,n):
  part=base.index_select(1,(torch.arange(past,past+n)%512).long())
  return part.contiguous().view(torch.uint16).numpy().tobytes()
 p=plan();assert len(p)==18 and sum(s['phase']=='formal'for s in p)==12
 assert ALERT==.02 and INVESTIGATE==.10
 refs={};records=[];snapshots=[]
 O=R/'candidates/Kimi-preRoPE-prefix-cache-r376-device-v1-CPU'
 def save_snapshot(value):
  snapshots.append(copy.deepcopy(value));p=O/'last-sequence.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(p)
 with torch.inference_mode():
  records,refs=run_plan(model,head,CountCodec,'cpu',160,'CPUwhole376',body,save_snapshot)
  assert len(records)==18 and all(r['passed']and r['normal_session_close']for r in records)
  assert sum(len(r['queries'])for r in records)==144
  assert sum(len(r['commands'])for r in records)==840
  assert sum(v['cmd']=='append_prompt'for r in records for v in r['commands'])==498
  assert sum(v['cmd']=='append_committed'for r in records for v in r['commands'])==126
  assert fences['count']==912;main_fences=fences['count']
  # Deliberate reference-only numeric failure: realvalues/logits/props already
  # produced, save snapshot BEFORE .10 stop; no invented failedquery fields.
  failed=[]
  try:run_sequence(model,head,CountCodec,'cpu',160,'CPUnegative376',body,p[0],lambda v:failed.append(copy.deepcopy(v)),reference=refs[p[0]['L']],fail_numeric=True)
  except RuntimeError:pass
  else:raise AssertionError('Numeric investigation stop did not fire')
  last=failed[-1];row=last['queries'][-1]
  (O/'expected-failure-all-fields.json').write_text(json.dumps(last)+'\n')
  assert last['completed']and not last['passed']and row['hidden']and row['logits']and len(row['proposals'])==7 and len(row['stages'])==20 and row['investigation_stop']
  assert modules==[(n,id(v))for n,v in model.named_modules()]
  assert all(torch.equal(v,weights[k])for k,v in model.state_dict().items())and torch.equal(head_before,head.weight)
  assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_loaded=False,shared_consumer=True,actual333_sessions=True,sequences=18,warm_queries=48,formal_queries=96,negative_test_extra_queries=1,total_CPU_query_dispatches=145,commands=840,prompt_appends=498,synthetic_committed_appends=126,main_codec_fences=main_fences,all_normal_context_close=True,partial_numeric_failure_saved_all_fields=True,failed_snapshot=dict(error=last['error'],hidden=row['hidden'],logits=row['logits'],proposals=row['proposals'],stage_count=len(row['stages']),state=row['state']),all_original_model_weights_modules_unchanged=True,max_hidden_NRMSE=max(v['hidden']['NRMSE']for r in records for v in r['queries']),max_logit_NRMSE=max(v['logits']['NRMSE']for r in records for v in r['queries']),max_stage_NRMSE=max(n['NRMSE']for r in records for v in r['queries']for n in v['stages'].values()),elapsed_s=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='TinyrandomBF16 only; exact sharedplan andcontext/protocol/partialsave closure, not actualstrictloader/GPU/floor/throughput/quality evidence')))

if __name__=='__main__':main()
