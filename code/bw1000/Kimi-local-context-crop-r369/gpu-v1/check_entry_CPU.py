"""Real tiny factory + base333 session + privatecontext + exact GPU run_plan, CPU only."""
from pathlib import Path
import json,sys,types,hashlib

def main():
 D=Path(__file__).resolve().parent;R=Path('/refactor');B=D.parent
 sys.path[:0]=[str(D),str(B),str(R/'incoming'),str(R/'incoming/Kimi-long-draft-RPC-CPU-r333')]
 import torch
 from dflash2_creator_factory_r9007 import make_factory
 from long_rpc import DraftSession,TorchFeatureCodec
 from window_context import WindowContext
 from GPU_driver import run_plan,schedule
 cfg=json.loads((R/'incoming/config.json').read_bytes());cfg.update(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4)
 cfg['dflash_config']=dict(cfg['dflash_config'],selector_rank=8,selector_top_k=4)
 torch.set_num_threads(1);torch.manual_seed(369)
 model=make_factory(cfg,R/'incoming')().to(torch.bfloat16).eval();head=torch.nn.Linear(32,cfg['vocab_size'],bias=False,dtype=torch.bfloat16).eval()
 c=WindowContext(model,'CPU-small-only',enabled=True);session=DraftSession(c,head,TorchFeatureCodec('cpu',raw_dim=160),enabled=True)
 runtime=types.SimpleNamespace(session=session);report={}
 with torch.inference_mode():
  base=(torch.randn(1,512,160,dtype=torch.bfloat16)*.1).contiguous().view(torch.uint16).numpy().tobytes()
  run_plan(runtime,report,lambda:None,lambda n:base[:n*160*2])
 assert len(report['queries'])==66 and report['normal_session_close']and c.raw is None and c.projected is None
 assert schedule()==[('warm',False),('warm',True)]+[('measure',x)for x in [False,True,True,False]*5]
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_loaded=False,actual_entry_run_plan=True,actual333_DraftSession=True,actual_private_WindowContext=True,query_records=len(report['queries']),protocol_commands=report['protocol_commands'],lengths=[v['L']for v in report['contexts']],normal_session_close=report['normal_session_close'],context_content_unchanged=all(v['full_context_content_unchanged']for v in report['contexts']),max_hidden_NRMSE=max(v['numeric']['hidden']['NRMSE']for v in report['queries']),max_logits_NRMSE=max(v['numeric']['logits']['NRMSE']for v in report['queries']),source='GPU_driver run_plan imported unchanged; smallBF16 parameters only; not strictloader/GPU/floor qualification')))

if __name__=='__main__':main()
