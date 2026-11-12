from pathlib import Path
import hashlib,importlib.util,json,os,resource,sys,time,traceback
W=Path(__file__).resolve().parents[1];S=W/'source';F=S/'framework';epoch=float(os.environ['SOURCE_EPOCH']);deadline=float(os.environ['SOURCE_COMPARE_DEADLINE']);started=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==780;sys.path[:0]=[str(F/'tools'),str(F/'generation/src')];result=None
try:
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def forbid(*_a,**_k):raise RuntimeError('GPU context forbidden in source comparison')
 torch.cuda._lazy_init=forbid;deps=json.loads((S/'DEPENDENCIES.json').read_bytes());import hygon_sdxl_compare_first_step as cmp
 sys.argv=[str(F/'tools/hygon_sdxl_compare_first_step.py'),'--reference',deps['reference'],'--policy',str(S/'policy.json'),'--observed',str(W/'audit/replay/observed.pt'),'--receipt',str(W/'audit/replay/summary.json'),'--output',str(W/'comparison.private.json'),'--accelerator','cuda'];rc=cmp.main();comparison=json.loads((W/'comparison.private.json').read_bytes());diag=json.loads((W/'GPU-result.json').read_bytes())['diagnostic'];previous=json.loads((S/'PREVIOUS_HIP.json').read_bytes());identities={n:dict(A100_SHA256=pin,HIP_SHA256=previous['operand_SHA256'][n],exact_identity=pin==previous['operand_SHA256'][n]) for n,pin in diag['operand_SHA256'].items()}
 result=dict(evidence_processed=True,numerical_passed=comparison['passed'],comparison_returncode=rc,checks=comparison['checks'],metrics={n:{k:v for k,v in row.items() if k not in ['per_tensor','worst_relative_l2']} for n,row in comparison['metrics'].items()},comparison_SHA256=hashlib.sha256((W/'comparison.private.json').read_bytes()).hexdigest(),operand_identity_vs_original_HIP176=identities,previous_HIP_native_vs_original_gradient=previous['native_vs_original_gradient'],actual_A100_native_vs_original_gradient=diag['native_vs_original_gradient'],local_F64_native_check=diag['native_vs_local_F64_rounded'],CUDA_initialized=False,raw_operand_cross_host_transfer_bytes=0,Model_GO=False)
except BaseException as error:
 (W/'compare-failure.private.txt').write_text(traceback.format_exc());result=dict(evidence_processed=False,numerical_passed=False,error_type=type(error).__name__,Model_GO=False)
result.update(epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(W/'comparison-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if not result['evidence_processed']:raise SystemExit(1)
