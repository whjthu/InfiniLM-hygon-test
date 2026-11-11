"""Summarize completed paired forward metadata without reading raw tensors."""
import argparse,json,statistics
from pathlib import Path

def summarize(v,p):
    assert v['independent_original_terminal_verified'] and v['completed_rank_arms']==16
    hip={x['rank']:x['Hygon_loss'] for x in p['loss_pairs'] if x['ordinal']==1}
    rows=[]
    for r in v['rank_results']:
        c=r['comparisons'];native=c['native_vs_original']['actual_loss'];original=c['native_vs_original']['reference_loss'];alternate=c['torch_chunk_vs_native']['actual_loss'];h=hip[r['rank']]
        rows.append(dict(rank=r['rank'],original_A100_loss=original,new_native_A100_loss=native,forced_Torch_chunk_A100_loss=alternate,original_Hygon_loss=h,native_original_logit_max_abs=c['native_vs_original']['logits']['max_abs'],native_original_logit_relative_l2=c['native_vs_original']['logits']['relative_l2'],alternate_native_logit_relative_l2=c['torch_chunk_vs_native']['logits']['relative_l2'],exact_native_reproduction=r['exact_native_reproduction'],new_native_to_Hygon_loss_abs=abs(native-h),forced_chunk_to_Hygon_loss_abs=abs(alternate-h),forced_chunk_closer_in_scalar_loss=abs(alternate-h)<abs(native-h),actual_chunk_q=r['calls'][0]['first_call_shapes']['q']))
    return dict(rank_rows=rows,planned_rank_arms=16,completed_rank_arms=16,optimizer_updates=0,exact_native_reproduction_ranks=[r['rank'] for r in rows if r['exact_native_reproduction']],unqualified_native_reproduction_ranks=[r['rank'] for r in rows if not r['exact_native_reproduction']],all_rank_scalar_comparison_conditional=True,conditional_closer_scalar_loss_ranks=sum(r['forced_chunk_closer_in_scalar_loss'] for r in rows),native_original_logit_relative_l2_max=max(r['native_original_logit_relative_l2'] for r in rows),native_to_Hygon_loss_abs_median=statistics.median(r['new_native_to_Hygon_loss_abs'] for r in rows),alternate_to_Hygon_loss_abs_median=statistics.median(r['forced_chunk_to_Hygon_loss_abs'] for r in rows),native_to_Hygon_loss_abs_max=max(r['new_native_to_Hygon_loss_abs'] for r in rows),alternate_to_Hygon_loss_abs_max=max(r['forced_chunk_to_Hygon_loss_abs'] for r in rows),original_keeper_failure_preserved=True,numerical_acceptance=False,formal_performance=False,Model_GO=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--terminal',type=Path,required=True);parser.add_argument('--stationary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();v=summarize(json.loads(args.terminal.read_bytes()),json.loads(args.stationary.read_bytes()))
    with args.output.open('x') as f:json.dump(v,f,indent=2)
    print(json.dumps({k:v[k] for k in v if k!='rank_rows'}))
