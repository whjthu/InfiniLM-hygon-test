"""Caller-only delta draft. Call only at the original reset+barrier boundary.
No factory, Native recompilation, model creation, input mutation or dispatch.
Parent owner must retain its source/input/provider checks and per-rank barrier.
"""
import os,threading
# Correct full parent pin is required, never a prefix comparison.
PARENT_B_SHA256='02b0b1e8cf529fcb73ad6bd429c78d158beb6d19d4577866977abe97f686136d'
def select_after_original_reset(ctx,arm,*,scope,row,verified_Native_SHA256,barrier_returned):
    if scope.get('GPU_GO') is not True or scope.get('Model_GO') is not True:
        raise ValueError('NEW_MODEL_CALLER_SCOPE_NOT_ADOPTED')
    if arm not in ('OFF','ON') or threading.get_native_id()!=os.getpid() or barrier_returned is not True:
        raise ValueError('ORIGINAL_OWNER_RESET_BARRIER_REQUIRED')
    if verified_Native_SHA256!=PARENT_B_SHA256:
        raise ValueError('EXACT_EXISTING179B_PROVIDER_REQUIRED')
    geo=ctx.geometry()  # Existing same-owner typed CABI, not a constructed RankInfo.
    if not (geo['Model_READY'] and geo['weights_loaded'] and geo['TP']==16 and geo['PP']==1 and geo['local_attention_heads']==4 and 0<=geo['tp_rank']<16 and geo['rows']==1 and geo['capacity_per_row']==32768):
        raise ValueError('ACTUAL_FULL_TP16_MODEL_GEOMETRY')
    if row['max_new_tokens']!=4096 or len(row['prompt_token_ids'])+4096>32768 or row['body'].get('temperature')!=0.0:
        raise ValueError('ORIGINAL_FULL_REQUEST_CONTRACT')
    if os.environ.get('GLM_SELECTED_VALUE_PACKED_LAYER0')!='0':
        raise ValueError('OLD_PACKED_BRANCH_MUST_BE_OFF')
    # Parent cfg must also freeze glm_selected_value_packed_layer0=False before
    # create. The existing constructor rejects JSON/env disagreement.
    os.environ['GLM_HISTORY_VALUE_ONLY_V1']='1' if arm=='ON' else '0'
    return dict(arm=arm,history_value_only_env=os.environ['GLM_HISTORY_VALUE_ONLY_V1'],Native_SHA256=verified_Native_SHA256,actual_TP=geo['TP'],actual_PP=geo['PP'],actual_local_heads=geo['local_attention_heads'],actual_global_rank=geo['tp_rank'],packed_env=os.environ['GLM_SELECTED_VALUE_PACKED_LAYER0'],GPU_dispatch_performed=False,qualification=False)
