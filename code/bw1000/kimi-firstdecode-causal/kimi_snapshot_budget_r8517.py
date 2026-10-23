"""Conservative raw snapshot byte bounds from actual TP8 native capture sites.

No GPU work. F32 is used as an upper bound for every floating intermediate,
including actual BF16 captures. Whole KV-page tails are retained only remotely;
valid-prefix masking remains the comparison contract.
"""
import math

HIDDEN=7168
HEADS=12
DIM=128
PROJECTION=HEADS*DIM
PAGE=256
KV_DIM=192

def snapshot_bound(job, *, state_rows='target', observe_kv=False):
    n=len(job['rows']);widths=list(map(len,job['inputs']));t=sum(widths)
    assert 0<n<=4 and t<=2048 and len(job['pasts'])==n
    assert all(p+w<=1024 for p,w in zip(job['pasts'],widths))
    assert state_rows in ('target','all')
    union_rows=2 if state_rows=='target' else 2*n
    # Original delta forward: each named raw capture, plus the two active
    # BaseQuantization::forward_allreduce captures of its one output projection.
    kda_features={
        'input':HIDDEN,'qkv_projected':3*PROJECTION,
        'conv_output':3*PROJECTION,'silu_qkv':3*PROJECTION,
        'f_a':DIM,'gate_projected':PROJECTION,'beta_projected':HEADS,
        'kda_output':PROJECTION,'normalized_f32':PROJECTION,
        'output_gate_input':PROJECTION,'gate_sigmoid':PROJECTION,
        'gated':PROJECTION,'output':HIDDEN,
        'row_output_local':HIDDEN,'row_output_reduced':HIDDEN}
    kda_intermediates=4*t*sum(kda_features.values())
    conv_row=3*PROJECTION*3*2;ssm_row=HEADS*DIM*DIM*2
    kda_states=2*union_rows*(conv_row+ssm_row)
    # I32 offsets/read/write rows; F32 A_log and dt_bias, no dtype conversion.
    kda_metadata=(n+1+2*n)*8+(HEADS+PROJECTION)*4
    mla_features={'mla_input':HIDDEN,'mla_q_a':1536,'mla_q_norm':1536,
        'mla_q_b':HEADS*192,'mla_kv_a':512+64,'mla_kv_norm':512,
        'mla_kv_b':HEADS*(128+128),'mla_key':HEADS*192,
        'mla_value_padded':HEADS*192,'mla_attention':HEADS*192,
        'mla_gate_linear':PROJECTION,'mla_gate_sigmoid':PROJECTION,
        'mla_gated':PROJECTION,'mla_output':HIDDEN}
    mla_intermediates=4*t*sum(mla_features.values())
    pages=sum(math.ceil((p+w)/PAGE)for p,w in zip(job['pasts'],widths))
    assert pages<=16
    one_page=2*PAGE*HEADS*KV_DIM*2
    mla_pages=2*pages*one_page if observe_kv else 0
    mla_metadata=2*(n+1+n+n+4*n+t)*8 if observe_kv else 0
    return dict(packed_tokens=t,requests=n,conv_row_bytes=conv_row,
        ssm_row_bytes=ssm_row,state_union_rows_upper=union_rows,
        KDA_state_before_after_bytes=kda_states,
        KDA_intermediate_bytes_upper=kda_intermediates,
        KDA_raw_bytes_upper=kda_intermediates+kda_states+kda_metadata,
        MLA_raw_bytes_upper=mla_intermediates+mla_pages+mla_metadata,
        MLA_active_page_union_upper=pages,MLA_one_page_bytes=one_page,
        MLA_before_after_page_bytes_upper=mla_pages,
        KDA_floating_feature_captures=kda_features,
        MLA_floating_feature_captures=mla_features,
        source_dtype_policy='BF16/F32 floating captures bounded by4B; actual integer metadata upper8B; pool BF16 exact2B',
        scope='one selected layer / one native owner / one job; raw files remote only')

def hidden_copy_bound(target_tokens, points):
    assert 0<target_tokens<=512 and points>0
    # trace_hidden narrows actual packed offsets before contiguous and D2H.
    return target_tokens*HIDDEN*2*points
