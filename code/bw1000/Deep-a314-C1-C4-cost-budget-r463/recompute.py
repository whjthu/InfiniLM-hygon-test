"""Finite logical decoder budget, stdlib only. No Runtime, GPU or raw input access.

FMA=2 counts mathematical dot work, including integer dot MACs. This is not
an instruction count, physical roofline, traffic measurement or latency model.
"""
import argparse
import hashlib
import json
from pathlib import Path


def budget(meta, requests):
    c = meta['model']; h = c['hidden_size']; tp = meta['TP']
    layers = c['num_hidden_layers']; heads = c['num_attention_heads']//tp
    d = c['head_dim']; q = c['q_lora_rank']; i = c['moe_intermediate_size']//tp
    k = c['num_experts_per_tok']; ih = c['index_n_heads']//tp
    ratios = c['compress_ratios'][:layers]
    n4 = ratios.count(4); n128 = ratios.count(128)
    tokens = sum(t for past, t in requests)
    assert 1 <= len(requests) <= 4 and all(p >= 0 and t > 0 and p+t <= 32768 for p,t in requests)
    # Current r458 smallt is default0. The existing >=8-row all-selected branch
    # is unreachable for serialC1 / activeC4. Query scoring is therefore retained.
    assert tokens <= 4096
    work = {
        'attention_dense_dot': layers*2*tokens*(q*h+d*h+heads*d*q+1024*4096+4096*1024),
        'routed_I8_dot': layers*6*tokens*k*h*i,
        'shared_I8_dot': layers*6*tokens*c['n_shared_experts']*h*i,
        'router_BF16_dot_including_hash_layers': layers*2*tokens*h*c['n_routed_experts'],
        'HC0_fn_dot': 2*layers*2*tokens*24*(4*h),
        'C4_main_compressor_BF16_dot': n4*4*tokens*(2*d)*h,
        'C128_main_compressor_F32_dot': n128*4*tokens*d*h,
        'C4_index_compressor_BF16_dot': n4*4*tokens*(2*c['index_head_dim'])*h,
        'C4_index_query_I8_dot': n4*2*tokens*ih*c['index_head_dim']*q,
        'C4_index_head_BF16_dot': n4*2*tokens*ih*h,
        'head_collapse_F32_dot_last_position_each_request': 2*len(requests)*4*(4*h),
        'LMhead_BF16_dot_last_position_each_request': 2*len(requests)*(c['vocab_size']//tp)*h,
    }
    qkpv = index_visible = index_rect = 0
    for past,t in requests:
        for j in range(t):
            s = past+j+1; sw = min(128,s)
            keys = ratios.count(0)*sw+n128*(sw+s//128)+n4*(sw+min(512,s//4))
            qkpv += 4*heads*d*keys
            index_visible += n4*2*ih*c['index_head_dim']*(s//4)
        # Scores API uses the request's final width; causal mask may avoid work.
        index_rect += n4*2*ih*c['index_head_dim']*t*max(1,(past+t)//4)
    work['attention_visible_QK_PV_dot'] = qkpv
    work['C4_index_visible_QK_dot'] = index_visible
    # Address union bound is per layer. Selected experts are unique within a row;
    # four rows can share all6 or choose24 distinct. Kernel still indexes by route.
    active = len(requests)
    expert_bytes = (2*i*h+h*i) + (2*i+h)*4
    union_min, union_max = k, min(c['n_routed_experts'],k*active)
    if any(t != 1 for _,t in requests):
        union_max = min(c['n_routed_experts'],k*tokens)
    weight = {
        'routed_expert_bytes_each_rank_each_expert': expert_bytes,
        'routed_unique_experts_per_layer_min_max': [union_min,union_max],
        'routed_unique_parameter_address_bytes_all_layers_min_max': [layers*union_min*expert_bytes,layers*union_max*expert_bytes],
        'routed_route_address_bytes_counting_repeated_routes': layers*k*tokens*expert_bytes,
        'shared_parameters_all_layers': layers*c['n_shared_experts']*expert_bytes,
        'router_BF16_parameters_all_layers': layers*2*h*c['n_routed_experts'],
        'attention_dense_weights_excluding_scales_norms': layers*(4+2+4+8+4)*1024**2,
        'C4_main_and_index_compressor_BF16_weights': n4*2*2*((2*d)+(2*c['index_head_dim']))*h,
        'C128_compressor_original_BF16_weights': n128*2*2*d*h,
        'C4_index_query_and_head_weights_excluding_scales': n4*(ih*c['index_head_dim']*q+2*ih*h),
        'HC_fn_F32_weights': 2*layers*24*(4*h)*4,
        'LMhead_BF16_parameters': (c['vocab_size']//tp)*h*2,
        'head_collapse_F32_fn_parameters': 4*(4*h)*4,
    }
    # Current unit20 explicitly casts both immutable C128 BF16 projection weights
    # to F32 in each prepare. These are logical cast source/output sizes, not HBM.
    cast_source = n128*2*d*h*2
    cast_output = n128*2*d*h*4
    comm = {'attention_allreduce_BF16_API_bytes_per_rank': layers*tokens*h*2,
            'MoE_allreduce_F32_API_bytes_per_rank': layers*tokens*h*4,
            'C4_index_allreduce_F32_API_bytes_per_rank': n4*sum(t*max(1,(p+t)//4)*4 for p,t in requests),
            'head_allgather_local_send_BF16_API_bytes_per_rank': len(requests)*(c['vocab_size']//tp)*2,
            'head_allgather_received_output_BF16_bytes_per_rank': len(requests)*c['vocab_size']*2,
            'attention_and_MoE_allreduce_calls_per_rank': 2*layers,
            'C4_index_allreduce_calls_per_rank': n4*len(requests)}
    subtotal = sum(work.values())
    known_weight_min = sum(v for key,v in weight.items() if isinstance(v,int) and key not in ['routed_expert_bytes_each_rank_each_expert','routed_route_address_bytes_counting_repeated_routes'])+weight['routed_unique_parameter_address_bytes_all_layers_min_max'][0]
    return {'requests_past_tokens':requests,'tokens_this_forward':tokens,'layers_by_ratio':{'SWA':ratios.count(0),'C4':n4,'C128':n128},
            'useful_dot_FLOPs_per_rank':work,'known_dot_subtotal_per_rank':subtotal,
            'index_score_rectangular_dot_if_all_lanes_compute':index_rect,
            'weights_logical_bytes_per_rank':weight,'C128_weight_cast_logical_read_write_bytes':[cast_source,cast_output],
            'collective_API_not_network_bytes_per_rank':comm,
            'hypothetical_references_NOT_bounds_or_predictions':{
                'all_known_dots_at_achieved_BF16_dense_rate_s':subtotal/(meta['achieved_reference']['BF16_TFLOPs_s']*1e12),
                'unique_known_weight_address_once_at_achieved_read_rate_s':known_weight_min/(meta['achieved_reference']['read_TB_s']*1e12),
                'C128_weight_cast_read_plus_write_serial_reference_s':cast_source/(1.23711e12)+cast_output/(.57279e12)},
            'excluded_from_dot_subtotal':['norm/Softmax/Sinkhorn/Softplus/Silu/TopK/reduction elementwise work','quant/pack/dequant instructions and scale applications','padding/backend algorithm extra work','cache gather/store/copy and host control/sync/allocator work'],
            'weight_scope':'unique address and repeated route models; excludes norms/scales except expert scales, cache/activations/embedding lookup; neither actual HBM traffic nor physical lower bound'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metadata',type=Path,default=Path(__file__).with_name('PINNED_METADATA.json'));ap.add_argument('--model-config',type=Path);args=ap.parse_args()
    meta=json.loads(args.metadata.read_bytes())
    if args.model_config:
        assert hashlib.sha256(args.model_config.read_bytes()).hexdigest()==meta['model_config_SHA256']
        cfg=json.loads(args.model_config.read_bytes());assert all(cfg[k]==v for k,v in meta['model'].items())
    result={'schema':1,'scope':meta['scope'],'Native_SHA256':meta['Native_SHA256'],'metadata_SHA256':hashlib.sha256(args.metadata.read_bytes()).hexdigest(),'new_jobs':0,'GPU_GO':False,
            'cases':{'decode_C1_past8496':budget(meta,[(8496,1)]),'decode_C4_samepast8496':budget(meta,[(8496,1)]*4),
                     'decode_C4_original_prompt_lengths':budget(meta,[(s,1)for s in meta['r458_prompt_tokens']]),
                     'decode_C1_last_position32768':budget(meta,[(32767,1)]),
                     'prefill_first1024':budget(meta,[(0,1024)]),'prefill_continuation1024':budget(meta,[(1024,1024)])}}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
