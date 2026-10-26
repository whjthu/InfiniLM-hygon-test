import json,pathlib,math,hashlib
p=pathlib.Path(__file__).parent;proof=json.loads((p/'source-proof-r9201.json').read_text());cfg=proof['config'];lin=cfg['linear_attn_config'];allocator=(p/'kimi_compact_allocate_r4974.cpp').read_text();kv=proof['kv_cpp']['text']
assert 'const size_t heads = linear_num_heads / tp_size;'in allocator
assert '{pool_size, 3 * heads * linear_head_dim, conv_kernel - 1}'in allocator
assert '{pool_size, heads, linear_head_dim, linear_head_dim}'in allocator
assert 'model_config->get_dtype()'in allocator and 'config->max_batch_size()+1'in allocator
assert '2, num_blocks_per_layer, block_size, num_rank_k_heads, k_dim'in kv
assert cfg['dtype']=='bfloat16'and cfg['qk_nope_head_dim']+cfg['qk_rope_head_dim']==192
assert lin['num_heads']==96 and cfg['num_attention_heads']==96 and lin['head_dim']==128 and lin['short_conv_kernel_size']==4
G=1024**3;M=1024**2;checks=0
def shape(C,pp,stage,compressed_pages=False):
 global checks
 assert C in [16,32] and pp in [2,4] and 0<=stage<pp
 ranges={2:[(0,46),(46,93)],4:[(0,24),(24,47),(47,71),(71,93)]};a,b=ranges[pp][stage];k=sum(a<=x-1<b for x in lin['kda_layers']);mla=b-a-k;physical=2*C;pool=physical+1;pages=4*((C if compressed_pages else physical)+1);h=96//16;D=128
 conv=pool*3*h*D*3*2;recurrent=pool*h*D*D*2;kv=2*pages*256*h*192*2
 checks+=1;return {'PP':pp,'stage':stage,'range':[a,b],'C':C,'physical_rows':physical,'pool_rows':pool,'pages':pages,'KDA_layers':k,'MLA_layers':mla,'conv_bytes_per_KDA':conv,'SSM_bytes_per_KDA':recurrent,'MLA_generic_KV_bytes_per_layer':kv,'payload_bytes_per_owner':k*(conv+recurrent)+mla*kv}
rows=[]
for pp in [2,4]:
 for stage in range(pp):
  a=shape(16,pp,stage);b=shape(32,pp,stage);diff=b['payload_bytes_per_owner']-a['payload_bytes_per_owner'];assert diff==a['KDA_layers']*6733824+a['MLA_layers']*150994944
  rows.append({'baseline':a,'cap32_unchanged_allocator':b,'delta_bytes':diff,'delta_GiB':diff/G,'suggested_logical_KV_pages_cap32':shape(32,pp,stage,True)})
# Plan transaction: each independent request active/shadow unique; all partial lengths and EOS per request.
for C,W in [(16,4),(32,2)]:
 ids=list(range(1,C+1));reads=ids;writes=[i+C for i in ids];assert len(set(reads+writes))==2*C and max(writes)<2*C+1 and C*W==64
 for i in ids:
  active=i;shadow=i+C;past=512;accepted=(i-1)%W;retained=accepted+1;newpast=past+retained;assert newpast<=1024
  # MLA indexes logical i, never recurrent shadow i+C; maxblock in proposed KV rows fits.
  kv_end=1+(i-1)*4+3;assert kv_end<4*(C+1);checks+=1
for C,W in [(16,8),(32,4),(32,8)]:
 assert C*W>64;checks+=1
margin=5496635392-4*G;tail=rows[1]['delta_bytes'];assert tail>margin
out={'CPU_only':True,'GPU_submitted':False,'CPU_native_library_loaded':False,'formula_checks':checks,'formula_math_integer_exact':True,'cache_payload_stages':rows,'current_C16_preload_min_bytes':5496635392,'current_C16_floor_bytes':4*G,'tail_unchanged_C32_delta_bytes':tail,'conditional_C32_preload_min_if_only_payload_changes_GiB':(5496635392-tail)/G,'unchanged_PP2cap32_constructor_GO':False,'reason':'actual cache payload increase alone exceeds observed preload headroom; allocator/forward peaks may cost more. This is not new actual GPU capacity.','PP4_cap32':'prior actual PP4C32 capacity+Model qualified; use independent shard-only PP4 runtime if needed, preserving allowner4GiB gate','proposed_logical_pages_C32_extra_vs_C16_tail_GiB':(rows[1]['suggested_logical_KV_pages_cap32']['payload_bytes_per_owner']-rows[1]['baseline']['payload_bytes_per_owner'])/G,'state_widths_ready_CPU_geometry_only':[[16,4],[32,2]],'unsupported_flatten_above64_rejected':True}
(p/'CPU-contract-r9201.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks':checks,'tail_delta_GiB':tail/G,'conditional_free_GiB':out['conditional_C32_preload_min_if_only_payload_changes_GiB'],'unchanged_PP2cap32_GO':False,'proposed_extraGiB':out['proposed_logical_pages_C32_extra_vs_C16_tail_GiB']}))
