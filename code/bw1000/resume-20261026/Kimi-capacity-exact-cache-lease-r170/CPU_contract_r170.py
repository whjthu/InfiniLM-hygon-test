"""No Runtime creation. Freeze cache zeroing and owner/lease source obligations."""
from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent;S=Path(sys.argv[1])if len(sys.argv)>1 else Path('/refactor/incoming/Kimi-TP32-I96-template-component-r163')
proof=json.loads((D/'source-proof-bodies-r170.json').read_bytes());pins=json.loads((D/'parent-pins-r170.json').read_bytes());old=proof['MoE167_source'];moe=(D/'kimi_moe_capacity_r170.cpp').read_text();lease=(D/'I96_scratch_lease_r170.hpp').read_text();cache=(D/'kimi_cache_exact_r170.cpp').read_text()
base='tiled_scratch_r1861_=infinilm::runtime::Tensor::empty(runtime,{r163::bytes/2},infinilm::runtime::DataType::kBFloat16);'
expected=old.replace('#include "contract_I96_r163.hpp"','#include "contract_I96_r163.hpp"\n#include "I96_scratch_lease_r170.hpp"').replace(base,'tiled_scratch_r1861_=r170_scratch::shared_workspace(runtime);')
point='        auto*p=reinterpret_cast<unsigned char*>(tiled_scratch_r1861_->data());auto&c=i96_counts_r163.at(runtime.device().index());';expected=expected.replace(point,'        r170_scratch::Lease workspace_lease_r170(runtime,tiled_scratch_r1861_);\n'+point);assert moe==expected
assert moe.count('Tensor::empty_parameter(')==old.count('Tensor::empty_parameter(')==4
assert moe.count('(rank_info.tp_size > 1 && !rank_info.comm)')==old.count('(rank_info.tp_size > 1 && !rank_info.comm)')==3
for obligation in ['std::weak_ptr<rt::Runtime>','std::weak_ptr<rt::TensorImpl>','std::owner_less<WeakOwner>','next_generation++','entry_->busy=true','it->second->busy','require_owner(r)','allocation_owner()','reinterpret_cast<uintptr_t>(r.stream())','Lease(const Lease&)=delete','owner_->syncStream();','entry_->busy=false','std::terminate();']:
 assert obligation in lease,obligation
assert lease.index('owner_->syncStream();')<lease.index('entry_->busy=false')
assert 'thread_local'not in lease and 'map<rt::Runtime*'not in lease
provider=(S/'provider_I96_r163.cu').read_bytes();assert hashlib.sha256(provider).hexdigest()==pins['provider_I96_SHA256'];p=provider.decode();p=p[p.index('extern "C" int modern_kimi_grouped_I96_r163'):]
assert p.count('<<<')==4 and p.count(',st>>>')==4 and ',0,st>>>'in p
assert 'auto st=static_cast<cudaStream_t>(stream)'in p and p.count('cudaMemsetAsync')==2 and 'cudaStreamCreate'not in p
factory=cache.split('infinilm::runtime::Tensor create_exact_mla_cache(',1)[1].split('\n} // namespace r170_cache',1)[0]
factory='infinilm::runtime::Tensor create_exact_mla_cache('+factory
restored=factory.replace('create_exact_mla_cache(','create_layer_kv_cache(').replace('const cache::PagedKVCacheConfig &','const PagedKVCacheConfig &').replace('    cache::validate_cache_runtime(runtime, rank);','    validate_cache_runtime(runtime, rank);').replace('Tensor::empty_parameter(runtime, kv_shape, dtype)','Tensor::empty(runtime, kv_shape, dtype)')
assert restored==proof['kv_original_factory'],'zero-fill, shape, strides and unwind must stay identical'
assert cache.count('Tensor::empty_parameter(')==1 and 'runtime.setDeviceMemoryAsync(kv_cache->data(), 0, kv_cache->nbytes(), runtime.stream())'in factory
assert 'config->num_blocks()==68 && config->block_size()==256' in cache and 'tp_size==32 && rank_info.pp_size==1 && num_layers==93' in cache
# Outside the one MLA factory dispatch and helper preface, retain KDA allocation and publication.
for tail in ['return {std::move(kv), std::move(conv), std::move(recurrent)};','runtime.setDeviceMemory(conv[layer_idx]->data(), 0, conv[layer_idx]->nbytes());','runtime.setDeviceMemory(recurrent[layer_idx]->data(), 0, recurrent[layer_idx]->nbytes());']:
 assert tail in cache and tail in proof['cache_base_source']
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'cache_exact_source_equivalent_zero_fill':True,'KDA_cache_and_cache_vectors_unchanged':True,'private_cache_scope':'TP32PP1/93layers/96heads/H192/pages68/block256/BF16 only','scratch_shared_owner_controlblock_generation':True,'weak_registry_no_strong_TLS_cycle':True,'actual_I96_provider_all_consumers_same_stream':True,'Lease_drains_before_unlock_reuse':True,'crossowner_crossstream_reentry_rejected':True,'class_layout_and_GPU_math_unchanged':True,'additional_per_I96_call_stream_fence':True,'performance':'UNKNOWN; synchronization changes scheduling','actual_lifecycle_GPU_validation':'NOT_RUN','fullModel_GO':False,'fullModel_peak':'UNKNOWN_STOP'}))
