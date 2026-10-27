#pragma once
#include "deep_prefix_transaction_r128.hpp"
#include "deepseek_v4_swa_attention.hpp"
#include "deepseek_v4_hca_attention.hpp"
#include "deepseek_v4_csa_attention.hpp"
namespace infinilm::models::deepseek_v4 {
class TextModelR20;
class PrefixNativeR128 {
public:
 static nlohmann::json inspect_model_r129(runtime::Runtime&,TextModelR20&,const std::vector<size_t>&);
 using Tensor=runtime::Tensor;using Rows=std::vector<prefix_r128::Row>;using Counts=prefix_r128::Counts;using Publish=prefix_r128::Publish;
 static Publish cache_store(runtime::Runtime&,PagedLatentCache&,const Tensor&,const Tensor&);
 static Publish append_store(runtime::Runtime&,PagedLatentCache&,const Tensor&,const Tensor&);
 static Tensor cache_gather(runtime::Runtime&,const PagedLatentCache&,const Tensor&);
 static Publish index_store(runtime::Runtime&,PagedIndexerCache&,const std::vector<int64_t>&,const PagedIndexerCache::Packed&);
 static PagedIndexerCache::Packed index_gather(runtime::Runtime&,const PagedIndexerCache&,const std::vector<int64_t>&);
 static Publish compressor(runtime::Runtime&,TemporalCompressor&,const TemporalCompressor::Batch&,const Rows&,const Counts&);
 static Publish indexer(runtime::Runtime&,C4Indexer&,const C4Indexer::Pending&,const Rows&,const Counts&);
 static void stage_swa(runtime::Runtime&,SwaAttention&,Tensor,const Rows&);
 static void stage_hca(runtime::Runtime&,HcaAttention&,Tensor,TemporalCompressor::Batch,const Rows&);
 static void stage_csa(runtime::Runtime&,CsaAttention&,Tensor,TemporalCompressor::Batch,C4Indexer::Pending,const Rows&);
};
}
