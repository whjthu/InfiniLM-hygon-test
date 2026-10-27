#include "deep_prefix_native_r128.hpp"
#include "execution/submit.hpp"
#include <numeric>
#include <algorithm>
#include <map>
#include <string>
#include <type_traits>
namespace infinilm::models::deepseek_v4 {
namespace rt=runtime;namespace ex=infinilm::execution;using P=prefix_r128::Publish;using Rows=PrefixNativeR128::Rows;using Counts=PrefixNativeR128::Counts;
namespace {
rt::Tensor ids(rt::Runtime&r,const std::vector<int64_t>&v){auto t=rt::Tensor::empty(r,{v.size()},rt::DataType::kInt64);r.memcpyH2D(t->data(),v.data(),v.size()*8,false);r.syncStream();return t;}
rt::Tensor take(rt::Runtime&r,const rt::Tensor&x,const std::vector<int64_t>&v){if(v.empty())return {};auto shape=x->shape();shape[0]=v.size();auto y=rt::Tensor::empty(r,shape,x->dtype());auto index=ids(r,v);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),x,index,int64_t{0},y);r.syncStream();return y;}
void status(rt::Runtime&r,const rt::Tensor&flag){int n=0;r.memcpyD2H(&n,flag->data(),4);r.syncStream();if(n)throw std::invalid_argument("prefix packed cache validation status="+std::to_string(n));}
P combine(std::vector<P>p){return [p=std::move(p)]()noexcept{for(auto&f:p){try{f();}catch(...){std::terminate();}}};}
std::vector<int64_t>accepted(const Rows&q,const Counts&c){std::vector<int64_t>out;size_t base=0;for(size_t j=0;j<q.size();++j){for(size_t t=0;t<c[j];++t)out.push_back(int64_t(base+t));base+=q[j].tokens;}return out;}
std::vector<int64_t>boundaries(const std::vector<int64_t>&b,const Rows&q,const Counts&c){auto a=accepted(q,c);std::vector<int64_t>out;for(size_t j=0;j<b.size();++j)if(std::binary_search(a.begin(),a.end(),b[j]))out.push_back(int64_t(j));return out;}
std::shared_ptr<std::vector<size_t>>next_lengths(const std::vector<size_t>&old,const Rows&q,const Counts&c){auto n=std::make_shared<std::vector<size_t>>(old);for(size_t j=0;j<q.size();++j)(*n)[q[j].row]=q[j].past+c[j];return n;}
P ring(rt::Runtime&r,PagedLatentCache&cache,const rt::Tensor&kv,const Rows&q,const Counts&c){auto a=accepted(q,c);std::vector<int64_t>slots;for(size_t j=0;j<q.size();++j)for(size_t t=0;t<c[j];++t)slots.push_back(int64_t(q[j].row*128+(q[j].past+t)%128));return PrefixNativeR128::cache_store(r,cache,ids(r,slots),take(r,kv,a));}
}
P PrefixNativeR128::cache_store(rt::Runtime&r,PagedLatentCache&cache,const Tensor&slot_tensor,const Tensor&values){
 auto slots=cache.validate(r,slot_tensor,true);if(slots.empty())return []{};
 if(cache.block_!=256)throw std::logic_error("prefix ring COW requires block256");
 // Copy only touched physical ring pages, preserving exact packed bytes and
 // padding. Persistent trie is keyed by PAGE, not history/token slot.
 std::map<size_t,std::vector<size_t>>pages;for(size_t j=0;j<slots.size();++j)pages[size_t(slots[j])/256].push_back(j);
 auto next=std::make_shared<decltype(cache.prefix_overlay_)>(cache.prefix_overlay_);
 for(auto&entry:pages){auto old=cache.prefix_overlay_.find(uint32_t(entry.first));PagedLatentCache page(r,1,256);auto base=old?old->raw:cache.data_->narrow({{0,entry.first,1}});page.data_->copy_from(r,base);prefix_r128::PackedSlot leaf{page.data_,0,256};if(old)leaf.valid=old->valid;else for(size_t i=0;i<256&&entry.first*256+i<cache.written_.size();++i)leaf.valid[i]=cache.written_[entry.first*256+i];std::vector<int64_t>source,local;for(auto j:entry.second){source.push_back(int64_t(j));local.push_back(slots[j]%256);leaf.valid[size_t(slots[j])%256]=1;}page.store(r,ids(r,local),take(r,values,source));*next=next->with(uint32_t(entry.first),std::move(leaf));}
 r.syncStream();return [&cache,next]()noexcept{cache.prefix_overlay_.swap(*next);};
}
P PrefixNativeR128::append_store(rt::Runtime&r,PagedLatentCache&cache,const Tensor&slot_tensor,const Tensor&values){
 auto slots=cache.validate(r,slot_tensor,true);for(auto slot:slots)if(cache.written_.at(size_t(slot)))throw std::invalid_argument("prefix compressed append overlaps published slot");
 // Append-only compressed slots are outside every published valid prefix.
 // Existing op can prepare their bytes after whole forward succeeds. Delay
 // written flags until publish; cancellation leaves unreachable scratch bytes.
 auto flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);ex::submit<infini::ops::DeepseekV4PagedCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4PagedCache::DefaultConfig(r.device().type()),cache.data_,slot_tensor,values,flag,cache.block_,size_t{0});status(r,flag);
 return [&cache,slots=std::move(slots)]()noexcept{for(auto slot:slots)cache.written_[size_t(slot)]=1;};
}
rt::Tensor PrefixNativeR128::cache_gather(rt::Runtime&r,const PagedLatentCache&cache,const Tensor&slot_tensor){
 auto slots=cache.validate(r,slot_tensor,false);auto out=rt::Tensor::empty(r,{slots.size(),512},rt::DataType::kBFloat16);auto flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);
 for(size_t j=0;j<slots.size();){auto leaf=cache.prefix_overlay_.find(uint32_t(slots[j]/256));auto raw=leaf?leaf->raw:cache.data_;std::vector<int64_t>local;size_t end=j;for(;end<slots.size();++end){auto current=cache.prefix_overlay_.find(uint32_t(slots[end]/256));auto backing=current?current->raw:cache.data_;if(backing->data()!=raw->data())break;local.push_back(current?slots[end]%256:slots[end]);}auto index=ids(r,local);auto dst=out->narrow({{0,j,end-j}});ex::submit<infini::ops::DeepseekV4PagedCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4PagedCache::DefaultConfig(r.device().type()),raw,index,dst,flag,cache.block_,size_t{1});status(r,flag);j=end;}
 return out;
}
P PrefixNativeR128::index_store(rt::Runtime&r,PagedIndexerCache&cache,const std::vector<int64_t>&slots,const PagedIndexerCache::Packed&packed){
 cache.validate(r,slots,true);for(auto slot:slots)if(cache.written_.at(size_t(slot)))throw std::invalid_argument("prefix index append overlaps published slot");auto flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);auto index=ids(r,slots);ex::submit<infini::ops::DeepseekV4IndexerCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4IndexerCache::DefaultConfig(r.device().type()),cache.raw_,index,packed.values,packed.scales,flag,size_t{0});status(r,flag);return [&cache,slots]()noexcept{for(auto slot:slots)cache.written_[size_t(slot)]=1;};
}
PagedIndexerCache::Packed PrefixNativeR128::index_gather(rt::Runtime&r,const PagedIndexerCache&cache,const std::vector<int64_t>&slots){
 cache.validate(r,slots,false);auto values=rt::Tensor::empty(r,{slots.size(),128},rt::DataType::kUInt8),scales=rt::Tensor::empty(r,{slots.size(),1},rt::DataType::kFloat32),flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);
 for(size_t j=0;j<slots.size();++j){auto leaf=cache.prefix_overlay_.find(uint32_t(slots[j]));auto raw=leaf?leaf->raw:cache.raw_;int64_t local=leaf?int64_t(leaf->slot):slots[j];ex::submit<infini::ops::DeepseekV4IndexerCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4IndexerCache::DefaultConfig(r.device().type()),raw,ids(r,{local}),values->narrow({{0,j,1}}),scales->narrow({{0,j,1}}),flag,size_t{1});status(r,flag);}
 return {values,scales};
}
P PrefixNativeR128::compressor(rt::Runtime&r,TemporalCompressor&self,const TemporalCompressor::Batch&batch,const Rows&q,const Counts&c){
 self.check_owner(r);if(batch.owner!=&self||batch.revision!=self.revision_||!batch.next_state)throw std::invalid_argument("prefix compressor stale/foreign");
 auto next=std::make_shared<rt::Tensor>();bool all=true;for(size_t j=0;j<c.size();++j)all=all&&c[j]==q[j].tokens;
 if(all)*next=batch.next_state;else{
  *next=rt::Tensor::empty(r,self.state_->shape(),self.state_->dtype());(*next)->copy_from(r,self.state_);size_t base=0;
  // Existing kernel persist writes exactly the projected score at this slot.
  // Reuse score bytes; no GEMM/pooling/RoPE/history replay, no new GPU kernel.
  for(size_t j=0;j<q.size();++j){for(size_t t=0;t<c[j];++t){size_t p=q[j].past+t,slot=self.ratio_==4?(q[j].row*2+(p/4)%2)*4+p%4:q[j].row*128+p%128;(*next)->narrow({{0,slot,1}})->copy_from(r,batch.score->narrow({{0,base+t,1}}));}base+=q[j].tokens;}
 }
 std::vector<P>plans;auto kept=boundaries(batch.boundaries,q,c);if(self.cache_&&!kept.empty()){std::vector<int64_t>slots;for(auto k:kept)slots.push_back(batch.slots.at(size_t(k)));plans.push_back(append_store(r,*self.cache_,ids(r,slots),take(r,batch.rotated,kept)));}
 auto lengths=next_lengths(self.lengths_,q,c);plans.push_back([&self,next,lengths]()noexcept{self.state_=*next;self.lengths_.swap(*lengths);++self.revision_;});r.syncStream();return combine(std::move(plans));
}
P PrefixNativeR128::indexer(rt::Runtime&r,C4Indexer&self,const C4Indexer::Pending&pending,const Rows&q,const Counts&c){
 self.check_owner(r);if(pending.owner!=&self||pending.revision!=self.revision_||self.poisoned_)throw std::invalid_argument("prefix indexer stale/foreign");std::vector<P>plans;
 auto kept=boundaries(pending.compression.boundaries,q,c);if(!kept.empty()){std::vector<int64_t>slots;for(auto k:kept)slots.push_back(pending.slots.at(size_t(k)));PagedIndexerCache::Packed packed{take(r,pending.new_keys.values,kept),take(r,pending.new_keys.scales,kept)};plans.push_back(index_store(r,*self.cache_,slots,packed));}
 plans.push_back(compressor(r,*self.compressor_,pending.compression,q,c));auto lengths=next_lengths(self.lengths_,q,c);plans.push_back([&self,lengths]()noexcept{self.lengths_.swap(*lengths);++self.revision_;});return combine(std::move(plans));
}
void PrefixNativeR128::stage_swa(rt::Runtime&r,SwaAttention&self,Tensor kv,const Rows&q){
 prefix_r128::validate_rows(q);prefix_r128::active->stage([&r,&self,kv,q](const Counts&c){std::vector<P>p;p.push_back(ring(r,*self.cache_,kv,q,c));auto lengths=next_lengths(self.lengths_,q,c);p.push_back([&self,lengths]()noexcept{self.lengths_.swap(*lengths);});return combine(std::move(p));});
}
void PrefixNativeR128::stage_hca(rt::Runtime&r,HcaAttention&self,Tensor kv,TemporalCompressor::Batch batch,const Rows&q){
 prefix_r128::validate_rows(q);prefix_r128::active->stage([&r,&self,kv,batch,q](const Counts&c){std::vector<P>p;p.push_back(ring(r,*self.cache_,kv,q,c));p.push_back(compressor(r,*self.compressor_,batch,q,c));auto lengths=next_lengths(self.lengths_,q,c);p.push_back([&self,lengths]()noexcept{self.lengths_.swap(*lengths);});return combine(std::move(p));});
}
void PrefixNativeR128::stage_csa(rt::Runtime&r,CsaAttention&self,Tensor kv,TemporalCompressor::Batch batch,C4Indexer::Pending pending,const Rows&q){
 prefix_r128::validate_rows(q);prefix_r128::active->stage([&r,&self,kv,batch,pending,q](const Counts&c){std::vector<P>p;p.push_back(ring(r,*self.cache_,kv,q,c));p.push_back(compressor(r,*self.compressor_,batch,q,c));p.push_back(indexer(r,*self.indexer_,pending,q,c));auto lengths=next_lengths(self.lengths_,q,c);auto selection=std::make_shared<std::vector<Tensor>>();for(size_t j=0;j<q.size();++j)selection->push_back(pending.indices.at(j)->narrow({{0,0,c[j]}}));p.push_back([&self,lengths,selection]()noexcept{self.lengths_.swap(*lengths);self.last_selection_.swap(*selection);});return combine(std::move(p));});
}
}
