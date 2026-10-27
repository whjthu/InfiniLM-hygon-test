"""Prepare private C++ source overlay only. Does not build or touch repository."""
import hashlib,json,difflib
from pathlib import Path
ROOT=Path(__file__).parent
REPO=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群')
BASE=ROOT/'base';SRC=ROOT/'src';BASE.mkdir(exist_ok=True);SRC.mkdir(exist_ok=True)
bundle=json.loads((REPO/'Deep-MTP-audit-source-bundle-r9287.json').read_text(encoding='utf-8-sig'))
record=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def add(name,text,origin,expected=None):
 (BASE/name).write_text(text,encoding='utf-8',newline='\n');(SRC/name).write_text(text,encoding='utf-8',newline='\n');record.append(dict(file=name,origin=str(origin),origin_SHA256=expected,private_base_SHA256=sha(BASE/name)))
for mapping in [bundle['prior_relevant_sources'],bundle['actual']['files']]:
 for path,rec in mapping.items():
  name=Path(path).name
  if rec.get('text') and name in ['deepseek_v4_swa_attention.cpp','deepseek_v4_hca_attention_batch_r784.cpp','deepseek_v4_csa_attention_batch_r784.cpp','deepseek_v4_paged_latent_cache_public.cpp','deepseek_v4_paged_latent_cache.hpp','deepseek_v4_c4_indexer.hpp','deepseek_v4_routed_text_model_r20.cpp','deepseek_v4_routed_text_model_r20.hpp']:
   add(name,rec['text'],path,rec['SHA256'])
for p in ['Deep-owner-device-private-r8918/deepseek_v4_native_routed_decoder_r20.cpp','deepseek_v4_native_routed_decoder_r20.hpp','Deep-C4-smallT-private-r9252/deep_c4_smallt_r9252.cpp']:
 path=REPO/p;add(path.name,path.read_text(encoding='utf-8-sig'),path,sha(path))
old=REPO/'latest-experimental-source-r3696/candidates/deep-attention-FP32-5f91ad7-r3688/incoming'
for name in ['deepseek_v4_temporal_compressor.hpp','deepseek_v4_swa_attention.hpp','deepseek_v4_hca_attention.hpp','deepseek_v4_csa_attention.hpp']:
 path=old/name;add(name,path.read_text(encoding='utf-8-sig'),path,sha(path))
# Fresh readonly ac937 object/MD binding supersedes historical header substrate.
actual=json.loads((ROOT/'actual_binding_readonly_r124.json').read_text())
for origin,rec in actual['files'].items():
 name=Path(origin).name
 assert rec['SHA256'] in [hashlib.sha256(rec['text'].encode()).hexdigest(),hashlib.sha256(rec['text'].replace('\n','\r\n').encode()).hexdigest()],origin
 if rec.get('expected_from_build'):assert rec['SHA256']==rec['expected_from_build'],origin
 record[:]=[x for x in record if x['file']!=name]
 add(name,rec['text'],origin,rec['SHA256'])
def edit(name,old,new):
 p=SRC/name;s=p.read_text();n=s.count(old)
 if n!=1:raise ValueError(f'{name}: expected exactly1 anchor, got{n}: {old[:100]}')
 p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
for name in ['deepseek_v4_paged_latent_cache.hpp','deepseek_v4_c4_indexer.hpp','deepseek_v4_temporal_compressor.hpp','deepseek_v4_swa_attention.hpp','deepseek_v4_hca_attention.hpp','deepseek_v4_csa_attention.hpp']:
 edit(name,'#pragma once','#pragma once\n#include "deep_prefix_cache_types_r124.hpp"')
 # Multiple private sections in C4 header: friends are safe in each class.
 p=SRC/name;p.write_text(p.read_text().replace('private:','private:\n friend class PrefixNativeR124;'),encoding='utf-8',newline='\n')
for name,field in [('deepseek_v4_paged_latent_cache.hpp','data_'),('deepseek_v4_c4_indexer.hpp','raw_')]:
 edit(name,'std::vector<uint8_t>written_;','std::vector<uint8_t>written_;prefix_r124::Slots<prefix_r124::PackedSlot>prefix_overlay_;')
 edit(name,f'const runtime::Tensor&raw()const{{return {field};}}',f'const runtime::Tensor&raw()const{{if(!prefix_overlay_.empty())throw std::logic_error("prefix segmented cache has no contiguous raw snapshot");return {field};}}')
for name in ['deepseek_v4_paged_latent_cache_public.cpp','deep_c4_smallt_r9252.cpp','deepseek_v4_swa_attention.cpp','deepseek_v4_hca_attention_batch_r784.cpp','deepseek_v4_csa_attention_batch_r784.cpp']:
 edit(name,'#include "execution/submit.hpp"','#include "execution/submit.hpp"\n#include "deep_prefix_native_r124.hpp"')
# Read validity consults committed overlay; legacy base flags remain unchanged.
edit('deepseek_v4_paged_latent_cache_public.cpp','(!write&&!written_[size_t(slot)])','(!write&&!written_[size_t(slot)]&&(!prefix_overlay_.find(uint32_t(slot)/256)||!prefix_overlay_.find(uint32_t(slot)/256)->valid[size_t(slot)%256]))')
edit('deep_c4_smallt_r9252.cpp','(!write&&!written_[size_t(slot)])','(!write&&!written_[size_t(slot)]&&!prefix_overlay_.find(uint32_t(slot)))')
edit('deepseek_v4_paged_latent_cache_public.cpp','auto host=validate(r,slots,true);if(!values','if(!prefix_overlay_.empty()){auto p=PrefixNativeR124::cache_store(r,*this,slots,values);p();return;}\n auto host=validate(r,slots,true);if(!values')
edit('deepseek_v4_paged_latent_cache_public.cpp','auto host=validate(r,slots,false);auto out=','if(!prefix_overlay_.empty())return PrefixNativeR124::cache_gather(r,*this,slots);\n auto host=validate(r,slots,false);auto out=')
edit('deepseek_v4_paged_latent_cache_public.cpp','std::fill(written_.begin(),written_.end(),0);','std::fill(written_.begin(),written_.end(),0);prefix_overlay_.clear();')
edit('deep_c4_smallt_r9252.cpp','void PagedIndexerCache::store(rt::Runtime&r,const std::vector<int64_t>&slots,const Packed&packed){validate','void PagedIndexerCache::store(rt::Runtime&r,const std::vector<int64_t>&slots,const Packed&packed){if(!prefix_overlay_.empty()){auto p=PrefixNativeR124::index_store(r,*this,slots,packed);p();return;}validate')
edit('deep_c4_smallt_r9252.cpp','PagedIndexerCache::Packed PagedIndexerCache::gather(rt::Runtime&r,const std::vector<int64_t>&slots)const{validate','PagedIndexerCache::Packed PagedIndexerCache::gather(rt::Runtime&r,const std::vector<int64_t>&slots)const{if(!prefix_overlay_.empty())return PrefixNativeR124::index_gather(r,*this,slots);validate')
edit('deep_c4_smallt_r9252.cpp','std::fill(written_.begin(),written_.end(),0);','std::fill(written_.begin(),written_.end(),0);prefix_overlay_.clear();')
# Actual attention forward retains existing math and full indexer path. Only
# authoritative commit statement is replaced under explicit thread-local Scope.
rows='std::vector<prefix_r124::Row>prefix_rows;for(auto q:requests)prefix_rows.push_back({q.row,q.past,q.tokens});'
for name,old,new in [
 ('deepseek_v4_swa_attention.cpp','cache_->store(r,device_ids(commit_slots),commit);for(auto request:requests)lengths_[request.row]=request.past+request.tokens;',f'if(prefix_r124::active){{{rows}PrefixNativeR124::stage_swa(r,*this,prep.kv,prefix_rows);}}else{{cache_->store(r,device_ids(commit_slots),commit);for(auto request:requests)lengths_[request.row]=request.past+request.tokens;}}'),
 ('deepseek_v4_hca_attention_batch_r784.cpp','cache_->store(r,device_ids(commit_slots),commit);compressor_->commit(r,pending);for(auto request:requests)lengths_[request.row]=request.past+request.tokens;',f'if(prefix_r124::active){{{rows}PrefixNativeR124::stage_hca(r,*this,prep.kv,std::move(pending),prefix_rows);}}else{{cache_->store(r,device_ids(commit_slots),commit);compressor_->commit(r,pending);for(auto request:requests)lengths_[request.row]=request.past+request.tokens;}}'),
 ('deepseek_v4_csa_attention_batch_r784.cpp','cache_->store(r,device_ids(commit_slots),commit);compressor_->commit(r,pending);indexer_->commit(r,index_pending);last_selection_=index_pending.indices;for(auto request:requests)lengths_[request.row]=request.past+request.tokens;',f'if(prefix_r124::active){{{rows}PrefixNativeR124::stage_csa(r,*this,prep.kv,std::move(pending),std::move(index_pending),prefix_rows);}}else{{cache_->store(r,device_ids(commit_slots),commit);compressor_->commit(r,pending);indexer_->commit(r,index_pending);last_selection_=index_pending.indices;for(auto request:requests)lengths_[request.row]=request.past+request.tokens;}}')]:edit(name,old,new)
edit('deepseek_v4_native_routed_decoder_r20.hpp','#pragma once','#pragma once\n#include "deep_prefix_transaction_r124.hpp"')
edit('deepseek_v4_native_routed_decoder_r20.hpp','runtime::Runtime*owner_;size_t rows_,maximum_,vocab_;','runtime::Runtime*owner_;size_t prefix_layer_;size_t rows_,maximum_,vocab_;')
edit('deepseek_v4_native_routed_decoder_r20.cpp',':owner_(&r),rows_(rows)',':owner_(&r),prefix_layer_(layer),rows_(rows)')
edit('deepseek_v4_native_routed_decoder_r20.cpp','for(auto q:requests)lengths_[q.row]=q.past+q.tokens;owner_context.beforeDrain();','prefix_r124::lengths(lengths_,requests);if(prefix_r124::active)prefix_r124::active->decoder_done(prefix_layer_);owner_context.beforeDrain();')
edit('deepseek_v4_routed_text_model_r20.hpp','#pragma once','#pragma once\n#include "deep_prefix_transaction_r124.hpp"')
p=SRC/'deepseek_v4_routed_text_model_r20.hpp';s=p.read_text();needle='class TextModelR20';assert needle in s
# Insert into TextModel only, never TextHead.
left,right=s.split(needle,1)
right=right.replace('public:','public:\n using PrefixHandle=std::shared_ptr<prefix_r124::Transaction>;\n PrefixHandle begin_prefix(runtime::Runtime&,const std::vector<Request>&);\n Output forward_prefix(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<Request>&,const PrefixHandle&);\n void seal_prefix(runtime::Runtime&,const PrefixHandle&,const prefix_r124::Counts&);\n void publish_prefix(runtime::Runtime&,const PrefixHandle&);\n void cancel_prefix(runtime::Runtime&,const PrefixHandle&);',1)
# Request/Output may be declared after public; relocate new methods to after
# existing forward declaration by matching header actual layout below.
method_start=right.index(' using PrefixHandle=');method_end=right.index(' void cancel_prefix',method_start);method_end=right.index(';',method_end)+1
methods=right[method_start:method_end];right=right[:method_start]+right[method_end:]
reset='void reset(runtime::Runtime&);';assert reset in right;right=right.replace(reset,reset+'\n'+methods,1)
right=right.replace('private:','private:\n std::shared_ptr<prefix_r124::Epoch>prefix_epoch_=std::make_shared<prefix_r124::Epoch>();PrefixHandle prefix_pending_;',1)
p.write_text(left+needle+right,encoding='utf-8',newline='\n')
edit('deepseek_v4_routed_text_model_r20.cpp','owner(r,owner_);\n if(!ready_||poisoned_||!ids','owner(r,owner_);\n if(prefix_pending_&&prefix_r124::active!=prefix_pending_.get())throw std::logic_error("prefix pending: ordinary forward forbidden");\n prefix_r124::validate_rows(requests);\n if(!ready_||poisoned_||!ids')
edit('deepseek_v4_routed_text_model_r20.cpp','auto result=head_->forward(r,chosen);for(auto q:requests)lengths_[q.row]=q.past+q.tokens;','auto result=head_->forward(r,chosen);prefix_r124::lengths(lengths_,requests);')
edit('deepseek_v4_routed_text_model_r20.cpp','void TextModelR20::reset(rt::Runtime&r){owner(r,owner_);','void TextModelR20::reset(rt::Runtime&r){owner(r,owner_);if(prefix_pending_){prefix_pending_->cancel();prefix_pending_.reset();}++prefix_epoch_->value;')
extra=r'''
TextModelR20::PrefixHandle TextModelR20::begin_prefix(rt::Runtime&r,const std::vector<Request>&q){owner(r,owner_);if(!ready_||poisoned_||prefix_pending_)throw std::logic_error("prefix begin model busy/unready");std::vector<prefix_r124::Row>rows;for(auto a:q){if(a.row>=rows_||a.past!=lengths_[a.row]||a.tokens!=2||a.past>capacity_||2>capacity_-a.past)throw std::invalid_argument("prefix begin row/past/capacity");rows.push_back({a.row,a.past,a.tokens});}prefix_pending_=std::make_shared<prefix_r124::Transaction>(this,prefix_epoch_,std::move(rows));return prefix_pending_;}
TextModelR20::Output TextModelR20::forward_prefix(rt::Runtime&r,const rt::Tensor&ids,const rt::Tensor&pos,const std::vector<Request>&q,const PrefixHandle&h){owner(r,owner_);if(!h||h!=prefix_pending_)throw std::invalid_argument("prefix foreign handle");h->check_owner(this);prefix_r124::Scope scope(h.get());try{auto out=forward(r,ids,pos,q,true);r.syncStream();h->forward_done();return out;}catch(...){h->cancel();prefix_pending_.reset();throw;}}
void TextModelR20::seal_prefix(rt::Runtime&r,const PrefixHandle&h,const prefix_r124::Counts&counts){owner(r,owner_);if(!h||h!=prefix_pending_)throw std::invalid_argument("prefix foreign seal");h->check_owner(this);h->seal(counts);r.syncStream();}
void TextModelR20::publish_prefix(rt::Runtime&r,const PrefixHandle&h){owner(r,owner_);if(!h||h!=prefix_pending_)throw std::invalid_argument("prefix foreign publish");h->check_owner(this);h->publish();prefix_pending_.reset();}
void TextModelR20::cancel_prefix(rt::Runtime&r,const PrefixHandle&h){owner(r,owner_);if(!h||h!=prefix_pending_)throw std::invalid_argument("prefix foreign cancel");h->check_owner(this);r.syncStream();h->cancel();prefix_pending_.reset();}
'''
p=SRC/'deepseek_v4_routed_text_model_r20.cpp';s=p.read_text();assert s.endswith('}\n');p.write_text(s[:-2]+extra+'}\n',encoding='utf-8',newline='\n')
patch=[]
for p in sorted(SRC.iterdir()):
 patch.extend(difflib.unified_diff((BASE/p.name).read_text().splitlines(True),p.read_text().splitlines(True),fromfile='a/'+p.name,tofile='b/'+p.name))
(ROOT/'native_prefix_r124.patch').write_text(''.join(patch),encoding='utf-8')
(ROOT/'source_bindings_r124.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PRIVATE_SOURCES_PREPARED_UNCOMPILED',files=len(record),patch_SHA256=sha(ROOT/'native_prefix_r124.patch'))))
