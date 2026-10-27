// Included inside the actual PrefixNativeR128 implementation TU.
nlohmann::json PrefixNativeR128::inspect_model_r129(rt::Runtime&r,TextModelR20&m,const std::vector<size_t>&rows){
 r.assertOwnerThread();if(m.owner_!=&r||rows.empty())throw std::invalid_argument("prefix inspect foreign owner/rows");r.syncStream();
 nlohmann::json out={{"configured_rows",m.rows_},{"model_poisoned",m.poisoned_},{"model_lengths",m.lengths_},{"epoch",m.prefix_epoch_->value},{"prefix_pending",bool(m.prefix_pending_)},{"layers",nlohmann::json::array()}};
 auto ring=[&](PagedLatentCache&cache,size_t row,size_t length){size_t valid=0;bool exact=true;for(size_t t=0;t<128;++t){size_t slot=row*128+t;auto leaf=cache.prefix_overlay_.find(uint32_t(slot/256));bool bit=leaf?bool(leaf->valid[slot%256]):bool(cache.written_.at(slot));valid+=bit;bool expected=length>=128||t<length;exact=exact&&(bit==expected);}return nlohmann::json{{"valid_ring_slots",valid},{"ring_validity_matches_committed_length",exact},{"future_logical_tokens_guarded_by_lengths",true}};};
 auto compressed=[&](TemporalCompressor&c,size_t row,size_t length){size_t valid=0;bool exact=true;for(size_t t=0;t<c.capacity_;++t){size_t slot=row*c.capacity_+t;bool bit=bool(c.cache_->written_.at(slot));valid+=bit;exact=exact&&(bit==(t<length/c.ratio_));}return nlohmann::json{{"length",c.lengths_.at(row)},{"ratio",c.ratio_},{"valid_compressed_slots",valid},{"validity_matches_committed_groups",exact},{"revision",c.revision_}};};
 for(size_t l=0;l<m.layers_.size();++l){auto&d=*m.layers_[l];nlohmann::json layer={{"layer",l},{"ratio",d.ratio_},{"decoder_poisoned",d.poisoned_},{"rows",nlohmann::json::array()}};
  for(size_t row:rows){if(row>=m.rows_)throw std::invalid_argument("prefix inspect row range");size_t length=m.lengths_[row];nlohmann::json item={{"row",row},{"model_length",length},{"decoder_length",d.lengths_.at(row)}};
   if(d.swa_){auto&a=*d.swa_;item["attention_length"]=a.lengths_.at(row);item["ring"]=ring(*a.cache_,row,length);}
   else if(d.hca_){auto&a=*d.hca_;item["attention_length"]=a.lengths_.at(row);item["ring"]=ring(*a.cache_,row,length);item["compressor"]=compressed(*a.compressor_,row,length);}
   else if(d.csa_){auto&a=*d.csa_;item["attention_length"]=a.lengths_.at(row);item["ring"]=ring(*a.cache_,row,length);item["compressor"]=compressed(*a.compressor_,row,length);auto&i=*a.indexer_;size_t valid=0;bool exact=true;for(size_t t=0;t<i.capacity_;++t){bool bit=bool(i.cache_->written_.at(row*i.capacity_+t));valid+=bit;exact=exact&&(bit==(t<length/4));}item["indexer"]={{"length",i.lengths_.at(row)},{"compressor_length",i.compressor_->lengths_.at(row)},{"valid_groups",valid},{"validity_matches_committed_groups",exact},{"revision",i.revision_}};}
   else throw std::logic_error("prefix decoder missing native attention");layer["rows"].push_back(item);
  }out["layers"].push_back(layer);
 }
 r.syncStream();return out;
}
