
int begin_long_validated_r305(void*ctx,uint64_t epoch,size_t count,const int32_t*logical,const int32_t*past,const int32_t*width){try{
 if(!ctx)return 1;Metadata next;next.epoch=epoch;next.scoped_long=true;next.context_limit=32768;next.pages_per_row=128;
 next.total=kimi_long_features_r305::validate_metadata(kimi_long_features_r305::long_scope(),epoch,count,logical,past,width);
 next.logical.assign(logical,logical+count);next.past.assign(past,past+count);next.width.assign(width,width+count);
 std::lock_guard<std::mutex>lock(mutex);armed.store(false,std::memory_order_release);records.clear();context=ctx;metadata=std::move(next);armed.store(true,std::memory_order_release);return 0;
 }catch(...){return 1;}}
