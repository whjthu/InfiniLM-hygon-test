#include <native_sparse_cache_r22.hpp>
#include <native_sparse_requests_r22.hpp>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#ifndef EXPECT_CAP_MAX
#error EXPECT_CAP_MAX is required
#endif
int main() {
    using infinilm::cache::NativeSparseCacheConfig;
    static_assert(sizeof(size_t)>=8);
    size_t cases=0, accepted=0;
    for(size_t rows: {size_t(0),size_t(1),size_t(2),size_t(128),size_t(129)}) {
        for(size_t cap: {size_t(0),size_t(1),size_t(4096),size_t(4097),size_t(32768),size_t(65536),size_t(65537)}) {
            const bool expected=rows>=1&&rows<=128&&cap>=1&&cap<=EXPECT_CAP_MAX;
            bool ok=false;
            try {
                NativeSparseCacheConfig config(rows,cap);
                if(config.rows()!=rows||config.capacity()!=cap) return 2;
                auto clone=config.unique_copy();
                auto *typed=dynamic_cast<NativeSparseCacheConfig*>(clone.get());
                if(!typed||typed->rows()!=rows||typed->capacity()!=cap) return 3;
                ok=true;
            } catch(const std::invalid_argument&) {}
            if(ok!=expected)return 4;
            ++cases;accepted+=ok;
        }
    }
    size_t continuity=0;
    for(size_t cap: {size_t(4096),size_t(32768),size_t(65536)}) {
        if(cap>EXPECT_CAP_MAX)continue;
        for(size_t rows: {size_t(1),size_t(2)}) {
            NativeSparseCacheConfig config(rows,cap);
            std::vector<size_t> lengths(rows,0);
            for(size_t p=0;p<cap;p+=1024) {
                auto requests=infinilm::models::native_sparse::validate_requests(rows,cap,lengths,{0},{int32_t(p)},{int32_t(p+1024)},{0,1024},1024);
                infinilm::models::native_sparse::commit_lengths(lengths,requests);
                if(lengths[0]!=p+1024||(rows==2&&lengths[1]!=0))return 5;
                ++continuity;
            }
            bool rejected=false;
            try {infinilm::models::native_sparse::validate_requests(rows,cap,lengths,{0},{int32_t(cap)},{int32_t(cap+1)},{0,1},1);}
            catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)return 6;
        }
    }
    bool too_many=false;
    try {infinilm::models::native_sparse::validate_requests(1,65536,{0},{0},{0},{4097},{0,4097},4097);}
    catch(const std::invalid_argument&){too_many=true;}
    if(!too_many)return 7;
    std::cout<<"{\"passed\":true,\"CPU_metadata_only\":true,\"cases\":"<<cases
             <<",\"accepted\":"<<accepted<<",\"capacity_guard\":"<<EXPECT_CAP_MAX
             <<",\"class_size\":"<<sizeof(NativeSparseCacheConfig)
             <<",\"real_request_metadata_chunk_commits\":"<<continuity<<",\"per_forward4097_rejected\":true}\n";
}
