#include "deep_pd_state_r382.hpp"
#include "deepseek_v4_routed_text_model_r20.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
void PdStateCodec::run(runtime::Runtime &r,TextModelR20 &m,std::vector<size_t> &outer,
                       pd_r382::Operation &op,unsigned rank) {
    using namespace pd_r382;
    r.assertOwnerThread();
    if(rank>=8||r.isGraphRecording()||m.owner_!=&r||m.rows_!=1||m.capacity_!=32768||
       !m.ready_||m.poisoned_||m.layers_.size()!=43||!op.past||op.past>=32768)
        throw std::invalid_argument("PD requires ready unpoisoned C1 TP8 context32768");
    const bool exporting=op.action==Action::Export,applying=op.action==Action::Import;
    const size_t before=exporting?op.past:0;
    struct Control {std::string key,kind;size_t *number=nullptr;bool *flag=nullptr;};
    struct Target {runtime::Tensor tensor;std::vector<uint8_t>*flags=nullptr;size_t prefix=0;};
    std::vector<Control>controls;std::vector<Target>targets;std::vector<runtime::Tensor>scratch_status;Shard local;
    size_t tensors=0,flags=0,cursors=0,revisions=0,poisons=0;
    auto lengths=[&](const std::string &key,std::vector<size_t> &v){
        if(v.size()!=1||v[0]!=before)throw std::invalid_argument("PD cursor mismatch: "+key);
        controls.push_back({key,"cursor",&v[0],nullptr});++cursors;
    };
    auto revision=[&](const std::string &key,size_t &v){
        if(v==std::numeric_limits<size_t>::max())throw std::invalid_argument("PD revision exhausted");
        controls.push_back({key,"revision",&v,nullptr});++revisions;
    };
    auto poison=[&](const std::string &key,bool &v){
        if(v)throw std::invalid_argument("PD poisoned source/destination: "+key);
        controls.push_back({key,"poison",nullptr,&v});++poisons;
    };
    auto tensor=[&](const std::string &key,const runtime::Tensor &t,size_t pages,size_t stride,const std::string &kind){
        if(!t||!t->is_contiguous()||t->device()!=r.device())throw std::invalid_argument("PD tensor owner: "+key);
        auto shape=t->shape();size_t n=t->nbytes();std::string dtype="bfloat16";
        if(kind=="packed_pages"){
            if(t->dtype()!=runtime::DataType::kUInt8||shape.size()!=2||shape[1]!=stride||pages>shape[0])throw std::invalid_argument("PD packed geometry");
            n=pages*stride;shape[0]=pages;dtype="uint8";
        }else if(t->dtype()!=runtime::DataType::kBFloat16)throw std::invalid_argument("PD rolling dtype");
        Block b;b.layout={{"block",local.blocks.size()},{"key",key},{"kind",kind},{"dtype",dtype},{"shape",shape},{"storage_shape",t->shape()},{"livepages",pages},{"nbytes",n}};
        b.bytes.resize(n);local.blocks.push_back(std::move(b));targets.push_back({t,nullptr,n});++tensors;
    };
    auto written=[&](const std::string &key,std::vector<uint8_t> &v,size_t prefix,size_t valid){
        if(prefix>v.size()||valid>prefix)throw std::invalid_argument("PD written bound");
        for(size_t i=0;i<v.size();++i)if(v[i]!=(exporting&&i<valid?1:0))throw std::invalid_argument("PD written validity: "+key);
        Block b;b.layout={{"block",local.blocks.size()},{"key",key},{"kind","written_flags"},{"dtype","uint8"},{"shape",Json::array({prefix})},{"storage_shape",Json::array({v.size()})},{"livepages",0},{"nbytes",prefix},{"valid_prefix",valid}};
        b.bytes.resize(prefix);local.blocks.push_back(std::move(b));targets.push_back({{},&v,prefix});++flags;
    };
    auto paged=[&](const std::string &key,PagedLatentCache &c,size_t valid){
        if(c.owner_.get()!=&r||!c.block_||c.data_->shape()[0]!=c.pages_)throw std::invalid_argument("PD latent owner/geometry");
        if(!c.flag_||c.flag_->dtype()!=runtime::DataType::kInt32||c.flag_->numel()!=1)throw std::invalid_argument("PD status scratch geometry");
        scratch_status.push_back(c.flag_);
        size_t pages=(valid+c.block_-1)/c.block_;
        tensor(key+".raw",c.data_,pages,c.stride_,"packed_pages");
        written(key+".written",c.written_,pages*c.block_,valid);
    };
    auto temporal=[&](const std::string &key,TemporalCompressor &c){
        if(c.owner_!=&r||c.rows_!=1||!c.ready_||(c.ratio_!=4&&c.ratio_!=128))throw std::invalid_argument("PD temporal geometry");
        lengths(key+".length",c.lengths_);revision(key+".revision",c.revision_);
        tensor(key+".rolling",c.state_,0,0,"rolling");
        if(c.dim_==512){if(!c.cache_)throw std::invalid_argument("PD missing compressor cache");paged(key+".cache",*c.cache_,op.past/c.ratio_);}
        else if(c.dim_!=128||c.ratio_!=4||c.cache_)throw std::invalid_argument("PD index temporal geometry");
    };
    auto indexer=[&](const std::string &key,C4Indexer &c){
        if(c.owner_!=&r||c.rows_!=1||!c.ready_||!c.cache_||!c.compressor_)throw std::invalid_argument("PD index owner/geometry");
        poison(key+".poison",c.poisoned_);lengths(key+".length",c.lengths_);revision(key+".revision",c.revision_);
        temporal(key+".compressor",*c.compressor_);
        auto &p=*c.cache_;if(p.owner_.get()!=&r||p.raw_->shape()!=runtime::Shape({p.pages_,8448}))throw std::invalid_argument("PD index pages");
        size_t valid=op.past/4,pages=(valid+63)/64;
        tensor(key+".cache.raw",p.raw_,pages,8448,"packed_pages");written(key+".cache.written",p.written_,pages*64,valid);
    };
    lengths("native.length",outer);lengths("model.length",m.lengths_);poison("model.poison",m.poisoned_);
    for(size_t i=0;i<m.layers_.size();++i){auto &d=*m.layers_[i];std::string k="layer."+std::to_string(i);
        if(d.owner_!=&r||d.rows_!=1||!d.ready_)throw std::invalid_argument("PD decoder owner/geometry");
        lengths(k+".length",d.lengths_);poison(k+".poison",d.poisoned_);
        if(d.ratio_==0){if(!d.swa_||d.hca_||d.csa_)throw std::invalid_argument("PD SWA branch");auto &a=*d.swa_;
            if(a.owner_!=&r||a.rows_!=1||a.window_!=128||!a.ready_)throw std::invalid_argument("PD SWA owner");
            lengths(k+".attention.length",a.lengths_);paged(k+".swa",*a.cache_,std::min<size_t>(op.past,128));
        }else if(d.ratio_==128){if(!d.hca_||d.swa_||d.csa_)throw std::invalid_argument("PD HCA branch");auto &a=*d.hca_;
            if(a.owner_!=&r||a.rows_!=1||a.window_!=128||!a.ready_)throw std::invalid_argument("PD HCA owner");
            lengths(k+".attention.length",a.lengths_);poison(k+".attention.poison",a.poisoned_);
            paged(k+".swa",*a.cache_,std::min<size_t>(op.past,128));temporal(k+".compressor",*a.compressor_);
        }else if(d.ratio_==4){if(!d.csa_||d.swa_||d.hca_)throw std::invalid_argument("PD CSA branch");auto &a=*d.csa_;
            if(a.owner_!=&r||a.rows_!=1||a.window_!=128||!a.ready_)throw std::invalid_argument("PD CSA owner");
            lengths(k+".attention.length",a.lengths_);poison(k+".attention.poison",a.poisoned_);
            paged(k+".swa",*a.cache_,std::min<size_t>(op.past,128));temporal(k+".compressor",*a.compressor_);indexer(k+".indexer",*a.indexer_);
            // last_selection_ is output-only instrumentation: forward overwrites it
            // before any use. It is intentionally not a causal state block.
        }else throw std::invalid_argument("PD unsupported compression ratio");
    }
    if(tensors!=167||flags!=105||cursors!=171||revisions!=83||poisons!=106||targets.size()!=272)
        throw std::runtime_error("PD complete state coverage changed");
    Json schema=Json::array();size_t controlbytes=8;
    for(const auto&c:controls){schema.push_back({c.key,c.kind});controlbytes+=c.flag?1:8;}
    auto schema_text=schema.dump();Block meta;
    meta.layout={{"block",272},{"key","control"},{"kind","control_le_v1"},{"dtype","uint8"},{"shape",Json::array({controlbytes})},{"storage_shape",Json::array({controlbytes})},{"livepages",0},{"nbytes",controlbytes},{"schema_sha256",sha256(schema_text.data(),schema_text.size())},{"control_schema",schema},{"cursors",cursors},{"revisions",revisions},{"poison_flags",poisons}};
    meta.bytes.resize(controlbytes);local.blocks.push_back(std::move(meta));
    auto &archive=op.shards[rank];
    if(applying){
        if(archive.blocks.size()!=273)throw std::invalid_argument("PD import block coverage");
        for(size_t i=0;i<273;++i)if(layout_only(archive.blocks[i].layout)!=local.blocks[i].layout||archive.blocks[i].bytes.size()!=local.blocks[i].bytes.size())throw std::invalid_argument("PD owner import layout/byte count");
        const auto &b=archive.blocks.back().bytes;
        if(b.size()!=controlbytes||std::memcmp(b.data(),"D382C1\0\1",8))throw std::invalid_argument("PD control encoding");
        size_t pos=8;
        for(const auto&c:controls){uint64_t v=0;if(c.flag)v=b[pos++];else{for(int j=0;j<8;++j)v|=uint64_t(b[pos++])<<(8*j);}
            if((c.kind=="cursor"&&v!=op.past)||(c.kind=="poison"&&v!=0)||(c.kind=="revision"&&v>=std::numeric_limits<size_t>::max()))throw std::invalid_argument("PD control value");
        }
        for(size_t i=0;i<targets.size();++i)if(targets[i].flags){const auto &src=archive.blocks[i].bytes;size_t valid=archive.blocks[i].layout.at("valid_prefix");for(size_t j=0;j<src.size();++j)if(src[j]!=(j<valid?1:0))throw std::invalid_argument("PD imported validity");}
        // All validation is complete before the first device write. Failure from
        // this point poisons the whole context; no partial import may resume.
        try{
            r.syncStream();
            for(const auto &status:scratch_status)r.setDeviceMemory(status->data(),0,status->nbytes());
            for(size_t i=0;i<targets.size();++i)if(targets[i].tensor&&targets[i].prefix)r.memcpyH2D(targets[i].tensor->data(),archive.blocks[i].bytes.data(),targets[i].prefix);
            r.syncStream();
            for(size_t i=0;i<targets.size();++i)if(targets[i].flags){auto &v=*targets[i].flags;std::fill(v.begin(),v.end(),0);std::copy(archive.blocks[i].bytes.begin(),archive.blocks[i].bytes.end(),v.begin());}
            pos=8;for(const auto&c:controls){if(c.flag){*c.flag=false;++pos;}else{uint64_t v=0;for(int j=0;j<8;++j)v|=uint64_t(b[pos++])<<(8*j);*c.number=size_t(v+(c.kind=="revision"?1:0));}}
        }catch(...){m.poisoned_=true;throw;}
    }else if(exporting){
        r.syncStream();
        for(size_t i=0;i<targets.size();++i){auto &dst=local.blocks[i].bytes;auto &t=targets[i];if(t.tensor&&t.prefix)r.memcpyD2H(dst.data(),t.tensor->data(),t.prefix);else if(t.flags)std::copy_n(t.flags->begin(),t.prefix,dst.begin());}
        r.syncStream();auto &b=local.blocks.back().bytes;std::memcpy(b.data(),"D382C1\0\1",8);size_t pos=8;
        for(const auto&c:controls)if(c.flag)b[pos++]=*c.flag;else for(int j=0;j<8;++j)b[pos++]=uint8_t(uint64_t(*c.number)>>(8*j));
        archive=std::move(local);
    }else archive=std::move(local);
}
}
