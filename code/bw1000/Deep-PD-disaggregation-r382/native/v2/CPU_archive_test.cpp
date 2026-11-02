#include "deep_pd_state_r382.hpp"
#include "deep_pd_lifecycle_r382.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace infinilm::pd_r382;

// Exact fixed C1 geometry from the pinned43-layer checkpoint/current constructors.
// This is a host archive consumer fixture, not a loaded Native/GPU state test.
Operation fixture(uint64_t past){
    Operation op(Action::Export,past);Json schema=Json::array();size_t nt=0,nf=0;
    auto control=[&](std::string k,std::string kind){schema.push_back({k,kind});};
    auto length=[&](std::string k){control(k,"cursor");};
    auto revision=[&](std::string k){control(k,"revision");};
    auto poison=[&](std::string k){control(k,"poison");};
    auto block=[&](std::string key,std::string kind,std::string dtype,Json shape,Json storage,size_t pages,size_t n,size_t valid=0){
        Block b;b.layout={{"block",op.shards[0].blocks.size()},{"key",key},{"kind",kind},{"dtype",dtype},{"shape",shape},{"storage_shape",storage},{"livepages",pages},{"nbytes",n}};
        b.bytes.resize(n,0x5a);
        if(kind=="written_flags"){b.layout["valid_prefix"]=valid;for(size_t i=0;i<n;++i)b.bytes[i]=i<valid;nf++;}else nt++;
        op.shards[0].blocks.push_back(std::move(b));
    };
    auto paged=[&](std::string key,size_t valid,size_t capacity,size_t group,size_t stride){size_t pages=(valid+group-1)/group;
        block(key+".raw","packed_pages","uint8",{pages,stride},{(capacity+group-1)/group,stride},pages,pages*stride);
        block(key+".written","written_flags","uint8",{pages*group},{((capacity+group-1)/group)*group},0,pages*group,valid);
    };
    auto temporal=[&](std::string key,int ratio,int dim){length(key+".length");revision(key+".revision");
        size_t rows=ratio==4?8:128,channels=ratio==4?4:2;
        block(key+".rolling","rolling","bfloat16",{rows,channels*dim},{rows,channels*dim},0,rows*channels*dim*2);
        if(dim==512)paged(key+".cache",past/ratio,1048576/ratio,ratio==4?64:2,ratio==4?37440:1728);
    };
    length("native.length");length("model.length");poison("model.poison");
    for(int i=0;i<43;++i){auto k="layer."+std::to_string(i);length(k+".length");poison(k+".poison");
        length(k+".attention.length");if(i>=2)poison(k+".attention.poison");
        paged(k+".swa",std::min<uint64_t>(past,128),128,256,149760);
        if(i>=2){int ratio=i%2?128:4;temporal(k+".compressor",ratio,512);
            if(ratio==4){poison(k+".indexer.poison");length(k+".indexer.length");revision(k+".indexer.revision");temporal(k+".indexer.compressor",4,128);
                paged(k+".indexer.cache",past/4,1048576/4,64,8448);
            }
        }
    }
    if(nt!=167||nf!=105||schema.size()!=360)throw std::runtime_error("CPU fixture coverage changed");
    size_t n=8;for(auto&c:schema)n+=c[1]=="poison"?1:8;
    auto st=schema.dump();Block b;b.layout={{"block",272},{"key","control"},{"kind","control_le_v1"},{"dtype","uint8"},{"shape",Json::array({n})},{"storage_shape",Json::array({n})},{"livepages",0},{"nbytes",n},{"schema_sha256",sha256(st.data(),st.size())},{"control_schema",schema},{"cursors",171},{"revisions",83},{"poison_flags",106}};
    b.bytes.resize(n);std::memcpy(b.bytes.data(),"D382C1\0\1",8);size_t pos=8;
    for(auto&c:schema){if(c[1]=="poison")b.bytes[pos++]=0;else{uint64_t v=c[1]=="cursor"?past:10;for(int j=0;j<8;++j)b.bytes[pos++]=uint8_t(v>>(8*j));}}
    op.shards[0].blocks.push_back(std::move(b));
    for(int i=1;i<8;++i)op.shards[i]=op.shards[0];
    return op;
}
void received(Operation &op,const Json &m){for(int r=0;r<8;++r)for(int b=0;b<273;++b){auto&v=op.shards[r].blocks[b];v.layout=m["shards"][r]["blocks"][b];v.received=v.bytes.size();}}
int main(){Json cases=Json::array();auto check=[&](std::string name,auto body,bool reject=false){bool failed=false;try{body();}catch(const std::exception&){failed=true;}bool ok=failed==reject;cases.push_back({{"name",name},{"passed",ok}});if(!ok)throw std::runtime_error(name);};
    try{
        check("sha256_empty",[]{if(sha256(nullptr,0)!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")throw std::runtime_error("sha");});
        check("sha256_abc",[]{if(sha256("abc",3)!="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")throw std::runtime_error("sha");});
        bool failed=false;
        check("export_retire_remains_usable",[&]{retire_session(failed,false);require_regular(failed,false);});
        check("import_abort_sticky_poison",[&]{retire_session(failed,true);require_regular(failed,false);},true);
        failed=false;
        check("postcommit_ACKloss_poison",[&]{poison_generation(failed,7,7);require_regular(failed,false);},true);
        check("poison_idempotent_current_generation",[&]{poison_generation(failed,7,7);if(!failed)throw std::runtime_error("not failed");});
        failed=false;
        check("oldgeneration_cannot_poison_new_request",[&]{try{poison_generation(failed,8,7);}catch(const std::invalid_argument&){require_regular(failed,false);return;}throw std::runtime_error("stale accepted");});
        check("active_import_blocks_ordinary_forward_reset",[&]{require_regular(false,true);},true);
        uint64_t payload=0,header=0;
        for(uint64_t past:{uint64_t(1),uint64_t(128),uint64_t(8489),uint64_t(8492),uint64_t(8575),uint64_t(8576),uint64_t(32767)}){
            auto op=fixture(past);auto m=describe(op,23,7);header=m.dump().size();payload=m["payload_nbytes"];
            check("manifest_full_"+std::to_string(past),[&]{check_manifest(m,op);if(header>1024*1024)throw std::runtime_error("header");});
            received(op,m);check("payload_full_"+std::to_string(past),[&]{validate_payload(op);});
            if(past==8489)std::cout<<"";
        }
        auto op=fixture(1);auto m=describe(op,23,7);received(op,m);
        check("missing_shard",[&]{auto bad=m;bad["shards"].erase(7);check_manifest(bad,fixture(1));},true);
        check("wrong_dtype",[&]{auto bad=m;bad["shards"][0]["blocks"][0]["dtype"]="float32";check_manifest(bad,fixture(1));},true);
        check("unwritten_pages_revived",[&]{auto&b=op.shards[0].blocks[1];b.bytes.back()=1;b.layout["sha256"]=sha256(b.bytes.data(),b.bytes.size());validate_payload(op);},true);
        op=fixture(1);m=describe(op,23,7);received(op,m);
        check("short_payload",[&]{op.shards[0].blocks[0].bytes.pop_back();validate_payload(op);},true);
        op=fixture(1);m=describe(op,23,7);received(op,m);
        check("incomplete_stream",[&]{--op.shards[0].blocks[0].received;validate_payload(op);},true);
        op=fixture(1);m=describe(op,23,7);received(op,m);
        check("poisoned_control",[&]{auto&b=op.shards[0].blocks.back();b.bytes[24]=1;b.layout["sha256"]=sha256(b.bytes.data(),b.bytes.size());validate_payload(op);},true);
        auto real_size=fixture(8489);auto sizes=describe(real_size,23,7);
        std::cout<<Json({{"completed",true},{"passed",true},{"cases",cases},{"archive_fixture_only",true},{"actual_Runtime",false},{"actual_GPU_state",false},{"blocks_per_owner",273},{"past8489_payload_nbytes",sizes["payload_nbytes"]},{"past8489_descriptor_nbytes",sizes.dump().size()}}).dump()<<"\n";return 0;
    }catch(const std::exception&e){std::cout<<Json({{"completed",true},{"passed",false},{"error",e.what()},{"cases",cases}}).dump()<<"\n";return 1;}
}
