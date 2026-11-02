#include "deep_pd_state_r382.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace infinilm::pd_r382 {
// FIPS 180-4 SHA-256, host-only. CPU closure checks standard vectors.
std::string sha256(const void *raw, size_t n) {
    static constexpr uint32_t k[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    auto rr=[](uint32_t x,int r){return (x>>r)|(x<<(32-r));};
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const auto *p=static_cast<const uint8_t *>(raw);
    if (n > UINT64_MAX/8 || (n && !p)) throw std::invalid_argument("PD SHA input");
    auto compress=[&](const uint8_t *b){
        uint32_t w[64];
        for(int i=0;i<16;++i)w[i]=(uint32_t(b[4*i])<<24)|(uint32_t(b[4*i+1])<<16)|(uint32_t(b[4*i+2])<<8)|b[4*i+3];
        for(int i=16;i<64;++i){auto a=w[i-15],z=w[i-2];w[i]=w[i-16]+(rr(a,7)^rr(a,18)^(a>>3))+w[i-7]+(rr(z,17)^rr(z,19)^(z>>10));}
        uint32_t a=h[0],b0=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;++i){auto t1=hh+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+k[i]+w[i];auto t2=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b0)^(a&c)^(b0&c));hh=g;g=f;f=e;e=d+t1;d=c;c=b0;b0=a;a=t1+t2;}
        uint32_t v[8]={a,b0,c,d,e,f,g,hh};for(int i=0;i<8;++i)h[i]+=v[i];
    };
    size_t full=n/64;for(size_t i=0;i<full;++i)compress(p+i*64);
    uint8_t tail[128]={};size_t rem=n%64;if(rem)std::memcpy(tail,p+full*64,rem);tail[rem]=0x80;
    size_t padded=rem<56?64:128;uint64_t bits=uint64_t(n)*8;
    for(int i=0;i<8;++i)tail[padded-1-i]=uint8_t(bits>>(8*i));
    compress(tail);if(padded==128)compress(tail+64);
    std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:h)s<<std::setw(8)<<v;return s.str();
}
Json layout_only(const Json &v) { auto out=v;out.erase("sha256");return out; }
Json describe(const Operation &op,uint64_t epoch,uint64_t generation) {
    Json out={{"abi",ABI},{"past",op.past},{"rows",1},{"tp",8},{"epoch",epoch},{"source_generation",generation},{"shards",Json::array()}};
    uint64_t total=0;
    for(unsigned rank=0;rank<8;++rank){
        Json shard={{"rank",rank},{"blocks",Json::array()}};
        for(const auto &b:op.shards[rank].blocks){auto d=b.layout;d["sha256"]=sha256(b.bytes.data(),b.bytes.size());shard["blocks"].push_back(d);total+=b.bytes.size();}
        out["shards"].push_back(shard);
    }
    out["payload_nbytes"]=total;return out;
}
void check_manifest(const Json &m,const Operation &expected) {
    if(m.at("abi")!=ABI||m.at("rows")!=1||m.at("tp")!=8||m.at("past")!=expected.past||m.at("shards").size()!=8)
        throw std::invalid_argument("PD manifest identity/layout differs");
    uint64_t total=0;
    for(unsigned r=0;r<8;++r){const auto &s=m.at("shards").at(r);const auto &blocks=s.at("blocks");
        if(s.at("rank")!=r||blocks.size()!=273||expected.shards[r].blocks.size()!=273)throw std::invalid_argument("PD shard coverage differs");
        for(unsigned i=0;i<blocks.size();++i){const auto&b=blocks[i];
            if(layout_only(b)!=expected.shards[r].blocks[i].layout||b.at("sha256").get<std::string>().size()!=64)throw std::invalid_argument("PD block layout differs");
            total+=b.at("nbytes").get<uint64_t>();
        }
    }
    if(total!=m.at("payload_nbytes")||total>uint64_t(64)*1024*1024*1024)throw std::invalid_argument("PD payload bound differs");
}
void validate_payload(const Operation &op) {
    for(const auto&s:op.shards)for(const auto&b:s.blocks){
        if(b.received!=b.bytes.size()||b.bytes.size()!=b.layout.at("nbytes").get<size_t>()||sha256(b.bytes.data(),b.bytes.size())!=b.layout.at("sha256"))
            throw std::invalid_argument("PD incomplete/corrupt block");
        if(b.layout.at("kind")=="written_flags"){
            size_t valid=b.layout.at("valid_prefix");
            for(size_t i=0;i<b.bytes.size();++i)if(b.bytes[i]!=(i<valid?1:0))throw std::invalid_argument("PD written payload differs");
        }else if(b.layout.at("kind")=="control_le_v1"){
            if(b.bytes.size()<8||std::memcmp(b.bytes.data(),"D382C1\0\1",8))throw std::invalid_argument("PD control magic");
            size_t pos=8;
            for(const auto &c:b.layout.at("control_schema")){
                auto kind=c.at(1).get<std::string>();size_t width=kind=="poison"?1:8;
                if(pos+width>b.bytes.size())throw std::invalid_argument("PD control short");
                uint64_t value=0;for(size_t j=0;j<width;++j)value|=uint64_t(b.bytes[pos++])<<(8*j);
                if((kind=="cursor"&&value!=op.past)||(kind=="poison"&&value!=0)||(kind=="revision"&&value>=UINT64_MAX-1))throw std::invalid_argument("PD control payload value");
            }
            if(pos!=b.bytes.size())throw std::invalid_argument("PD control trailing bytes");
        }
    }
}
}
