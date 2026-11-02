#define main unchanged_archive_main
#include "../v2/CPU_archive_test.cpp"
#undef main
int main(){
    const char *boundary="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    std::string million(1000000,'a');
    if(sha256(boundary,std::strlen(boundary))!="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"||
       sha256(million.data(),million.size())!="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"){
        std::cout<<"{\"completed\":true,\"passed\":false,\"error\":\"SHA256 FIPS long/boundary vector\"}\n";return 1;
    }
    return unchanged_archive_main();
}
