#include "deep_c4_smallt_guard_r9252.hpp"
#include <vector>
#include <cassert>
#include <iostream>
struct Row{size_t row,past,tokens;};namespace s=infinilm::models::deepseek_v4::small_c4_r9252;
int main(){for(size_t n:{1,2,4,7}){std::vector<Row>r;for(size_t i=0;i<n;++i)r.push_back({i*3,391+i,1});assert(s::decode(n,r)&&s::eligible(true,n,r,512)&&!s::eligible(false,n,r,512));r[0].past=2050;assert(s::eligible(true,n,r,512));r[0].past=2051;assert(!s::eligible(true,n,r,512));r[0].past=0;assert(!s::eligible(true,n,r,512));r[0].past=SIZE_MAX;assert(!s::eligible(true,n,r,512));r[0].past=391;r[0].tokens=2;assert(!s::eligible(true,n,r,512));}std::vector<Row>legacy8(8,{0,391,1});assert(!s::eligible(true,8,legacy8,512));std::vector<Row>prefill(1,{0,0,128});assert(!s::eligible(true,128,prefill,512));assert(!s::eligible(true,2,std::vector<Row>{{0,400,1}},512));std::cout<<"C4_SMALLT_CPU_PASS defaultOFF/T1T2T4T7/512to513/mixed/zeroPast/overflow/multitoken/legacyControls"<<std::endl;}
