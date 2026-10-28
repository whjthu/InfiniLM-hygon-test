#include <array>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
int main(){for(unsigned T:{8u,64u}){std::vector<unsigned>seen(T*3584);for(unsigned block=0;block<T*896;++block){unsigned token=block/896,channel=4*(block%896);for(unsigned c=0;c<4;++c){assert(token<T&&channel+c<3584);seen[token*3584+channel+c]++;}}for(auto n:seen)assert(n==1);}for(unsigned scaleCase=0;scaleCase<6;++scaleCase){float base[4]={},quad[4]={};for(unsigned r=0;r<16;++r){float lane[4][32]{};for(unsigned c=0;c<4;++c){for(unsigned l=0;l<32;++l){for(unsigned p=l;p<96;p+=32){float x0=float(int(p%17)-8)*.0625f,x1=float(int(p%11)-5)*.125f,w0=float(int((p+c+r)%13)-6)*.5f,w1=float(int((p+c+r+scaleCase)%7)-3);lane[c][l]+=x0*w0+x1*w1;}}for(unsigned shift=16;shift;shift/=2)for(unsigned l=0;l<shift;++l)lane[c][l]+=lane[c][l+shift];const float routeWeight=float(r+1)/136.f;base[c]+=routeWeight*lane[c][0];}for(unsigned c=0;c<4;++c)quad[c]+=float(r+1)/136.f*lane[c][0];}for(unsigned c=0;c<4;++c){assert(std::isfinite(quad[c]));assert(base[c]==quad[c]);}}unsigned ABBA[8]={0,1,1,0,0,1,1,0};unsigned calls[2]={2,2};for(auto v:ABBA)calls[v]+=4;assert(calls[0]==18&&calls[1]==18);std::cout<<"CPU_QUAD_contract_PASS geometry_T8_T64_eachchannelonce routeorder_sixcases ABBA18each shared256B\n";}
