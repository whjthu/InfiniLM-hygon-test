#include "contract.hpp"
#include <cassert>
int main(){int rejected=0;for(int p=0;p<2;++p){auto g=kimi_r432::geometry(p);assert(g.n*g.k*2==51380224);for(int m=0;m<2;++m)kimi_r432::validate(p,m,g.n,g.k,g.k);for(int bad=0;bad<4;++bad)try{kimi_r432::validate(p,bad==0?2:0,g.n+(bad==1),g.k+(bad==2),g.k+(bad==3));}catch(const std::invalid_argument&){++rejected;}}assert(rejected==8);try{kimi_r432::geometry(2);}catch(const std::invalid_argument&){++rejected;}assert(rejected==9);}
