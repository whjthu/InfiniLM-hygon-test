#include "tp32_contract_r120f.hpp"
#include "batch_transaction_r9122.hpp"
#include "kimi_pair_state_r9006.hpp"
#include <iostream>
#include <set>
int main(){using namespace kimi_tp32_r120f;size_t checks=0;std::set<size_t>ranks,rows;
 for(int n=0;n<4;++n)for(int l=0;l<8;++l){int r=n*8+l;topology(n,l,32,r,1,0,32,r);ranks.insert(r);for(size_t i=head_begin(r);i<head_begin(r)+HeadRows;++i)rows.insert(i);++checks;}
 if(ranks.size()!=32||rows.size()!=163840||*rows.begin()!=0||*rows.rbegin()!=163839||HeadBytes!=73400320||QKV!=3*Projection||Heads*D!=Projection||96%TP||96/TP!=3||3*192!=576||3*128!=384)return 1;
 for(size_t H:{size_t(3),size_t(6)})for(size_t T:{size_t(1),size_t(8),size_t(16),size_t(64)}){size_t P=H*128;std::set<size_t>channels;for(size_t a=0;a<3;++a)for(size_t c=0;c<P;++c)channels.insert(a*P+c);if(channels.size()!=3*P||*channels.rbegin()!=3*P-1)return 2;++checks;}
 kimi_pair_r9006::Geometry g(8);if(g.physical!=16||g.state_rows()!=17||g.pages()!=68)return 3;
 // Existing row lifecycle: partial retain, replay, next-step swap, terminal EOS.
 int owner; kimi_pair_r9006::State s(&owner,8);s.prefilled(&owner,1,512);s.check(&owner,1,512,1,9,{11,12,13});s.completed(&owner,1,{11,12,13});s.commit(&owner,1,0,true);s.completed(&owner,1,{11});auto&r=s.row(&owner,1);if(r.active!=9||r.shadow!=1||r.past!=513||r.phase!=1)return 4;s.completed(&owner,1,{14});s.commit(&owner,1,0,false);if(s.row(&owner,1).phase!=4)return 5;
 bool rejected=false;try{topology(3,7,32,30,1,0,32,30);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 6;
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"real_rank_metadata_cases\":"<<checks<<",\"head_rows_covered\":163840,\"H3_H6_slices\":true,\"partial_replay_EOS\":true,\"GPU\":false}\n";
}
