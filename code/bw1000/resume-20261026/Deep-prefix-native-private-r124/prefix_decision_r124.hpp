#pragma once
#include <cstddef>
#include <stdexcept>
namespace infinilm::models::deepseek_v4::prefix_r124 {
enum class End{None,EOS,Cap};
struct Decision{size_t committed_inputs,emitted;bool verify;End end;};
// Pure token-policy helper. Tokens themselves stay with the caller; EOS from a
// rejected proposal is never observed as accepted EOS. MTP catchup is omitted
// only after explicit terminal release, never on continuing accept1 rows.
inline Decision decide(bool anchor_eos,bool accepted,bool proposal_eos,bool replacement_or_bonus_eos,size_t remaining){
 if(anchor_eos)return {0,0,false,End::EOS};
 if(!remaining)return {0,0,false,End::Cap};
 if(!accepted)return {1,1,true,replacement_or_bonus_eos?End::EOS:(remaining==1?End::Cap:End::None)};
 if(proposal_eos)return {2,1,true,End::EOS};
 if(remaining==1)return {2,1,true,End::Cap};
 return {2,2,true,replacement_or_bonus_eos?End::EOS:(remaining==2?End::Cap:End::None)};
}
}
