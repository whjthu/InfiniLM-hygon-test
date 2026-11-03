#include "row_contract.hpp"
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace kimi_r437_row;
void require(bool b){if(!b)throw std::runtime_error("437 pure row contract failure");}
int main(){try{size_t cases=0;
 for(size_t k:{size_t(7168),size_t(3584)}){
  std::vector<uint16_t>storage(2*k,0x4000);for(size_t j=0;j<k;++j)storage[j]=uint16_t(0x3f80+j%4);const auto before=storage;
  const auto base=reinterpret_cast<uintptr_t>(storage.data());const auto capacity=bytes(2*k);
  for(ptrdiff_t pitch:{ptrdiff_t(k),ptrdiff_t(2*k),ptrdiff_t(1),std::numeric_limits<ptrdiff_t>::max()}){
   require(input_row(1,k,pitch,1,k));require(within(base,base,capacity,bytes(k)));
   for(size_t j=0;j<k;++j){const size_t logical_offset=size_t(0)*size_t(pitch)+j;require(logical_offset==j);require(reinterpret_cast<uintptr_t>(&storage[logical_offset])==base+2*j);require(storage[logical_offset]==before[j]);}++cases;
  }
  require(storage==before);++cases;
  require(!input_row(2,k,ptrdiff_t(k),1,k));require(!input_row(1,k,0,1,k));require(!input_row(1,k,-1,1,k));require(!input_row(1,k,ptrdiff_t(k),2,k));require(!input_row(1,k-1,ptrdiff_t(k),1,k));cases+=5;
  require(!within(base-1,base,capacity,bytes(k)));require(!within(base+capacity,base,capacity,bytes(k)));require(within(base+bytes(k),base,capacity,bytes(k)));require(!within(base+bytes(k)+2,base,capacity,bytes(k)));cases+=4;
 }
 require(weight(3584,7168,7168,1,3584,7168));require(weight(7168,3584,3584,1,7168,3584));require(!weight(3584,7168,14336,1,3584,7168));require(!weight(7168,3584,3584,2,7168,3584));cases+=4;
 bool overflow=false;try{bytes(std::numeric_limits<size_t>::max());}catch(const std::overflow_error&){overflow=true;}require(overflow);++cases;
 std::cout<<"{\"passed\":true,\"address_and_rejection_cases\":"<<cases<<",\"actual_K\":[7168,3584],\"source_math\":\"M1 address=data+(0*rowpitch+j)*2\",\"CPU_pure_only\":true,\"SDK_or_Tensor_runtime_executed\":false,\"GPU\":0}"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
