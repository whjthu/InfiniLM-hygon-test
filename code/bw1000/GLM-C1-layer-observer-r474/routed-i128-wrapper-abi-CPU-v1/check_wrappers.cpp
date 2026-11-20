#include "routed_i128.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

extern "C" int glm_r470_i128_validate(const int64_t *, size_t, size_t, int *, void *);

using Gate = int(*)(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
using Down = int(*)(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
static_assert(std::is_same_v<decltype(&glm_r470_i128_gate), Gate>);
static_assert(std::is_same_v<decltype(&glm_r470_i128_down), Down>);

template<typename... Args>
void check_launch(dim3 grid, dim3 block, void *stream, unsigned before, Args... args) {
  assert(launches == before + 1 && error_calls == launches);
  assert(observed.grid.x == grid.x && observed.grid.y == grid.y && observed.grid.z == grid.z);
  assert(observed.block.x == block.x && observed.block.y == block.y && observed.block.z == block.z);
  assert(observed.stream == stream && observed.shared == 0);
  assert(observed.argc == sizeof...(Args));
  const std::array<std::uintptr_t,11> wanted{word(args)...};
  assert(observed.argv == wanted);
}

int main() {
  int buf[10]{};
  unsigned valid=0, rejected=0, validations=0;
  auto gate=[&](size_t n,size_t shift,void *stream) {
    return glm_r470_i128_gate(&buf[0],&buf[1],&buf[2],&buf[3],&buf[4],&buf[5],
                              &buf[6],&buf[7],&buf[8],n,shift,stream);
  };
  auto down=[&](size_t n,size_t shift,void *stream) {
    return glm_r470_i128_down(&buf[0],&buf[2],&buf[3],&buf[4],&buf[5],&buf[6],n,shift,stream);
  };
  for (size_t n : {size_t{8},size_t{64},size_t{32768}}) {
    for (size_t offset : {size_t{0},size_t{128},size_t{1920}}) {
      for (void *stream : {static_cast<void*>(nullptr),static_cast<void*>(&buf[9])}) {
        for (int result : {0,73}) {
          injected_status=result;
          unsigned prior=launches;
          assert(gate(n,offset,stream)==result);
          check_launch(dim3(128/GATE_ROWS,unsigned(n)),dim3(64*GATE_ROWS),stream,prior,
                       &buf[0],&buf[1],&buf[2],&buf[3],&buf[4],&buf[5],&buf[6],&buf[7],&buf[8],n,offset);
          prior=launches;
          assert(down(n,offset,stream)==result);
          check_launch(dim3(6144/DOWN_ROWS,unsigned(n)),dim3(32*DOWN_ROWS),stream,prior,
                       &buf[0],&buf[2],&buf[3],&buf[4],&buf[5],&buf[6],n,offset);
          valid+=2;
        }
      }
    }
  }
  for (const auto &bad : {std::array<size_t,2>{0,0},{7,0},{32769,0},{32776,0},{8,1},{8,2048}}) {
    auto prior=launches;
    assert(gate(bad[0],bad[1],nullptr)==1);
    assert(down(bad[0],bad[1],nullptr)==1);
    assert(launches==prior && error_calls==prior);
    rejected+=2;
  }
  std::array<int64_t,257> ids{};
  for (size_t length : {size_t{1},size_t{255},size_t{256},size_t{257}}) {
    for (int64_t tail : {int64_t{-1},int64_t{0},int64_t{255},int64_t{256}}) {
      ids.fill(0);
      ids[length-1]=tail;
      int status=0;
      injected_status=0;
      unsigned prior=launches;
      assert(glm_r470_i128_validate(ids.data(),length,256,&status,&buf[9])==0);
      check_launch(dim3(unsigned((length+255)/256)),dim3(256),&buf[9],prior,ids.data(),length,size_t{256},&status);
      assert(status == (tail<0 || tail>=256 ? 1 : 0));
      ++validations;
    }
  }
  std::cout << "{\"valid_launch_checks\":"<<valid<<",\"rejected_argument_checks\":"<<rejected
            <<",\"compiled_validate_kernel_checks\":"<<validations<<"}\n";
}
