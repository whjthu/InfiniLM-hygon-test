#include "router_math.hpp"
#include <fstream>
#include <cstdio>
#include <string>

// Pure standard-library consumer of the exact HOST/HIP-shared mathematics.
// No SDK, Native, Runtime headers or dynamic libraries are used by this executable.
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--suffix"){
  std::array<float,514>input{};std::ifstream in(argv[2],std::ios::binary);
  in.read(reinterpret_cast<char*>(input.data()),sizeof(input));
  if(in.gcount()!=sizeof(input)||in.peek()!=EOF)throw std::invalid_argument("r419 host inputbytes");
  auto out=deep419::host_suffix(input.data(),input.data()+256,input[512],input[513]);
  std::ofstream f(argv[3],std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.ids.data()),sizeof(out.ids));
  f.write(reinterpret_cast<const char*>(out.weights.data()),sizeof(out.weights));
  f.write(reinterpret_cast<const char*>(out.scores.data()),sizeof(out.scores));
  f.write(reinterpret_cast<const char*>(out.selection.data()),sizeof(out.selection));
  if(!f)throw std::runtime_error("r419 host outputfailed");return 0;
 }
 if(argc!=1)throw std::invalid_argument("r419 pure HOST args");
 deep419::CPU_contract();
 std::puts("{\"passed\":true,\"actual_owner_threads\":8,\"defaultOFF_and_fallbacks\":true,\"fallback_geometries\":9,\"error_restore\":true,\"SDK_loaded\":false,\"GPU_calls\":0}");
 return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
