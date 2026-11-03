#include "actquant_math.hpp"
#include "dispatch.hpp"
#include <fstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
int main(int argc,char**argv){try{
 if(argc==1){deep438::CPU_contract();std::cout<<"{\"passed\":true,\"owner_threads\":8,\"SDK_loaded\":false,\"GPU\":false}"<<std::endl;return 0;}
 if(argc!=4)throw std::invalid_argument("r438 pure math arguments");
 std::string mode=argv[1];std::ifstream input(argv[2],std::ios::binary);std::vector<char>raw((std::istreambuf_iterator<char>(input)),{});
 if(raw.empty()||raw.size()%(512*sizeof(uint16_t)))throw std::invalid_argument("r438 pure math input bytes");
 size_t rows=raw.size()/(512*sizeof(uint16_t));std::vector<uint16_t>gu(raw.size()/2);std::memcpy(gu.data(),raw.data(),raw.size());std::ofstream output(argv[3],std::ios::binary);
 for(size_t row=0;row<rows;++row){
  if(mode=="--activation"){std::array<uint16_t,256>a{};for(int i=0;i<256;++i)a[i]=deep438::host_activation(gu[row*512+i],gu[row*512+i+256]);output.write(reinterpret_cast<const char*>(a.data()),sizeof(a));}
  else if(mode=="--row"){auto a=deep438::host_row(gu.data()+row*512);output.write(reinterpret_cast<const char*>(a.activation.data()),sizeof(a.activation));output.write(reinterpret_cast<const char*>(a.packed.data()),sizeof(a.packed));output.write(reinterpret_cast<const char*>(&a.scale),sizeof(a.scale));}
  else throw std::invalid_argument("r438 pure math mode");
 }
 if(!output)throw std::runtime_error("r438 output failed");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
