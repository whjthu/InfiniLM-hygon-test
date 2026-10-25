import pathlib,json,hashlib,base64,zlib,difflib,ast
old=pathlib.Path('Kimi-rowreduce-group-private-r9042');b=pathlib.Path('Kimi-independent-rowAR-private-r9043');b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();original=(old/'delta-final.cpp').read_text();s=original.replace('#include <nccl.h>','')
start=s.index('    infinilm::runtime::Tensor forward_group_r9042(');end=s.index('    infinilm::runtime::Tensor forward_prefill512(',start)
method='''    infinilm::runtime::Tensor forward_rowar_r9043(infinilm::runtime::Runtime &runtime,infinilm::runtime::Tensor &input,int mode)const{
      runtime.assertOwnerThread();if(runtime.isGraphRecording()||tp_size_!=16||!communicator_||input->dtype()!=infinilm::runtime::DataType::kBFloat16||input->ndim()!=3||input->size(0)!=1||input->size(2)!=768||(input->size(1)!=1&&input->size(1)!=8)||has_bias()||alpha()!=1.f||output_dtype()!=input->dtype()||mode<0||mode>2||infinilm::engine::distributed::reduce_precision_r3226::selected(runtime)!=0)throw std::invalid_argument("9043 bounded original BF16 TP16 o_proj");
      std::vector<infinilm::runtime::Tensor>held,parts;infinilm::runtime::Tensor output;
      struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      if(mode==2&&input->size(1)==8){output=infinilm::runtime::Tensor::empty(runtime,{1,8,7168},input->dtype());for(size_t j=0;j<8;++j){auto row=input->narrow({{1,j,1}});held.push_back(row);auto part=compute_linear(runtime,row);held.push_back(part);if(part->shape()!=infinilm::runtime::Shape{1,1,7168})throw std::logic_error("9043 canonical W1 shape");output->narrow({{1,j,1}})->copy_from(runtime,part);}}
      else output=compute_linear(runtime,input);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9042_o_proj_local");
      if(mode&&input->size(1)==8){
        for(size_t j=0;j<8;++j){auto part=infinilm::runtime::Tensor::empty(runtime,{1,1,7168},output->dtype());part->copy_from(runtime,output->narrow({{1,j,1}}));if(!part->is_contiguous()||part->numel()!=7168)throw std::logic_error("9043 packed row");parts.push_back(part);}
        // Eight independent original messages. No external group, no per-row stream wait.
        for(auto &part:parts){infinilm::engine::distributed::allreduce_(runtime,part,part,infinicclSum,communicator_);++kimi_rowar_counts_r9043[2];}
        ++kimi_rowar_counts_r9043[mode-1];
        for(size_t j=0;j<8;++j)output->narrow({{1,j,1}})->copy_from(runtime,parts[j]);
      }else infinilm::engine::distributed::allreduce_(runtime,output,output,infinicclSum,communicator_);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9042_o_proj_reduced");return output;
    }
''';s=s[:start]+method+s[end:];needle='    auto result = [&]() {';start=s.index(needle);end=s.index('        const char *cc=',start)
selection='''    auto result = [&]() {
        const char *p9043=std::getenv("INFINILM_KIMI_ROWAR_R9043");int mode9043=0;if(p9043){if(std::string(p9043)=="1")mode9043=1;else if(std::string(p9043)=="2")mode9043=2;else if(std::string(p9043)!="0")throw std::invalid_argument("9043 mode0/1/2");}
        const char *t9043=std::getenv("INFINILM_KIMI_ROWAR_TRACE_R9043");bool trace9043=t9043&&std::string(t9043)=="1";if(t9043&&std::string(t9043)!="0"&&std::string(t9043)!="1")throw std::invalid_argument("9043 trace0/1");
        if(((mode9043&&seq_len==8)||trace9043)&&batch_size==1&&(seq_len==1||seq_len==8)){
          auto p=std::dynamic_pointer_cast<Prefill512RowR8603>(o_proj_);if(!p)throw std::logic_error("9043 original class");return p->forward_rowar_r9043(runtime,gated,mode9043);
        }
''';s=s[:start]+selection+s[end:];counter='''#include <cstdint>
namespace {thread_local std::array<std::uint64_t,3>kimi_rowar_counts_r9043{};}
extern "C" int modern_kimi_rowar_stats_r9043(std::uint64_t*out,std::size_t count){if(!out||count!=3)return 1;for(std::size_t j=0;j<3;++j)out[j]=kimi_rowar_counts_r9043[j];return 0;}
''';index=s.index('namespace ');s=s[:index]+counter+s[index:];assert 'ncclGroupStart('not in s and'ncclGroupEnd('not in s;assert 'INFINILM_KIMI_ROWGROUP'not in s;(b/'delta.cpp').write_bytes(s.encode());(b/'delta.patch').write_bytes(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='CPU282group_delta.cpp',tofile='independent_rowAR_delta.cpp')).encode())
cpu=(old/'Kimi_oproj_CPU_r9042b.py').read_text().replace('r9042b','r9043').replace('Kimi-rowreduce-group','Kimi-independent-rowAR');cpu=cpu.replace("link.extend(['-L/opt/dtk/lib','-lrccl']);",'');start=cpu.index("mainreloc=subprocess.check_output");end=cpu.index("q.update(completed=True,native=",start);cpu=cpu[:start]+"assert 'modern_kimi_rowar_stats_r9043'in subprocess.check_output(['nm','-D','--defined-only',str(main)],text=True);q['RCCL']={'path':'/opt/dtk/lib/librccl.so.1','SHA256':sha('/opt/dtk/lib/librccl.so.1'),'Main_GOT_group_slots':{}};"+cpu[end:];ast.parse(cpu);(b/'CPU_build.py').write_bytes(cpu.encode());inside='#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_oproj_CPU_r9043.py\n';(b/'CPU_build_inside.sh').write_bytes(inside.encode());payload=base64.b64encode(zlib.compress(json.dumps({'Kimi_oproj_delta_r9043.cpp':s,'Kimi_oproj_CPU_r9043.py':cpu,'Kimi_oproj_CPU_r9043_inside.sh':inside}).encode())).decode();remote=pathlib.Path('invoke_Kimi_group_CPU_r9042b.py').read_text();start=remote.index('base64.b64decode(')+len('base64.b64decode(');end=remote.index('))).items()',start);remote=remote[:start]+repr(payload)+remote[end:];remote=remote.replace('r9042b','r9043');compile(remote,'invoke','exec');pathlib.Path('invoke_Kimi_rowar_CPU_r9043.py').write_bytes(remote.encode());read=pathlib.Path('read_Kimi_group_CPU_r9042b.py').read_text().replace('r9042b','r9043');pathlib.Path('read_Kimi_rowar_CPU_r9043.py').write_bytes(read.encode());print('1HOST source/AST/LF/noGroup + actual TLS counter ready')
