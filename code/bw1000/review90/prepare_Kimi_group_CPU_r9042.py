import pathlib,json,hashlib,difflib,base64,zlib,ast
b=pathlib.Path('Kimi-rowreduce-group-private-r9042');b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();original=pathlib.Path('Kimi-delta-selector-private-r9027/original_delta.cpp').read_text();assert hashlib.sha256(original.encode()).hexdigest()=='787af2aac64ae58dedfa2654f8efd94cc449c1aab7de03549b16789fed3898cd'
s=original.replace('#include "kimi_spec_math_r3676.hpp"','#include "kimi_spec_math_r3676.hpp"\n#include <rccl/rccl.h>')
assert s!=original
needle='    using infinilm::layers::linear::RowParallelLinear::RowParallelLinear;'
method='''
    infinilm::runtime::Tensor forward_group_r9042(infinilm::runtime::Runtime &runtime,infinilm::runtime::Tensor &input,bool grouped)const{
      runtime.assertOwnerThread();if(runtime.isGraphRecording()||tp_size_!=16||!communicator_||input->dtype()!=infinilm::runtime::DataType::kBFloat16||input->ndim()!=3||input->size(0)!=1||input->size(2)!=768||(input->size(1)!=1&&input->size(1)!=8)||has_bias()||alpha()!=1.f||output_dtype()!=input->dtype()||infinilm::engine::distributed::reduce_precision_r3226::selected(runtime)!=0)throw std::invalid_argument("9042 bounded original BF16 TP16 o_proj");
      std::vector<infinilm::runtime::Tensor>held;auto output=compute_linear(runtime,input);
      struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9042_o_proj_local");
      if(grouped&&input->size(1)==8){
        // Materialize all arguments/copies before opening RCCL group; no sync/ops inside group.
        for(size_t j=0;j<8;++j){auto part=infinilm::runtime::Tensor::empty(runtime,{1,1,7168},output->dtype());part->copy_from(runtime,output->narrow({{1,j,1}}));if(!part->is_contiguous()||part->numel()!=7168)throw std::logic_error("9042 packed row");held.push_back(part);}
        if(ncclGroupStart()!=ncclSuccess)std::_Exit(86);
        for(auto &part:held)infinilm::engine::distributed::allreduce_(runtime,part,part,infinicclSum,communicator_);
        if(ncclGroupEnd()!=ncclSuccess)std::_Exit(86);
        for(size_t j=0;j<8;++j)output->narrow({{1,j,1}})->copy_from(runtime,held[j]);
      }else infinilm::engine::distributed::allreduce_(runtime,output,output,infinicclSum,communicator_);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9042_o_proj_reduced");return output;
    }
'''
# Selected() declaration comes from the real Common header, not a new precision policy.
s=s.replace('#include <rccl/rccl.h>','#include <rccl/rccl.h>\n#include "engine/distributed/kimi_reduce_precision_r3226.hpp"');assert s.count(needle)==1;s=s.replace(needle,needle+method)
needle='    auto result = [&]() {'
replacement='''    auto result = [&]() {
        auto flag9042=[](const char*n){const char*p=std::getenv(n);if(!p||std::string(p)=="0")return false;if(std::string(p)=="1")return true;throw std::invalid_argument("9042 flag0or1");};
        bool selected9042=flag9042("INFINILM_KIMI_ROWGROUP_R9042"),trace9042=flag9042("INFINILM_KIMI_ROWGROUP_TRACE_R9042");
        if(((selected9042&&seq_len==8)||trace9042)&&batch_size==1&&(seq_len==1||seq_len==8)){
          auto p=std::dynamic_pointer_cast<Prefill512RowR8603>(o_proj_);if(!p)throw std::logic_error("9042 original class");return p->forward_group_r9042(runtime,gated,selected9042);
        }'''
assert s.count(needle)==1;s=s.replace(needle,replacement);(b/'delta.cpp').write_bytes(s.encode());(b/'delta.patch').write_bytes(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='actual787af_delta.cpp',tofile='delta.cpp')).encode())
cpu=pathlib.Path('Kimi-oproj-canonical-private-r9040/Kimi_oproj_CPU_r9040.py').read_text().replace('r9040','r9042').replace('Kimi-oproj-canonical','Kimi-rowreduce-group')
cpu=cpu.replace("cmd=parent['commands'][0][:]","cmd=parent['commands'][0][:];cmd.insert(1,'-I/opt/dtk/include')")
cpu=cpu.replace("main=b/'libKimi_oproj_Main_r9042.so'","link.extend(['-L/opt/dtk/lib','-lrccl']);main=b/'libKimi_oproj_Main_r9042.so'")
needle="q.update(completed=True,native="
rep="""mainreloc=subprocess.check_output(['readelf','-rW',str(main)],text=True);groupslots={}
 for name in ['ncclGroupStart','ncclGroupEnd']:
  rr=[x.split()for x in mainreloc.splitlines()if name in x and('JUMP_SLOT'in x or'GLOB_DAT'in x)];assert len(rr)==1;groupslots[name]=int(rr[0][0],16)
 rccl=pathlib.Path('/opt/dtk/lib/librccl.so.1');assert sha(rccl)=='32aa13da8bbd4d9a4ec37be08014f0783f5f576c7439dc10a3e1c661aa35427a';q['RCCL']={'path':str(rccl),'SHA256':sha(rccl),'Main_GOT_group_slots':groupslots};q.update(completed=True,native="""
assert cpu.count(needle)==1;cpu=cpu.replace(needle,rep);ast.parse(cpu);(b/'Kimi_oproj_CPU_r9042.py').write_bytes(cpu.encode())
inside=pathlib.Path('Kimi-oproj-canonical-private-r9040/CPU_inside.sh').read_text() # Build uses the successful original compiler runtime, not frontend-only shell.
buildinside='#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_oproj_CPU_r9042.py\n';(b/'CPU_build_inside.sh').write_bytes(buildinside.encode())
files={'Kimi_oproj_delta_r9042.cpp':s,'Kimi_oproj_CPU_r9042.py':cpu,'Kimi_oproj_CPU_r9042_inside.sh':buildinside};payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
remote="""import pathlib,json,zlib,base64,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists();assert chr(13)not in s;p.write_bytes(s.encode())
subprocess.run(['bash','-n',str(i/'Kimi_oproj_CPU_r9042_inside.sh')],check=True)
a=['srun','--jobid=910407','--overlap','--exact','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-oproj-CPU-r9042','--output='+str(r/'logs/Kimi-oproj-CPU-r9042-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_oproj_CPU_r9042_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-oproj-CPU-r9042-launch.log').open('x')as f:p=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':a}))
""".replace('PAYLOAD',repr(payload));compile(remote,'remote','exec');pathlib.Path('invoke_Kimi_group_CPU_r9042.py').write_bytes(remote.encode());print('source/AST/LF ready')
