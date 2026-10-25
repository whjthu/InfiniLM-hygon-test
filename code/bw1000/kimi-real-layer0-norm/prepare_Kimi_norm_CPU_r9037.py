import pathlib,json,zlib,base64
r=pathlib.Path('.');b=r/'Kimi-norm-real-input-private-r9037';b.mkdir(exist_ok=False)
native=r'''#include "runtime/runtime.hpp"
#include "layers/embedding/embedding.hpp"
#include "layers/normalization/rmsnorm.hpp"
#include <infini/rt/device_guard.h>
#include <array>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
namespace rt=infinilm::runtime;
namespace {thread_local std::string error;struct C{std::shared_ptr<rt::Runtime>r;std::unique_ptr<infinilm::layers::Embedding>e;std::unique_ptr<infinilm::layers::RMSNorm>n;std::vector<unsigned char>er,nr;std::array<uint64_t,4>counts{};size_t cap=0;};void drain(C*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}rt::Tensor host(const void*p,rt::Shape s,rt::DataType d){return rt::Tensor::from_blob(const_cast<void*>(p),s,d,{rt::Device::Type::kCpu,0});}}
extern "C" const char*norm_error_r9037(){return error.c_str();}
extern "C" void*norm_create_r9037(const void*e,const void*n,double eps,size_t cap){std::unique_ptr<C>c;try{if(!e||!n||cap<2||cap>16)throw std::invalid_argument("bounded actual embedding/norm inputs");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,0,true};c=std::make_unique<C>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,0});c->r->enableOwnerDrainFailStop();c->cap=cap;c->e=std::make_unique<infinilm::layers::Embedding>(*c->r,213,7168,std::nullopt,rt::DataType::kBFloat16);c->n=std::make_unique<infinilm::layers::RMSNorm>(*c->r,7168,eps,rt::DataType::kBFloat16);c->e->load_parameter(*c->r,"weight",host(e,{213,7168},rt::DataType::kBFloat16));c->n->load_parameter(*c->r,"weight",host(n,{7168},rt::DataType::kBFloat16));c->er.assign((const unsigned char*)e,(const unsigned char*)e+213*7168*2);c->nr.assign((const unsigned char*)n,(const unsigned char*)n+7168*2);drain(c.get());return c.release();}catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" int norm_run_r9037(void*raw,size_t m,const int64_t*ids,void*emb,void*out){auto*c=(C*)raw;rt::Tensor h,x,value,res,scratch,y;try{if(!c||!ids||!emb||!out||(m!=1&&m!=8&&m!=512))throw std::invalid_argument("M512/M1/M8 only");auto&r=*c->r;r.assertOwnerThread();for(size_t i=0;i<m;++i)if(ids[i]<0||ids[i]>=213)throw std::invalid_argument("compact ID range");h=host(ids,{1,m},rt::DataType::kInt64);x=rt::Tensor::empty(r,{1,m},rt::DataType::kInt64);x->copy_from(r,h);value=c->e->forward(r,x);res=rt::Tensor::empty(r,{m,c->cap,7168},rt::DataType::kBFloat16);scratch=res->narrow({{1,0,1}})->view({1,m,7168});scratch->copy_from(r,value);if(scratch->strides().back()!=1||(m>1&&scratch->strides()[1]!=c->cap*7168))throw std::logic_error("actual residual scratch stride");y=c->n->forward(r,scratch);if(y->shape()!=rt::Shape{1,m,7168}||!y->is_contiguous())throw std::logic_error("actual norm result layout");r.syncStream();r.memcpyD2H(emb,value->data(),m*7168*2);r.memcpyD2H(out,y->data(),m*7168*2);r.syncStream();++c->counts[m==512?0:m==1?1:2];c->counts[3]+=m;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int norm_readonly_r9037(void*raw){auto*c=(C*)raw;try{if(!c)throw std::invalid_argument("null");drain(c);std::vector<unsigned char>e(c->er.size()),n(c->nr.size());c->r->memcpyD2H(e.data(),c->e->weight()->data(),e.size());c->r->memcpyD2H(n.data(),c->n->weight()->data(),n.size());drain(c);if(e!=c->er||n!=c->nr)throw std::runtime_error("GPU weight mutated");return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int norm_counts_r9037(void*raw,uint64_t*out){auto*c=(C*)raw;if(!c||!out)return 1;c->r->assertOwnerThread();for(size_t i=0;i<4;++i)out[i]=c->counts[i];return 0;}
extern "C" int norm_close_r9037(void*raw){auto*c=(C*)raw;if(!c)return 1;drain(c);c->n.reset();c->e.reset();drain(c);delete c;return 0;}
'''
build=r'''import pathlib,json,subprocess,hashlib,os,traceback,struct
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/Kimi-norm-real-input-r9037';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'native_host_TUs':1}
try:
 old=json.loads((r/'results/Kimi-projection-CPU-r9030c-910407.261.json').read_text());p=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert sha(p)=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625';prompt=json.loads(p.read_text())['natural_input_ids'][0];rp=r/'results/kimi-crossover-r9026-TP16PP2-I1928669zh-8e319d252bb323d5-node2.json';assert sha(rp)=='9c0b2fd68291c69379449617f07390b3edd7d4ce651c3df9a0b83f219d0dc83d';report=json.loads(rp.read_text());A=next(a for a in report['crossover_arms']if a['arm']=='A');ids=prompt+A['anchor']+A['actual_ids'][:31];unique=sorted(set(ids));assert len(ids)==544 and len(unique)==213;compact=[unique.index(x)for x in ids];model=pathlib.Path('/public/home/aclsylqidf/shared/Kimi-K3');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];specs=[('language_model.model.embed_tokens.weight','embedding213.bf16'),('language_model.model.layers.0.input_layernorm.weight','norm0.bf16')];weights={}
 for key,name in specs:
  p=model/index[key]
  with p.open('rb')as f:
   n=int.from_bytes(f.read(8),'little');header=f.read(n);m=json.loads(header)[key];assert m['dtype']=='BF16';base=8+n+m['data_offsets'][0]
   if 'embed_tokens'in key:
    assert m['shape']==[163840,7168];data=bytearray()
    for token in unique:f.seek(base+token*7168*2);v=f.read(7168*2);assert len(v)==7168*2;data.extend(v)
   else:assert m['shape']==[7168];f.seek(base);data=f.read(7168*2);assert len(data)==7168*2
  p=b/name;p.write_bytes(data);weights[key]={'path':str(p),'SHA256':sha(p),'bytes':len(data),'checkpoint':str(model/index[key]),'header_SHA256':hashlib.sha256(header).hexdigest(),'shape':m['shape'],'dtype':m['dtype']}
 cfg=json.loads((model/'config.json').read_text())['text_config'];eps=cfg['rms_norm_eps'];block=cfg['attn_res_block_size'];stage0end=report['PP_ranges'][0][1];cap=(stage0end+block-1)//block+1;assert cap>=2;meta={'original_ids':ids,'unique_token_ids':unique,'compact_ids':compact,'weights':weights,'eps':eps,'attn_res_block_size':block,'stage0end':stage0end,'residual_capacity':cap,'scale':None,'bias':None,'source_report_SHA256':sha(rp),'embedding_rowbytes':14336};(b/'fixture.json').write_text(json.dumps(meta,indent=2));q['fixture']=meta
 src=b/'norm_native.cpp';src.write_bytes((i/'Kimi_norm_native_r9037.cpp').read_bytes());cmd=old['command'][:];cmd[cmd.index('-shared')+1]=str(src);out=b/'libKimi_norm_r9037.so';cmd[cmd.index('-o')+1]=str(out);q['command']=cmd;p=subprocess.run(cmd,text=True,capture_output=True,timeout=150);(b/'build.log').write_text(p.stdout+p.stderr);q['compile_returncode']=p.returncode;assert p.returncode==0,p.stderr[-6000:];nm=subprocess.check_output(['nm','-D','--defined-only',str(out)],text=True);assert all(x in nm for x in ['norm_create_r9037','norm_run_r9037','norm_close_r9037','norm_readonly_r9037','norm_counts_r9037']);q.update(completed=True,native=str(out),native_SHA256=sha(out),source_SHA256=sha(src),provider_DSO=old['provider_DSO'],provider_DSO_SHA256=old['provider_DSO_SHA256']);symbols=subprocess.check_output(['nm','-D',old['provider_DSO']],text=True);q['provider_symbols']=[x.split()[-1]for x in symbols.splitlines()if ('Embedding'in x or 'RMSNorm'in x)and('forward'in x or 'C1'in x)]
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-norm-CPU-r9037-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
'''
inside='#!/bin/bash\nset -euo pipefail\nulimit -c 0\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_norm_CPU_r9037.py\n';files={'Kimi_norm_native_r9037.cpp':native,'Kimi_norm_CPU_r9037.py':build,'Kimi_norm_CPU_r9037_inside.sh':inside}
for n,s in files.items():assert chr(13)not in s;(b/n).write_bytes(s.encode())
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
remote="""import pathlib,json,zlib,base64,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists(),n;assert chr(13)not in s;p.write_bytes(s.encode())
subprocess.run(['bash','-n',str(i/'Kimi_norm_CPU_r9037_inside.sh')],check=True)
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-norm-CPU-r9037','--output='+str(r/'logs/Kimi-norm-CPU-r9037-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_norm_CPU_r9037_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-norm-CPU-r9037-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args}))
""".replace('PAYLOAD',repr(payload));compile(build,'build','exec');compile(remote,'remote','exec');(r/'invoke_Kimi_norm_CPU_r9037.py').write_bytes(remote.encode());print('source/LF/AST ready')
