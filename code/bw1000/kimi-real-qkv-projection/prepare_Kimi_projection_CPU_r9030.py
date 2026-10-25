import pathlib,json,zlib,base64
r=pathlib.Path(__file__).resolve().parent;b=r/'Kimi-projection-real-input-private-r9030';b.mkdir(exist_ok=False)
s='''#include "runtime/runtime.hpp"
#include "layers/quantization/none_quantization.hpp"
#include <infini/rt/device_guard.h>
#include <array>
#include <memory>
#include <string>
namespace rt=infinilm::runtime;namespace qt=infinilm::quantization;
namespace{thread_local std::string error;struct C{std::shared_ptr<rt::Runtime>r;rt::Tensor w;qt::ParamsMap p;std::shared_ptr<qt::NoneQuantization>q;std::array<uint64_t,3>counts{};};void drain(C*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}}
extern "C"const char*proj_error(){return error.c_str();}
extern "C"void*proj_create(const void*w){std::unique_ptr<C>c;try{if(!w)throw std::invalid_argument("null weight");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,0,true};c=std::make_unique<C>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,0});c->r->enableOwnerDrainFailStop();c->q=std::make_shared<qt::NoneQuantization>();c->w=rt::Tensor::empty(*c->r,{2304,7168},rt::DataType::kBFloat16);auto h=rt::Tensor::from_blob(const_cast<void*>(w),{2304,7168},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});c->w->copy_from(*c->r,h);c->p.emplace("weight",c->w);drain(c.get());return c.release();}catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C"int proj_run(void*raw,size_t m,const void*x,void*y){auto*c=static_cast<C*>(raw);rt::Tensor host,input,out;try{if(!c||!x||!y||(m!=1&&m!=8))throw std::invalid_argument("M1/M8 only");auto&r=*c->r;r.assertOwnerThread();host=rt::Tensor::from_blob(const_cast<void*>(x),{1,m,7168},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});input=rt::Tensor::empty(r,{1,m,7168},rt::DataType::kBFloat16);input->copy_from(r,host);out=c->q->forward(r,c->p,input,false,1.f);if(out->shape()!=rt::Shape{1,m,2304}||out->dtype()!=rt::DataType::kBFloat16)throw std::logic_error("provider output geometry");r.syncStream();r.memcpyD2H(y,out->data(),out->nbytes());r.syncStream();++c->counts[m==1?0:1];c->counts[2]+=m;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C"int proj_counts(void*raw,uint64_t*out){auto*c=static_cast<C*>(raw);if(!c||!out)return 1;c->r->assertOwnerThread();for(size_t i=0;i<3;++i)out[i]=c->counts[i];return 0;}
extern "C"int proj_close(void*raw){auto*c=static_cast<C*>(raw);if(!c)return 1;drain(c);c->p.clear();c->w={};c->q.reset();drain(c);delete c;return 0;}
'''
(b/'projection_native.cpp').write_bytes(s.encode())
code='''import pathlib,json,hashlib,subprocess,shlex,os,traceback
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-projection-real-input-r9030';b.mkdir(exist_ok=False);i=r/'incoming';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 m=json.loads((r/'candidates/Kimi-qkv-component-r8544/operands.json').read_text());wm=m['weight'];old=pathlib.Path(wm['path'].replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014','/refactor'));raw=old.read_bytes();assert sha(old)==wm['SHA256']=='83a380fdcdb0aeec37f52063da686d92e9e74e881ee606692eb2dcb4a4d02c06';parts=[];proof=[]
 for j,key in enumerate(wm['keys']):
  selected=raw[j*1536*7168*2:j*1536*7168*2+768*7168*2];src=pathlib.Path(key['source']);src=pathlib.Path(str(src).replace('/public/home/aclsylqidf/shared','/shared'))if not src.exists()else src
  with src.open('rb')as f:
   size=int.from_bytes(f.read(8),'little');header=f.read(size);assert hashlib.sha256(header).hexdigest()==key['header_SHA256'];meta=json.loads(header)[key['key']];assert meta['shape']==[12288,7168]and meta['dtype']=='BF16';start=8+size+meta['data_offsets'][0];f.seek(start);actual=f.read(768*7168*2)
  assert selected==actual;parts.append(selected);proof.append({'key':key['key'],'shape':[768,7168],'SHA256':hashlib.sha256(selected).hexdigest(),'actual_checkpoint_byte_range':[start,start+len(actual)],'source':str(src)})
 weight=b/'qkv-TP16-rank0.bf16';weight.write_bytes(b''.join(parts))
 packet=json.loads((r/'results/kimi-crossover-r9026-TP16PP2-I1928669zh-8e319d252bb323d5-node2.json').read_text());rows={};records=[]
 for call in packet['crossover_calls']:
  if call['arm']not in ['A','C']:continue
  for owner in call.get('owners',[]):
   if owner['node']!=0:continue
   for z in owner.get('trace_after',{}).get('records',[]):
    if z['layer']!=0 or z['point']!='attn_norm':continue
    for row,pos in enumerate(z['positions']):
     if pos not in [536,537]:continue
     p=r/z['remote_owned_buffer'].removeprefix('/refactor/');data=p.read_bytes()[row*14336:(row+1)*14336];assert len(data)==14336;key=(call['arm'],pos)
     if key in rows:assert rows[key]==data
     rows[key]=data;records.append({'arm':call['arm'],'position':pos,'path':str(p),'row':row,'SHA256':hashlib.sha256(data).hexdigest()})
 assert rows[('A',536)]==rows[('C',536)]and rows[('A',537)]==rows[('C',537)]
 inp=b/'norm0-536537-plus6zero.bf16';inp.write_bytes(rows[('A',536)]+rows[('A',537)]+bytes(6*14336))
 q['fixture']={'weight':str(weight),'weight_SHA256':sha(weight),'weight_shape':[2304,7168],'checkpoint_slices':proof,'input':str(inp),'input_SHA256':sha(inp),'input_shape':[8,7168],'true_rows':[536,537],'geometry_only_zero_rows':[2,3,4,5,6,7],'records':records,'scope':'QKV-only before KDA; no state/conv/gates/outputprojection/TP reduction, geometry-onlyM8 companions. No finalID or causalModelclaim.'}
 src=b/'projection_native.cpp';src.write_bytes((i/'Kimi_projection_native_r9030.cpp').read_bytes());oldcpu=json.loads((r/'candidates/Kimi-KDA-only-serial-r9020/CPU-build.json').read_text());flags=oldcpu['commands'][0][:oldcpu['commands'][0].index('-shared')];parent=r/'candidates/Kimi-ordinary-pair-r9007/libKimi_pair_r9007.so';assert sha(parent)=='8c79e1c060fca5fa2cceeddef51af50f4605418345983d28a8a28bfc6de75df7';out=b/'libKimi_projection_real_r9030.so';common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';sdk=r/'install/hygon-kimi-5f91ad7-r3672'
 cmd=[*flags,'-shared',str(src),str(parent),'-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfinirt','-linfiniops','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-Wl,--no-undefined','-o',str(out)];q['command']=cmd;p=subprocess.run(cmd,text=True,capture_output=True,timeout=150);(b/'build.log').write_text(p.stdout+p.stderr);q['compile_returncode']=p.returncode;assert p.returncode==0,p.stderr[-6000:];q.update(completed=True,native=str(out),native_SHA256=sha(out),source_SHA256=sha(src),provider_DSO=str(parent),provider_DSO_SHA256=sha(parent),undefined_symbols=subprocess.check_output(['nm','-D','--undefined-only',str(out)],text=True))
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
p=r/'results'/('Kimi-projection-CPU-r9030-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['command','undefined_symbols']}));assert q['completed']
''';compile(code,'build','exec');(b/'CPU_build.py').write_bytes(code.encode());inside='#!/bin/bash\nset -euo pipefail\nulimit -c 0\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_projection_CPU_r9030.py\n';(b/'CPU_inside.sh').write_bytes(inside.encode());payload=base64.b64encode(zlib.compress(json.dumps({'Kimi_projection_native_r9030.cpp':s,'Kimi_projection_CPU_r9030.py':code,'Kimi_projection_CPU_r9030_inside.sh':inside}).encode())).decode()
remote="""import pathlib,json,zlib,base64,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists(),n;p.write_bytes(s.encode())
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-projection-CPU-r9030','--output='+str(r/'logs/Kimi-projection-CPU-r9030-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_projection_CPU_r9030_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-projection-CPU-r9030-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args}))
""".replace('PAYLOAD',repr(payload));(r/'invoke_Kimi_projection_CPU_r9030.py').write_bytes(remote.encode())
