import pathlib,json,difflib,base64,zlib
r=pathlib.Path('.');b=r/'Kimi-oproj-canonical-private-r9040';b.mkdir(exist_ok=False);old=(r/'Kimi-delta-selector-private-r9027/original_delta.cpp').read_text();needle='    infinilm::runtime::Tensor forward_prefill512(';idx=old.index(needle);method=r'''    infinilm::runtime::Tensor forward_canonical_r9040(infinilm::runtime::Runtime &runtime,infinilm::runtime::Tensor &input,bool serial)const{
      runtime.assertOwnerThread();if(runtime.isGraphRecording()||tp_size_!=16||!communicator_||input->dtype()!=infinilm::runtime::DataType::kBFloat16||input->ndim()!=3||input->size(0)!=1||input->size(2)!=768||(input->size(1)!=1&&input->size(1)!=8)||has_bias()||alpha()!=1.f||output_dtype()!=input->dtype())throw std::invalid_argument("9040 bounded originalNone TP16 o_proj");
      std::vector<infinilm::runtime::Tensor>held;infinilm::runtime::Tensor output;
      struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      if(serial&&input->size(1)==8){output=infinilm::runtime::Tensor::empty(runtime,{1,8,7168},input->dtype());for(size_t j=0;j<8;++j){auto row=input->narrow({{1,j,1}});held.push_back(row);auto part=compute_linear(runtime,row);held.push_back(part);if(part->shape()!=infinilm::runtime::Shape{1,1,7168})throw std::logic_error("9040 W1 shape");output->narrow({{1,j,1}})->copy_from(runtime,part);}}
      else output=compute_linear(runtime,input);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9040_o_proj_local");
      infinilm::engine::distributed::allreduce_(runtime,output,output,infinicclSum,communicator_);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9040_o_proj_reduced");
      return output;
    }
''';s=old[:idx]+method+old[idx:];needle='        const char *cc=std::getenv("INFINILM_KIMI_CCL512_R8603");';insert=r'''        auto flag9040=[](const char*n){const char*p=std::getenv(n);if(!p||std::string(p)=="0")return false;if(std::string(p)=="1")return true;throw std::invalid_argument("9040 flag0or1");};
        bool selected9040=flag9040("INFINILM_KIMI_OPROJ_W1_R9040"),trace9040=flag9040("INFINILM_KIMI_OPROJ_TRACE_R9040");
        if((selected9040||trace9040)&&batch_size==1&&(seq_len==1||seq_len==8)){
          auto p=std::dynamic_pointer_cast<Prefill512RowR8603>(o_proj_);if(!p)throw std::logic_error("9040 original constructed class");return p->forward_canonical_r9040(runtime,gated,selected9040);
        }
''';assert s.count(needle)==1;s=s.replace(needle,insert+needle);(b/'delta.cpp').write_bytes(s.encode());(b/'delta.patch').write_bytes(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='actual787af_delta.cpp',tofile='delta.cpp')).encode());code=r'''import pathlib,json,hashlib,subprocess,os,traceback
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-oproj-canonical-r9040';b.mkdir(exist_ok=False);(b/'src').mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'changed_HOST_TUs':1,'unchanged_shim_relink':True,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 parent=json.loads((r/'results/Kimi-CCL512-CPU-r8606.json').read_text());cmd=parent['commands'][0][:];src=b/'src/delta.cpp';src.write_bytes((r/'incoming/Kimi_oproj_delta_r9040.cpp').read_bytes());helper=r/'candidates/Kimi-CCL512-r8606/src/kimi_spec_math_r3676.hpp';(src.parent/helper.name).write_bytes(helper.read_bytes());cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(b/'delta.o');cmd[cmd.index('-MF')+1]=str(b/'delta.d');trace=json.loads((r/'candidates/Kimi-target-taps-r9022/CPU-build.json').read_text());link=trace['commands'][-1][:];old='/refactor/candidates/Kimi-CCL512-r8606/delta.o';assert link.count(old)==1;link[link.index(old)]=str(b/'delta.o');main=b/'libKimi_oproj_Main_r9040.so';link[link.index('-o')+1]=str(main)
 for j,a in enumerate(link):
  if a.startswith('-Wl,-Map,')or a.startswith('-Wl,-Map='):link[j]='-Wl,-Map,'+str(b/'link.map')
 shimold=json.loads((r/'results/Kimi-layer0-link-CPU-r9038e-910407.272.json').read_text());shim=shimold['command'][:];shim[shim.index(shimold['actual678_DSO'])]=str(main);native=b/'libKimi_oproj_shim_r9040.so';shim[shim.index('-o')+1]=str(native);q['commands']=[cmd,link,shim]
 for j,c in enumerate(q['commands']):
  p=subprocess.run(c,capture_output=True,text=True,timeout=140);(b/f'command{j}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-6000:]
 reloc=subprocess.check_output(['readelf','-rW',str(native)],text=True);slots={}
 for name in shimold['provider_relocation_slots']:
  rows=[x.split()for x in reloc.splitlines()if name in x and('JUMP_SLOT'in x or'GLOB_DAT'in x)];assert len(rows)==1;slots[name]=int(rows[0][0],16)
 expected={k:(sha(main)if 'KimiK3DeltaAttention'in k else shimold['provider_DSO_SHA256'])for k in slots};needed=[x for x in subprocess.check_output(['readelf','-d',str(native)],text=True).splitlines()if'NEEDED'in x];assert str(main)in needed[0];q.update(completed=True,native=str(native),native_SHA256=sha(native),candidate_Main=str(main),candidate_Main_SHA256=sha(main),actual678_DSO=str(main),actual678_DSO_SHA256=sha(main),provider_DSO=shimold['provider_DSO'],provider_DSO_SHA256=shimold['provider_DSO_SHA256'],provider_relocation_slots=slots,provider_expected_SHA256=expected,delta_source_SHA256=sha(src),delta_object_SHA256=sha(b/'delta.o'),math_header_SHA256=sha(helper),DT_NEEDED=needed)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-oproj-CPU-r9040-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
''';inside='#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_oproj_CPU_r9040.py\n';files={'Kimi_oproj_delta_r9040.cpp':s,'Kimi_oproj_CPU_r9040.py':code,'Kimi_oproj_CPU_r9040_inside.sh':inside}
for n,t in files.items():(b/n).write_bytes(t.encode())
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode();remote="""import pathlib,json,zlib,base64,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists();assert chr(13)not in s;p.write_bytes(s.encode())
subprocess.run(['bash','-n',str(i/'Kimi_oproj_CPU_r9040_inside.sh')],check=True)
a=['srun','--jobid=910407','--overlap','--exact','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-oproj-CPU-r9040','--output='+str(r/'logs/Kimi-oproj-CPU-r9040-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_oproj_CPU_r9040_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-oproj-CPU-r9040-launch.log').open('x')as f:p=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'args':a}))
""".replace('PAYLOAD',repr(payload));compile(code,'cpu','exec');compile(remote,'inv','exec');(r/'invoke_Kimi_oproj_CPU_r9040.py').write_bytes(remote.encode())
