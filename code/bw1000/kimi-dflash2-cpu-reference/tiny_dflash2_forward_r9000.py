"""Execute reviewed creator classes, CPU SDPA backend, independent NumPy forward.
Synthetic tiny F32 parameters only. No CUDA compilation, checkpoint values or GPU.
"""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('ROCR_VISIBLE_DEVICES')==''
import pathlib,json,hashlib,ast,math,types,sys,typing,functools,__future__,traceback,time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
r=pathlib.Path('/refactor');inc=r/'incoming';started=time.monotonic();out=r/'results/DFlash2-tiny-CPU-r9000.json';d={'completed':False,'CPU_only':True,'GPU':False,'trained_weight_values_read':False,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def selected(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names];assert {n.name for n in nodes}==set(names)
 class Eager(ast.NodeTransformer):
  def visit_FunctionDef(self,n):
   n.decorator_list=[x for x in n.decorator_list if not isinstance(x,ast.Call)or ast.unparse(x.func)!='torch.compile'];return self.generic_visit(n)
 tree=Eager().visit(ast.Module(body=nodes,type_ignores=[]));ast.fix_missing_locations(tree);exec(compile(tree,str(path),'exec',flags=__future__.annotations.compiler_flag),env)
def rms(x,w):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-5))*w
def lin(x,w):return x@w.T
def yarn_np(dim,positions):
 f=np.float32(32);base=np.float32(50000);a=np.arange(0,dim,2,dtype=np.float32)/np.float32(dim);extra=1/(base**a);inter=1/(f*(base**a))
 corr=lambda rot:dim*math.log(32768/(rot*2*math.pi))/(2*math.log(50000))
 low=max(math.floor(corr(32)),0);high=min(math.ceil(corr(1)),dim-1)
 if low==high:high+=.001
 ramp=np.clip((np.arange(dim//2,dtype=np.float32)-low)/(high-low),0,1);inv=inter*ramp+extra*(1-ramp)
 phase=np.asarray(positions,np.float32)[:,None]*inv[None,:];return np.repeat(np.cos(phase),2,-1),np.repeat(np.sin(phase),2,-1),inv
def rotate_np(x,positions):
 co,si,_=yarn_np(x.shape[-1],positions);rot=np.stack((-x[...,1::2],x[...,::2]),-1).reshape(x.shape);return x*co[None,None]+rot*si[None,None]
def conv_np(x,dyn,base,side):
 b,t,h=x.shape;g=h//16;xx=x.reshape(b,t,g,16);dyn=dyn.reshape(b,t,2,g);result=np.zeros_like(xx)
 for off in range(2):
  shifted=np.zeros_like(xx)
  if off==0:shifted=xx
  else:shifted[:,off:]=xx[:,:-off]
  coefficient=base[side,off].reshape(1,1,g,16)+dyn[:,:,off,:,None];result=result+coefficient*shifted
 return result.reshape(x.shape)
def prepare_np(x,w,prefix):
 dyn=lin(x,w[prefix+'.kernel_projection.weight']).reshape(1,8,2,2,2);return conv_np(x,dyn[:,:,0],w[prefix+'.base_kernel'],0),dyn[:,:,1]
def attn_np(x,ctx,w,prefix,mask,dp,cp):
 q=lin(rms(lin(x,w[prefix+'.q_a_proj.weight']),w[prefix+'.q_a_layernorm.weight']),w[prefix+'.q_b_proj.weight']).reshape(1,8,2,16).transpose(0,2,1,3)
 combined=lin(np.concatenate((ctx,x),1),w[prefix+'.kv_a_proj_with_mqa.weight']);latent,krope=combined[...,:12],combined[...,12:]
 kv=lin(rms(latent,w[prefix+'.kv_a_layernorm.weight']),w[prefix+'.kv_b_proj.weight']).reshape(1,-1,2,16).transpose(0,2,1,3);kn,v=kv[...,:8],kv[...,8:]
 qr=rotate_np(q[...,8:],dp);kr=rotate_np(krope[:,None],np.concatenate((cp,dp)));query=np.concatenate((q[...,:8],qr),-1);key=np.concatenate((kn,np.broadcast_to(kr,kn.shape)),-1)
 scale=np.float32((1+.1*math.log(32))**2/math.sqrt(16));scores=(query@key.transpose(0,1,3,2))*scale
 if mask is not None:scores=np.where(mask,scores,-np.inf)
 scores=scores-np.max(scores,axis=-1,keepdims=True);prob=np.exp(scores);prob=prob/np.sum(prob,axis=-1,keepdims=True,dtype=np.float32)
 z=(prob@v).transpose(0,2,1,3).reshape(1,8,16);return lin(z,w[prefix+'.o_proj.weight'])
def layer_np(x,ctx,w,prefix,mask,dp,cp):
 y=rms(x,w[prefix+'.input_layernorm.weight']);y,k=prepare_np(y,w,prefix+'.attention_conv');y=attn_np(y,ctx,w,prefix+'.self_attn',mask,dp,cp);x=x+conv_np(y,k,w[prefix+'.attention_conv.base_kernel'],1)
 y=rms(x,w[prefix+'.post_attention_layernorm.weight']);y,k=prepare_np(y,w,prefix+'.mlp_conv');gate=lin(y,w[prefix+'.mlp.gate_proj.weight']);up=lin(y,w[prefix+'.mlp.up_proj.weight']);y=lin((gate/(1+np.exp(-gate)))*up,w[prefix+'.mlp.down_proj.weight']);return x+conv_np(y,k,w[prefix+'.mlp_conv.base_kernel'],1)
def err(a,b):
 a=np.asarray(a,np.float32);b=np.asarray(b,np.float32);return {'finite':bool(np.isfinite(a).all()and np.isfinite(b).all()),'maxabs':float(np.abs(a-b).max()),'relativeRMS':float(np.sqrt(np.mean((a-b)**2,dtype=np.float32))/max(1e-30,float(np.sqrt(np.mean(b*b,dtype=np.float32)))))}
try:
 proof=json.loads((inc/'DFlash2-CPU-source-r9000.json').read_text());assert all(sha(inc/n)==v for n,v in proof['source_SHA256'].items());env=dict(torch=torch,nn=nn,F=F,math=math,partial=functools.partial,Optional=typing.Optional,Tuple=typing.Tuple,PretrainedConfig=types.SimpleNamespace,DFlashConfig=types.SimpleNamespace,DFlash2Config=types.SimpleNamespace,K3DSparkConfig=types.SimpleNamespace,resolve_rope_theta=lambda c:c.rope_parameters['rope_theta'])
 selected(inc/'r9000_torchspec_llama3_eagle.py',['LlamaRotaryEmbedding','LlamaYarnRotaryEmbedding','yarn_find_correction_dim','yarn_find_correction_range','yarn_get_mscale','yarn_linear_ramp_mask','rope_config_get','build_rotary_embedding'],env)
 selected(inc/'r9000_torchspec_deepseek_eagle.py',['_rotate_half_interleaved','DeepSeekMLAAttention'],env)
 selected(inc/'r9000_torchspec_dflash.py',['DFlashRMSNorm','DFlashMLP'],env)
 def cpu_flex(*,query,key,value,block_mask,scale):return F.scaled_dot_product_attention(query,key,value,attn_mask=block_mask,dropout_p=0.,is_causal=False,scale=scale)
 env['compile_friendly_flex_attention']=cpu_flex;selected(inc/'r9000_torchspec_dspark.py',['_apply_rope_interleaved','K3DSparkMLAAttention'],env);env['DFlashAttention']=env['K3DSparkMLAAttention'];selected(inc/'r9000_torchspec_dflash.py',['DFlashDecoderLayer'],env)
 for name in ['torchspec','torchspec.models','torchspec.models.draft','torchspec.models.draft.dspark']:sys.modules[name]=types.ModuleType(name)
 sys.modules['torchspec.models.draft.dspark'].K3DSparkMLAAttention=env['K3DSparkMLAAttention'];selected(inc/'r9000_torchspec_dflash2.py',['DFlashGroupedConv','CandidateSelector','DFlash2DecoderLayer'],env)
 rope=dict(rope_type='yarn',factor=32.,original_max_position_embeddings=32768,rope_theta=50000.,beta_fast=32,beta_slow=1,mscale=1.,mscale_all_dim=1.)
 cfg=types.SimpleNamespace(hidden_size=32,intermediate_size=64,num_attention_heads=2,q_lora_rank=16,kv_lora_rank=12,qk_nope_head_dim=8,qk_rope_head_dim=8,v_head_dim=8,mla_use_output_gate=False,max_position_embeddings=8192,rope_scaling=rope,rope_parameters=rope,rms_norm_eps=1e-5,block_size=8,conv_kernel_size=2,conv_group_size=16,attention_mode='mla')
 class Tiny(nn.Module):
  def __init__(self):
   super().__init__();self.embed_tokens=nn.Embedding(32,32);self.fc=nn.Linear(160,32,bias=False);self.hidden_norm=env['DFlashRMSNorm'](32,1e-5);self.norm=env['DFlashRMSNorm'](32,1e-5);self.layers=nn.ModuleList([env['DFlash2DecoderLayer'](cfg)for _ in range(5)]);self.candidate_selector=env['CandidateSelector'](32,32,256,16)
 model=Tiny().eval();rng=np.random.default_rng(9000)
 with torch.no_grad():
  for name,param in model.named_parameters():
   value=rng.normal(0,.035,size=tuple(param.shape)).astype(np.float32)
   if name.endswith('norm.weight')or'layernorm.weight'in name:value+=1
   if name.endswith('base_kernel'):value*=.05;value[:,0]+=1
   param.copy_(torch.from_numpy(value))
 header=json.loads((inc/'actual-safetensors-header.json').read_text());actual={k:v for k,v in header.items()if k!='__metadata__'};assert set(model.state_dict())==set(actual)and len(actual)==87
 # Actual dimensional schema checked independently, without meta allocating 3.69B params.
 actual_shapes={}
 for key in actual:
  suffix=key.split('self_attn.')[-1]
  known={'q_a_proj.weight':[1536,7168],'q_a_layernorm.weight':[1536],'q_b_proj.weight':[12288,1536],'kv_a_proj_with_mqa.weight':[576,7168],'kv_a_layernorm.weight':[512],'kv_b_proj.weight':[16384,512],'o_proj.weight':[7168,8192]}
  if 'self_attn.'in key:expected=known[suffix]
  elif key=='fc.weight':expected=[7168,35840]
  elif key=='embed_tokens.weight':expected=[163840,7168]
  elif key=='candidate_selector.hidden_projection.weight':expected=[256,7168]
  elif key.startswith('candidate_selector.'):expected=[163840,256]
  elif key.endswith('base_kernel'):expected=[2,2,7168]
  elif key.endswith('kernel_projection.weight'):expected=[1792,7168]
  elif key.endswith('mlp.down_proj.weight'):expected=[7168,14336]
  elif key.endswith(('mlp.gate_proj.weight','mlp.up_proj.weight')):expected=[14336,7168]
  else:expected=[7168]
  assert actual[key]['shape']==expected and actual[key]['dtype']=='BF16',(key,actual[key],expected);actual_shapes[key]=expected
 weights={k:v.detach().numpy().copy()for k,v in model.state_dict().items()};raw=rng.normal(0,.3,size=(1,4097,160)).astype(np.float32);tokens=np.array([[3,4,4,4,4,4,4,4]],np.int64);cp=np.arange(4097,dtype=np.int64);dp=np.arange(4097,4105,dtype=np.int64);fullpos=np.arange(4105);mask=(np.abs(dp[:,None]-fullpos[None,:])<4096)[None,None]
 ctx_np=rms(lin(raw,weights['fc.weight']),weights['hidden_norm.weight']);x_np=weights['embed_tokens.weight'][tokens];stage=[]
 with torch.inference_mode():
  raw_t=torch.from_numpy(raw.copy());ctx=model.hidden_norm(model.fc(raw_t));x=model.embed_tokens(torch.from_numpy(tokens));cp_t=torch.from_numpy(cp[None]);dp_t=torch.from_numpy(dp[None]);mask_t=torch.from_numpy(mask);stage.append({'stage':'context','error':err(ctx.numpy(),ctx_np)})
  for i,layer in enumerate(model.layers):
   active_mask=mask if i<4 else None;x=layer(x,ctx,dp_t,cp_t,torch.from_numpy(active_mask)if active_mask is not None else None);x_np=layer_np(x_np,ctx_np,weights,f'layers.{i}',active_mask,dp,cp);stage.append({'stage':f'layer{i}','error':err(x.numpy(),x_np)})
  final=model.norm(x);final_np=rms(x_np,weights['norm.weight']);stage.append({'stage':'final','error':err(final.numpy(),final_np)})
  # Canonical actual creator YaRN vs independent trig/frequency calculation.
  rotary=model.layers[0].self_attn.rotary_emb;_,_,inv=yarn_np(8,dp);yarn_error=err(rotary.inv_freq.numpy(),inv);co,si=rotary(torch.empty(1),seq_len=4105);nco,nsi,_=yarn_np(8,dp);interleaved=np.repeat(co.numpy()[0,0,dp,:4],2,-1);trig_error=err(interleaved,nco)
  # Source selector, applied sequentially over only the seven proposal rows.
  lm=rng.normal(0,.05,size=(32,32)).astype(np.float32);logits=F.linear(final[:,1:],torch.from_numpy(lm));np_logits=lin(final_np[:,1:],lm);prev=torch.tensor([3]);prev_np=np.array([3]);paths=[];np_paths=[]
  for j in range(7):
   scores,ids=model.candidate_selector.score_candidates(final[:,j+1],logits[:,j],prev);prev=ids.gather(-1,scores.argmax(-1,keepdim=True))[:,0];paths.append(int(prev.item()))
   ids_np=np.argsort(-np_logits[:,j],axis=-1)[...,:16];unary=np.take_along_axis(np_logits[:,j],ids_np,-1);latent=lin(final_np[:,j+1],weights['candidate_selector.hidden_projection.weight']);score_np=unary+np.sum((weights['candidate_selector.predecessor_codebook'][prev_np]*latent)[:,None,:]*weights['candidate_selector.successor_codebook'][ids_np],-1);prev_np=ids_np[np.arange(1),score_np.argmax(-1)];np_paths.append(int(prev_np[0]))
  # Attention-only poison: excluded old context cannot affect SWA, must affect full.
  xx=model.embed_tokens(torch.from_numpy(tokens));ct=ctx.clone();ct[:,0]+=torch.linspace(-20,20,32);attention=model.layers[0].self_attn;swa_a=attention(xx,ctx,dp_t,cp_t,mask_t);swa_b=attention(xx,ct,dp_t,cp_t,mask_t);full_a=attention(xx,ctx,dp_t,cp_t,None);full_b=attention(xx,ct,dp_t,cp_t,None);poison={'SWA_maxabs':float((swa_a-swa_b).abs().max()),'full_maxabs':float((full_a-full_b).abs().max())};assert poison['SWA_maxabs']==0 and poison['full_maxabs']>0
 assert all(v['error']['finite']and v['error']['relativeRMS']<.002 for v in stage),stage
 assert paths==np_paths and yarn_error['maxabs']<1e-6 and trig_error['maxabs']<1e-3
 assert not torch.cuda.is_initialized();assert np.array_equal(raw_t.numpy(),raw),'input mutation'
 d.update(completed=True,source_SHA256=proof['source_SHA256'],torch_version=torch.__version__,actual_header_key_count=87,actual_header_parameter_count=3692877824,actual_header_shapes_all_matched=True,actual_header_contains_embedding=True,actual_header_contains_lm_head=False,tiny={'H':32,'I':64,'heads':2,'q_lora':16,'kv_lora':12,'nope':8,'rope':8,'v':8,'layers':5,'layer_types':['SWA4096']*4+['full'],'context_tokens':4097,'draft_queries':8,'proposal_rows':list(range(1,8)),'selector_rank':256,'selector_top_k':16,'dtype':'F32 synthetic, not checkpoint BF16values'},stage_errors=stage,YaRN_frequency_error=yarn_error,YaRN_trig_error=trig_error,selector_paths={'actual_creator':paths,'independent_NumPy':np_paths},mask_poison=poison,input_canary=True,GPU_initialized=torch.cuda.is_initialized(),backend='ActualcreatorAST classes eagerCPU; torch.compile decorators removed; CPU SDPA with identicalboolmask replaces flex CUDAkernel; independentNumPy fullforward. Nottrainedweights/acceptance/performance.',remaining_GPU_interfaces=['Efficient incremental MLAcontextKV/cache+mixed4SWA/full attention backend onBW1000 notimplemented bythisCPUfixture','Fullcheckpoint load/embedded targetcompatibility andsharedLMhead values nottested','New5tap targetfeatures andtargetMath0 verifier handledseparately; noKDAchanges here'])
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 d['elapsed_s']=time.monotonic()-started;assert not out.exists();out.write_bytes((json.dumps(d,indent=2)+'\n').encode());print(json.dumps({k:d.get(k)for k in ['completed','step','error','elapsed_s','stage_errors','YaRN_trig_error','selector_paths','mask_poison']}))
