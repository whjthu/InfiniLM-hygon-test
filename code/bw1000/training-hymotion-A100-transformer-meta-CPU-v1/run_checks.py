"""Original Transformer meta/LoRA/GC structure only; no pipeline or tensor data."""
from pathlib import Path
import ast,copy,hashlib,importlib.metadata as metadata,io,json,math,os,resource,signal,sys,time,traceback
from contextlib import redirect_stdout,redirect_stderr
from typing import Any
W=Path(__file__).resolve().parents[1];S=W/'source';epoch=float(os.environ['META_CPU_EPOCH']);deadline=float(os.environ['META_CPU_DEADLINE']);started=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==180
def expire(*_):raise TimeoutError('original transformer meta CPU epoch exhausted')
signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,deadline-time.time());available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(available[:2]));assert len(os.sched_getaffinity(0))<=2
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),affinity=sorted(os.sched_getaffinity(0)));phase='published-source';result=None
try:
 manifest=json.loads((S/'SOURCE_FILES.json').read_bytes())['files']
 for n,pin in manifest.items():
  p=S/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 binding=json.loads((S/'MODEL_BINDING.json').read_bytes());up=json.loads((S/'UPSTREAM_FILES.json').read_bytes());U=Path(up['root']);assert len(up['files'])==16
 for n,pin in up['files'].items():
  p=U/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 assert sha(Path(binding['model_config_path']))==binding['model_config_SHA256'] and sha(Path(binding['original_backend']))==binding['original_backend_SHA256']
 versions={n:metadata.version(n) for n in binding['expected_versions']};assert versions==binding['expected_versions'];phase='native-meta-import';assert not any(n=='hymotion' or n.startswith('hymotion.') for n in sys.modules)
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def forbid(*_a,**_kw):raise RuntimeError('device context forbidden during Transformer meta qualification')
 torch.cuda._lazy_init=forbid
 from accelerate import init_empty_weights
 sys.path.insert(0,str(U));os.chdir(U)
 from hymotion.utils.loaders import load_object
 functions=['_apply_lora','_apply_gradient_checkpointing','_canonical_adapter_parameters','_adapter_parameter_name'];current=ast.parse((S/'hy_motion_training.py').read_bytes());old=ast.parse(Path(binding['original_backend']).read_bytes());source_nodes={n.name:n for n in current.body if isinstance(n,ast.FunctionDef)};old_nodes={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};ns=dict(Any=Any,math=math)
 for n in functions:
  assert ast.dump(source_nodes[n],include_attributes=False)==ast.dump(old_nodes[n],include_attributes=False);exec(compile(ast.Module(body=[source_nodes[n]],type_ignores=[]),'published_native_helper','exec'),ns)
 phase='real-Transformer-meta-and-LoRA';cfg=json.loads((S/'ORIGINAL_RECIPE.json').read_bytes());mc=binding['model_config'];log=io.StringIO()
 with redirect_stdout(log),redirect_stderr(log),init_empty_weights(include_buffers=False):
  model=load_object(mc['classes']['network_module'],copy.deepcopy(mc['network_module_args']));assert model.motion_input_dim==model.output_dim==201 and model.num_layers==27 and model.feat_dim==1280 and len(model.double_blocks)==9 and len(model.single_blocks)==18;assert all(p.is_meta for p in model.parameters());base_numel=sum(p.numel() for p in model.parameters());model.to(dtype=torch.bfloat16);layers=ns['_apply_lora'](model,torch,rank=cfg['lora_rank'],alpha=cfg['lora_alpha'],target_fragments=cfg['lora_target_modules'])
  def schema():return {n:dict(shape=list(p.shape),dtype=str(p.dtype),numel=p.numel(),meta=p.is_meta) for n,p in ns['_canonical_adapter_parameters'](model).items()}
  before=schema();assert before and len(before)==2*len(layers) and all(v['meta'] and v['dtype']=='torch.float32' for v in before.values());named=dict(model.named_parameters());assert sum(p.requires_grad for p in named.values())==len(before) and not any(p.requires_grad for n,p in named.items() if not (n.endswith('.lora_A') or n.endswith('.lora_B')))
  ns['_apply_gradient_checkpointing'](model);after=schema();assert after==before;trainable=list(ns['_canonical_adapter_parameters'](model).values());optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);assert not optimizer.state and len(optimizer.param_groups)==1 and [id(p) for p in optimizer.param_groups[0]['params']]==[id(p) for p in trainable]
 (W/'model-construction.private.txt').write_text(log.getvalue());phase='runtime-module-source-identity';runtime={};namespaces=[]
 for n,module in sorted(sys.modules.items()):
  if n=='hymotion' or n.startswith('hymotion.'):
   file=getattr(module,'__file__',None)
   if file is None:namespaces.append(n);continue
   p=Path(file).resolve();assert p.is_relative_to(U.resolve());rel=p.relative_to(U.resolve()).as_posix();assert rel in up['files'] and sha(p)==up['files'][rel]['SHA256'];runtime[n]=dict(path=rel,SHA256=sha(p))
 assert not torch.cuda.is_initialized();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert rss<4*1024**3
 result=dict(passed=True,phase='TERMINAL_REAL_NATIVE_TRANSFORMER_META_LORA_GC_CPU_PASS',owner=owner,versions=versions,torch_version=torch.__version__,cuda_version=torch.version.cuda,CUDA_initialized=False,framework_commit=binding['framework_commit'],framework_files=331,actual_framework_subset_files=1,actual_original_helper_ASTs_verified=functions,base_parameters_meta=base_numel,trainable_tensors=len(before),trainable_parameters=sum(x['numel'] for x in before.values()),trainable_dtype='torch.float32',target_fragments=cfg['lora_target_modules'],lora_rank=4,lora_alpha=8,lora_layers=sorted(layers),trainable_manifest=before,GC_canonical_manifest_unchanged=True,base_frozen=True,optimizer_empty_and_exact_binding=True,actual_network_input_dimension=201,actual_network_output_dimension=201,actual_network_layers=27,actual_double_blocks=9,actual_single_blocks=18,runtime_project_modules=runtime,namespace_packages=namespaces,source_files_verified=16,full_pipeline_constructed=False,weights_or_cache_tensor_bytes_read=0,model_values_qualified=False,frame_length_qualified=False,frame_decision_pending=True,GPU_cards=0,forward_backward_updates=0,formal_performance=False,heldout_effect=False,Model_GO=False,native_extension_import_side_effects='Not inferred from CPU-only Torch qualification')
except BaseException as error:
 (W/'failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner,GPU_cards=0,weights_or_cache_tensor_bytes_read=0,forward_backward_updates=0,frame_decision_pending=True,Model_GO=False)
result.update(epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(W/'CPU-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result.get(k) for k in ['passed','phase','error_type','trainable_tensors','trainable_parameters','CUDA_initialized','Model_GO']}))
if not result['passed']:raise SystemExit(1)
