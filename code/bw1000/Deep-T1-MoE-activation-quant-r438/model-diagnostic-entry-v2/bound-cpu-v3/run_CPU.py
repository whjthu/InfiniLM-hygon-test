"""One real source/CPU-graph/processor handoff. No Native/Runtime execution."""
from pathlib import Path
import ctypes as C
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import binding_proof
import entry
import input_bridge
import stage_context

D=Path(__file__).resolve().parent
B=Path('/refactor/candidates/Deep438-diagnostic-entry-CPU-v3')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    B.mkdir(exist_ok=False);started=time.monotonic()
    q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
           Native_dlopen=False,Runtime_created=False,weights_payload_opened=False,GPU=False,Model_READY=False,
           verified_source_manifests={},stages=[],weight_open_attempts=[],project_dlopen_attempts=[])
    def save():
        p=B/'result.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(p)
    def verified(root,h):
        value=entry.verify_tree(root,h);q['verified_source_manifests'][str(Path(root)/'SOURCE_MANIFEST.json')]=h;return value
    def observe(stage):
        lines=Path('/proc/self/maps').read_text().splitlines()
        paths=sorted({line.split()[-1]for line in lines if len(line.split())>=6 and line.split()[-1].startswith('/')})
        forbidden=['libDeep_','libGLM_','libKimi_','_infinilm.cpython','libinfinirt.so','libinfinilm_runtime','libinfiniops.so','libinfinicore.so','libinfiniccl.so']
        project=[p for p in paths if p.startswith('/refactor/')or any(s in p for s in forbidden)]
        torch=sys.modules.get('torch');initialized=None if torch is None else bool(torch.cuda.is_initialized())
        q['stages'].append(dict(stage=stage,project_maps=project,Torch_imported=torch is not None,
            torch_cuda_Python_initialized=initialized,framework_modules={k:getattr(sys.modules.get(k),'__file__',None)for k in ['torch','transformers','tokenizers','numpy']}));save()
        assert not project and not initialized,('Forbidden project/device initialization',project,initialized)
    def audit(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0])
            if path.lower().endswith(('.safetensors','.bin','.pt','.pth')):
                q['weight_open_attempts'].append(path);raise RuntimeError('Model payload forbidden in entry CPU')
        if event=='ctypes.dlopen'and args and args[0]is not None:
            path=os.fsdecode(args[0])
            if path.startswith('/refactor/')or any(v in path for v in ['libDeep_','_infinilm','libinfini']):
                q['project_dlopen_attempts'].append(path);raise RuntimeError('Native/Runtime/SDK dlopen forbidden')
    save()
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        assert sha(D/'SOURCE_MANIFEST.json')==os.environ['DEEP438_ENTRY_CPU_SOURCE_SHA256']
        m=verified(D,os.environ['DEEP438_ENTRY_CPU_SOURCE_SHA256'])
        cfg=json.loads((D/'config.template.json').read_bytes());pin=json.loads((D/'BINDING_PIN.json').read_bytes())
        plan=json.loads((D/'CPU_PLAN.json').read_bytes())
        assert cfg['execute_authorized']is False and cfg['gpu_start_epoch']is None
        entry.configure(cfg,pin,require_ready=False)
        parent=Path(cfg['parent_root']);capture=Path(cfg['diagnostic_root'])
        original=verified(D.parent,plan['original_entry_manifest_SHA256'])
        for name in ['entry.py','binding_proof.py','input_bridge.py','stage_context.py']:
            assert sha(D/name)==original['files_SHA256'][name]
        oldm=verified(parent,cfg['parent_manifest_SHA256'])
        for path,h in oldm['parents_manifest_SHA256'].items():verified(path,h)
        verified(capture,pin['source_manifest_SHA256'])
        old_cpu=Path(plan['processor_CPU_root']);verified(old_cpu,plan['processor_CPU_manifest_SHA256'])
        controller=Path(plan['controller_root']);cm=verified(controller,plan['controller_manifest_SHA256'])
        for name in cm['files_SHA256']:
            p=controller/name
            if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
            if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True,timeout=10)
        source_only=json.loads(subprocess.check_output([sys.executable,'-B',str(controller/'controller.py'),
            '--manifest-sha',plan['controller_manifest_SHA256'],'--publication-commit',plan['controller_publication'],
            '--source-only'],text=True,timeout=25))
        assert source_only==dict(source_verified=True,Slurm=0,GPU=0,Native_dlopen=False)
        q['controller_source_only']=source_only
        for path,h in plan['metadata_assets_SHA256'].items():assert sha(path)==h,path
        sys.addaudithook(audit);sys.path[:0]=[str(capture),str(parent),cfg['base_provider_root']]
        observe('before_derivation')
        receipt=binding_proof.derive(pin)
        q['actual_Native_derivation']=True;q['actual_Native_derivation_receipt']=dict(
            library=receipt['library'],SHA256=receipt['library_SHA256'],CPU_receipt_SHA256=pin['CPU_receipt_SHA256'],
            objects=49,reused_parent_objects=47,new_HOST=2)
        proof=stage_context.stage(parent,cfg['parent_manifest_SHA256'],pin,B/'context')
        q['actual_static_context_derivation']=proof
        cfg=dict(cfg,context_package_root=str(B/'context'))
        (B/'bound-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (B/'BINDING_PIN.json').write_bytes((D/'BINDING_PIN.json').read_bytes())
        sys.path.insert(0,str(B/'context'))
        loader=entry.module('diagnostic_deep_context.checkpoint_loader',B/'context/diagnostic_deep_context/checkpoint_loader.py')
        context_module=entry.module('diagnostic_deep_context.native_context',B/'context/diagnostic_deep_context/native_context.py')
        assert loader.NATIVE_SHA==pin['library_SHA256']and context_module.NATIVE_SHA==pin['library_SHA256']
        cfg_read,mapping=loader.read_model_metadata(cfg['model_path'])
        assert cfg_read['num_hidden_layers']==43 and cfg_read['vocab_size']==129280 and len(mapping)>=67569
        # Exact old + additive declaration bodies, static actual DSO symbol
        # set, then real libffi high64-bit ctx callback. No C++ DSO is loaded.
        class Schema:
            def __init__(self):self.values={}
            def __getattr__(self,name):return self.values.setdefault(name,SimpleNamespace())
        table=Schema();loader.declare(table)
        model_api=entry.module('model_api',capture/'model_api.py');capture_api=entry.module('capture_api',capture/'capture_api.py')
        model_api.bind(table);capture_api.bind(table)
        names={line.split()[-1]for line in subprocess.check_output(['nm','-D','--defined-only',pin['library']],text=True,timeout=25).splitlines()if len(line.split())>=3}
        assert set(table.values)<=names
        HIGH=0x123456789abcdef0;seen=[];keep=[]
        def reset(ctx):assert ctx==HIGH;seen.append(ctx);return 0
        fn=C.CFUNCTYPE(C.c_int,C.c_void_p)(reset);keep.append(fn);table.values['modern_native_engine_reset']=fn
        binding=loader.declare(table);assert table.modern_native_engine_reset(HIGH)==0 and seen==[HIGH]
        assert table.modern_native_engine_forward.argtypes[0]is C.c_void_p and table.modern_native_engine_forward.restype is C.c_int
        for name in binding_proof.CAPTURE_NAMES+binding_proof.MODE_NAMES:assert table.values[name].argtypes is not None
        py_context=context_module.NativeContext(binding,receipt,config=cfg,cfg=cfg_read,mapping=mapping,
            torch=None,safe_open=None,report={},deadline_check=lambda *a,**kw:None)
        assert py_context.ctx is None and py_context.thread is None and py_context.closed is False
        q['actual_typed_declarations']=dict(symbols=len(table.values),actual_DSO_exports_checked=True,
            reset_libffi_high_ctx=hex(HIGH),Native_leaf='One libffi callback only; no actual CAPI invocation',
            actual_private_Python_context_init=True,Native_create_called=False)
        observe('derived_context_and_typed_declarations')
        reference=json.loads((capture/'TEACHER_REFERENCE.json').read_bytes())
        body,teacher,provenance=input_bridge.read_assets(cfg['body_path'],cfg['teacher_path'],reference)
        runner=entry.module('teacher_runner',capture/'teacher_runner.py')
        assert runner.teacher_from_provenance(provenance,reference)==teacher
        spec=importlib.util.spec_from_file_location('qualified_430_actual_processor_CPU',old_cpu/'CPU_entry.py')
        actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
        public=Path(cfg['public_package_root'])/'public-python'
        _,_,history=actual.pure_public(public)
        processor,method_sources=actual.actual_processor(public,Path(cfg['processor_path']),history)
        prompt,input_proof=input_bridge.encode(body,processor)
        assert len(prompt)==8489 and input_bridge.ids_sha(prompt)=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
        schedule=runner.plan(prompt,teacher)
        assert len(schedule)==92 and [s['past']for s in schedule if s['output_index']in[82,83]]==[8570,8571]
        q['actual_input_processor']=True;q['actual_input']=dict(input_proof,processor_method_sources=method_sources,
            tokenizer_class=type(processor.get_tokenizer()).__module__+'.'+type(processor.get_tokenizer()).__qualname__,
            CPU_scaffolding='Reuses qualified430 CPU package-init exclusion and exact public Deep initializer/template/Basic call bodies; actual checkpoint tokenizer, no mocked tokenizer or Native scheduling')
        observe('real_full_body_processor_encoded')
        assert not q['weight_open_attempts']and not q['project_dlopen_attempts']
        q.update(passed=True,binding_pin_SHA256=sha(B/'BINDING_PIN.json'),entry_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),
            bound_config=dict(path=str(B/'bound-config.json'),SHA256=sha(B/'bound-config.json')),
            bound_pin=dict(path=str(B/'BINDING_PIN.json'),SHA256=sha(B/'BINDING_PIN.json')),
            context_derivation=dict(path=str(B/'context/CONTEXT_DERIVATION.json'),SHA256=sha(B/'context/CONTEXT_DERIVATION.json')),
            old_suites_replayed=False,actual_capture_or_teacher_forwards=0)
    except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,wall_s=time.monotonic()-started);save()
        if q['passed']:
            compact={k:q[k]for k in ['passed','actual_CPU','actual_input_processor','actual_Native_derivation',
                'Native_dlopen','GPU','Model_READY','binding_pin_SHA256','entry_manifest_SHA256','verified_source_manifests']}
            compact.update(bound_config=q['bound_config'],bound_pin=q['bound_pin'],context_derivation=q['context_derivation'],
                original_CPU_receipt=dict(path=str(B/'result.json'),SHA256=sha(B/'result.json')))
            (B/'ENTRY_CPU.json').write_text(json.dumps(compact,indent=2)+'\n')
        print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','error','wall_s']}),flush=True)
    return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
