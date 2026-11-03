"""One owned Context, two OFF teacher passes. No HTTP, queue or free generation."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path,PurePosixPath
import socket
import sys
import time
import traceback

from binding_proof import derive,validate_pin,verify_diagnostic_functions
from input_bridge import read_assets,encode

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_tree(root,expected):
    root=Path(root).resolve();mp=root/'SOURCE_MANIFEST.json';assert sha(mp)==expected
    m=json.loads(mp.read_bytes())
    for name,h in m['files_SHA256'].items():
        p=root/name;assert p.resolve().is_relative_to(root)and sha(p)==h,name
    return m


def configure(config,pin,*,require_ready=True):
    validate_pin(pin,require_ready=require_ready)
    assert config['diagnostic_only']is True and config['HTTP']==0 and config['free_generation']==0
    assert (config['TP'],config['PP'],config['rows'],config['context'],config['HC_mode'])==(8,1,1,32768,0)
    assert config['Native_seconds']==2700 and config['modes']==[0,0]
    assert config['max_forwards']==184 and config['capture_outputs']==[82,83]and config['raw_limit']==32*1024**2
    assert config['Native_library']==pin['library']and config['Native_SHA256']==pin['library_SHA256']
    for key in ['model_path','processor_path','Native_library','Common_extension','Common_python_root',
                'public_package_root','parent_root','diagnostic_root','boundary_root','base_provider_root','context_package_root','body_path','teacher_path','output_root']:
        value=config[key];assert type(value)is str and '\\'not in value and PurePosixPath(value).is_absolute(),key
    assert config['public_package_SHA256']=='122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'
    assert config['Common_extension_SHA256']=='6e1a7e7643f84a52ab147cfc3c55eee4816affc6898109dbef31d559feaa86d1'
    if config['gpu_start_epoch']is None:
        assert not require_ready and config['execute_authorized']is False
        return None
    assert type(config['gpu_start_epoch'])in[int,float]and config['gpu_start_epoch']>0
    return config['gpu_start_epoch']+config['Native_seconds']


def module(name,path):
    result=importlib.import_module(name)
    assert Path(result.__file__).resolve()==Path(path).resolve(),('Wrong ordinary module binding',name,result.__file__,str(path))
    return result


def execute(config,pin):
    absolute=configure(config,pin)
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,passed=False,Model_READY=False,quality_PASS=False,performance_PASS=False,
        diagnostic_only=True,HTTP=0,free_generation=0,Native_SHA256=pin['library_SHA256'],
        actual_step=config['actual_step'],gpu_start_epoch=config['gpu_start_epoch'],Native_deadline=absolute,
        TP=8,PP=1,rows=1,context=32768,HC_mode=0,phase='source_and_asset_checks',normal_Native_shutdown=False)
    def save():
        p=root/'entry-report.json';temp=p.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(p)
    def deadline_check(where,minimum=0):
        if time.time()+minimum>=absolute:raise TimeoutError('Original single diagnostic epoch: '+where)
    save()
    try:
        deadline_check('entry',minimum=120)
        assert config['execute_authorized']is True
        assert socket.gethostname().split('.')[0]==config['node']
        assert os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']==config['actual_step']
        parent=Path(config['parent_root']);diag=Path(config['diagnostic_root']);ctxroot=Path(config['context_package_root'])
        parent_manifest=verify_tree(parent,config['parent_manifest_SHA256'])
        for path,h in parent_manifest.get('parents_manifest_SHA256',{}).items():verify_tree(path,h)
        verify_tree(diag,'cf570e363016fac276037184ffdc3bdcd03f3bc3b8896ddae1c1031e0d3b552d');verify_tree(config['boundary_root'],pin['source_manifest_SHA256']);verify_tree(config['public_package_root'],config['public_package_SHA256'])
        proof=json.loads((ctxroot/'CONTEXT_DERIVATION.json').read_bytes())
        assert proof['Native_SHA256']==pin['library_SHA256']and proof['Native_CPU_receipt_SHA256']==pin['CPU_receipt_SHA256']
        for name,v in proof['modules'].items():
            assert sha(ctxroot/'diagnostic_deep_context'/name)==v['private_SHA256']
            assert sha(parent/'public_deep_model'/name)==v['parent_SHA256']
        reference=json.loads((diag/'TEACHER_REFERENCE.json').read_bytes())
        body,teacher,teacher_provenance=read_assets(config['body_path'],config['teacher_path'],reference)
        # Bare imports inside the original diagnostic module are source-pinned
        # and resolved before parent worker imports its unchanged mode helper.
        for path in reversed([str(diag),str(ctxroot),str(parent),config['base_provider_root']]):sys.path.insert(0,path)
        for name in ['model_api','capture_api','quiescent_mode','teacher_runner']:module(name,diag/(name+'.py'))
        model_api=sys.modules['model_api'];old_runner=sys.modules['teacher_runner']
        boundary=Path(config['boundary_root'])
        sys.path.insert(0,str(boundary))
        boundary_api=module('boundary_api',boundary/'boundary_api.py')
        runner=module('run_teacher',boundary/'run_teacher.py')
        assert old_runner.teacher_from_provenance(teacher_provenance,reference)==teacher
        parent_worker=module('public_deep_model.worker',parent/'public_deep_model/worker.py')
        parent_worker.configure_process(config)
        receipt=derive(pin);report['Native_derivation_verified']=True;save()
        deadline_check('Torch/Common preparation',minimum=120)
        torch=parent_worker.torch_setup();safe_open=parent_worker.safe_open_dependency()
        public=Path(config['public_package_root'])/'public-python'
        sys.path.insert(0,config['Common_python_root']);sys.path.insert(0,str(public))
        import infinilm,_infinilm
        assert Path(infinilm.__file__).resolve()==(public/'infinilm/__init__.py').resolve()
        assert Path(_infinilm.__file__).resolve()==Path(config['Common_extension']).resolve()and sha(_infinilm.__file__)==config['Common_extension_SHA256']
        from infinilm.processors import AutoInfinilmProcessor
        processor=AutoInfinilmProcessor.from_pretrained(config['processor_path'])
        prompt,input_proof=encode(body,processor);report['input_bridge']=input_proof;save()
        # Raw IDs stay in the private run output only, never in source/Git.
        (root/'private-input-IDs.json').write_text(json.dumps(dict(prompt_token_ids=prompt,teacher_token_ids=teacher),separators=(',',':'))+'\n')
        loader=module('diagnostic_deep_context.checkpoint_loader',ctxroot/'diagnostic_deep_context/checkpoint_loader.py')
        context_module=module('diagnostic_deep_context.native_context',ctxroot/'diagnostic_deep_context/native_context.py')
        cfg,mapping=loader.read_model_metadata(config['model_path'])
        assert cfg['num_hidden_layers']==43 and cfg['vocab_size']==129280
        deadline_check('Native bind/create/load',minimum=120)
        binding=loader.bind_library(pin['library'],pin['library_SHA256'])
        model_api.bind(binding.lib)
        # This constructor only declares the seven new ctypes signatures.
        boundary_api.Capture(binding.lib,None)
        report['actual_diagnostic_providers']=verify_diagnostic_functions(binding.lib,pin);save()
        context=context_module.NativeContext(binding,receipt,config=config,cfg=cfg,mapping=mapping,
            torch=torch,safe_open=safe_open,report=report,deadline_check=deadline_check)
        with context:
            assert report['loaded_key_count']==67569
            report['phase']='teacher_diagnostic';save()
            import numpy as np
            def progress(row):
                report['diagnostic_progress']={k:row.get(k)for k in ['completed','passed','actual_forward_attempts','completed_forwards','early_stop','raw_budget_used','primary_error']};save()
            diagnostic=runner.run(context,prompt,teacher,root/'typed',np=np,deadline=absolute,reference=reference,save_leaf=progress)
            assert diagnostic['completed']and diagnostic['passed']
            report['diagnostic']=dict(report='typed/report.json',SHA256=sha(root/'typed/report.json'),
                completed=diagnostic['completed'],passed=diagnostic['passed'],early_stop=diagnostic['early_stop'],
                actual_forward_attempts=diagnostic['actual_forward_attempts'],completed_forwards=diagnostic['completed_forwards'])
            report['owner_affinity_after_diagnostic']=context_module.verify_affinity(binding.lib)
            report['GPU_memory_after_diagnostic']=context.floor();report['phase']='normal_context_exit';save()
        assert report['engine_worker_shutdown_completed']and report['normal_Native_shutdown']
        report.update(completed=True,passed=True,phase='diagnostic_context_closed',Model_READY=False);save()
    except BaseException as error:
        report.update(completed=True,passed=False,error=repr(error),traceback=traceback.format_exc())
        try:save()
        except BaseException as persistence:
            if hasattr(error,'add_note'):error.add_note('Primary diagnostic entry persistence: '+repr(persistence))
        raise
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ['config','pin']:p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha',required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    verify_tree(Path(__file__).parent,a.source_sha)
    assert sha(a.config)==a.config_sha and sha(a.pin)==a.pin_sha
    config=json.loads(a.config.read_bytes());pin=json.loads(a.pin.read_bytes())
    validate_pin(pin,require_ready=a.execute)
    configure(config,pin,require_ready=a.execute)
    if not a.execute:
        print(json.dumps(dict(source_schema=True,CPU_READY=pin['CPU_READY'],Native_dlopen=False,Model_READY=False,GPU=False)));return
    execute(config,pin)


if __name__=='__main__':main()
