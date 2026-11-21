"""Exercise frozen command and parse_args AST only; no trainer/Torch import."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
from types import SimpleNamespace

from prepare_candidate import SOURCE, SOURCE_PIN, TRAINER, pin, revised

ARCHIVE=('code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz',
         '85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f')
BACKEND='1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7'
DATA_BINDING=('code/bw1000/training-sdxl-USGS-style18-portable-entry-v1/DATA_BINDING.json',
              'bc268682bdfad89dbd3a05a93214fdeffa7415ca3b3d9dc3c8804c9ec9f96c58')


def pinned(repo,item):
    raw=(repo/item[0]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item[1]:raise ValueError('Exact source metadata differs')
    return raw


def parser_only(source):
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
    # Only enum member NAMES are used by this parser's dynamic choices list.
    # This fixture supplies names, not torchvision/PIL/native transformation.
    enum=type('InterpolationModeNamesOnly',(),{n:None for n in ('NEAREST','NEAREST_EXACT','BILINEAR','BICUBIC','BOX','HAMMING','LANCZOS')})
    ns=dict(argparse=argparse,os=SimpleNamespace(environ={}),
            transforms=SimpleNamespace(InterpolationMode=enum))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 'actual_parse_args_AST_only','exec'),ns)
    return ns['parse_args']


def backend_command(source,config,resume=None):
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_sdxl_training')
    def assigned(n,name):
        return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    start=next(i for i,n in enumerate(function.body) if assigned(n,'command'))
    stop=next(i for i,n in enumerate(function.body) if assigned(n,'environment'))
    root=PurePosixPath('/NEW_DOMAIN_SCOPE_NOT_EXECUTED')
    ns=dict(config=config,sys=SimpleNamespace(executable='/PYTHON_NOT_EXECUTED'),
            script=root/'source'/TRAINER,weights=root/'weights',dataset=root/'train',output_dir=root/'out',
            metrics_path=root/'out/steps.ndjson',initial_adapter=root/'out/initial_adapter.safetensors',
            max_steps=config['max_train_steps'],checkpointing_steps=config['checkpointing_steps'],
            checkpoints_total_limit=config['checkpoints_total_limit'],resume_checkpoint=resume)
    exec(compile(ast.fix_missing_locations(ast.Module(body=function.body[start:stop],type_ignores=[])),
                 'actual_common_backend_command_AST_only','exec'),ns)
    return ns['command']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    args.output.mkdir(mode=0o700)
    original=(args.repo/SOURCE).read_bytes();candidate=revised(original)
    deployed=(args.prepared/'candidate-source-root'/TRAINER).read_bytes()
    archive=pinned(args.repo,ARCHIVE)
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as bundle:
        backend=bundle.extractfile('framework/generation/src/gencuda/backends/sdxl_training.py').read()
    if hashlib.sha256(backend).hexdigest()!=BACKEND:raise ValueError('Common backend source differs')
    binding=json.loads(pinned(args.repo,DATA_BINDING));config=binding['path_independent_config']
    command=backend_command(backend,config)
    checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    check('exact_new_private_source_reproducible',deployed==candidate)
    try:
        with contextlib.redirect_stderr(io.StringIO()):parser_only(original)(command[2:])
    except SystemExit as error:rejected=error.code==2
    else:rejected=False
    check('original_cached_source_rejects_actual_common_CLI',rejected)
    parsed=parser_only(candidate)(command[2:])
    check('new_source_accepts_all_actual_fresh_command_flags',
          parsed.genmlu_metrics_path.endswith('/steps.ndjson') and parsed.genmlu_checkpoint_steps=='20,25'
          and parsed.genmlu_initial_adapter_path.endswith('/initial_adapter.safetensors'))
    check('native_recipe_arguments_preserved',
          (parsed.max_train_steps,parsed.resolution,parsed.rank,parsed.train_batch_size,
           parsed.gradient_accumulation_steps,parsed.mixed_precision,parsed.lr_scheduler,
           parsed.lr_warmup_steps,parsed.gradient_checkpointing,parsed.train_text_encoder)
          ==(25,256,4,1,1,'bf16','constant',0,False,False))
    check('imagefolder_columns_local_train_only_no_validation',
          parsed.dataset_name is None and parsed.image_column=='image' and parsed.caption_column=='text'
          and parsed.train_data_dir.endswith('/train') and parsed.validation_prompt is None
          and parsed.random_flip is False and parsed.center_crop is True and parsed.max_train_samples is None)
    # Actual guard comes before Torch/accelerator work in candidate main.
    resume_command=backend_command(backend,config,PurePosixPath('/OLD_SCOPE/checkpoint-20'))
    resumed=parser_only(candidate)(resume_command[2:])
    main_function=next(n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    guard=next(n for n in main_function.body if isinstance(n,ast.If))
    ns={'args':resumed}
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),
                     'actual_external_resume_first_guard_only','exec'),ns)
    except ValueError:resume_rejected=True
    else:resume_rejected=False
    check('actual_guard_refuses_common_backend_external_resume_without_fullcheckpoint',resume_rejected)
    check('existing_data_worker2_and_fullcheckpoint_worker0_conflict_explicit',
          config['dataloader_num_workers']==2 and resumed.genmlu_full_checkpoint is False)
    check('candidate_does_not_claim_A100_original_identity',
          pin(candidate)['SHA256']!='15bf8b928f58d54858f8553ce51da4eaf302928bfd627b5a44da49b66c32b13c')
    result=dict(state='NEW_DOMAIN_SDXL_FRESH_CLI_SOURCE_VERIFIED_NATIVE_IO_PENDING',
                passed=all(x['passed'] for x in checks),checks=checks,source=SOURCE_PIN,candidate=pin(candidate),
                backend=pin(backend),data_binding=DATA_BINDING[1],
                command_SHA256=hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest(),
                parser_interpolation_enum_names_fixture=True,trainer_module_imported=False,
                native_tokenizer_VAE_ImageFolder_executed=False,model_weight_bodies_read=False,
                actual_optimizer_updates=0,HTTP_GETs=0,target_calls=0,training_READY=False,
                warmup_semantics='5 steps measurement exclusion; actual constant scheduler warmup=0',
                target_full_pipeline_weights_manifest='NOT_SUPPLIED_BY_THIS_SCOPE',
                supports_external_resume=False,wall_seconds=time.monotonic()-start)
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    (args.output/'result.json').write_bytes(raw)
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'result':pin(raw)},sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
