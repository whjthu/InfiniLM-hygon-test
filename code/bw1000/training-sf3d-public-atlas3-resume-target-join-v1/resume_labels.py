"""Bind real training labels into native-compatible config hash and metadata checks.

No Torch import, state tensor construction or model-identity fabrication here.
The caller must supply the actual native model identity and device IDs.
"""
import ast
import copy
import hashlib
import json
import tarfile

from target_join3 import G,RECIPE,checked


def load_native_contract_code():
    archive=G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
    checked(archive,{'bytes':738802,'SHA256':'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'})
    with tarfile.open(archive,'r:gz') as t:raw=t.extractfile('framework/generation/src/gencuda/backends/sf3d_training.py').read()
    if hashlib.sha256(raw).hexdigest()!='dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802':raise ValueError('Frozen native backend differs')
    tree=ast.parse(raw);names={'_sf3d_require','_sf3d_json_sha256','_sf3d_training_config_sha256','_sf3d_validate_contract','_sf3d_contract','_sf3d_checkpoint_metadata'}
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.ClassDef) and n.name=='SF3DCheckpointValidationError'
        or isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id in ('SF3D_CHECKPOINT_SCHEMA','_SF3D_IDENTITY_FIELDS') for v in n.targets)]
    if {n.name for n in keep if isinstance(n,ast.FunctionDef)}!=names:raise ValueError('Native contract helper closure differs')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+keep,type_ignores=[])
    ns={'json':json,'hashlib':hashlib};exec(compile(ast.fix_missing_locations(module),'frozen-SF3D-contract-only','exec'),ns)
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sf3d_prepare_checkpoint')
    checks=[copy.deepcopy(n) for n in prepare.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_sf3d_require'
        and ('metadata.get(\'contract\') == contract' in ast.unparse(n.value.args[0]) or '_sf3d_checkpoint_metadata(contract' in ast.unparse(n.value.args[0]))]
    if len(checks)!=2:raise ValueError('Exact original metadata compatibility closure differs')
    ns['__metadata_checks__']=compile(ast.fix_missing_locations(ast.Module(body=checks,type_ignores=[])),'original-SF3D-prepare-checkpoint-metadata-clauses','exec')
    ns['__proof__']={'backend_SHA256':hashlib.sha256(raw).hexdigest(),'functions':{n.name:{'line':n.lineno,'end_line':n.end_lineno,'AST_SHA256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()} for n in keep if isinstance(n,ast.FunctionDef)},
        'metadata_clause_lines':[n.lineno for n in checks],'full_native_state_tensor_or_rng_validator_loaded':False}
    return ns


def training_label_identity(b):
    # Ignore development/blind rows, filesystem spelling and physical host root.
    # Preserve optimizer data order and each real object's complete target body.
    rows=[{'group':r['group'],'input':r['input_pin'],'target':r['target_pin']} for r in b['train']]
    if len(rows)!=2 or len({r['group'] for r in rows})!=2 or len({r['input']['SHA256'] for r in rows})!=2:raise ValueError('Two distinct true training inputs required')
    document={'schema':'SF3D_PUBLIC_ATLAS_TRAIN_LABEL_INPUT_ID_V1','recipe':RECIPE,'train_ordered':rows}
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),document


def domain_contract(native,b,config,*,actual_native_identity,actual_device_ids):
    # No defaults for actual model/device identity: unresolved runtime remains
    # unresolved. CPU tests explicitly mark their synthetic metadata identity.
    labels,document=training_label_identity(b)
    base=native['_sf3d_training_config_sha256'](config)
    combined=native['_sf3d_json_sha256']({'schema':'SF3D_ATLAS_CONFIG_WITH_TRAIN_LABELS_V1','native_config_sha256':base,'training_label_input_SHA256':labels})
    inputs=native['_sf3d_json_sha256']([{'sha256':r['input_pin']['SHA256']} for r in b['train']])
    contract=native['_sf3d_contract'](identity=actual_native_identity,config_sha256=combined,input_sha256=inputs,
        seed=int(config.get('seed',20260901)),warmup_steps=int(config.get('warmup_steps',0)),max_steps=int(config.get('max_steps',20)),batch_count=len(b['train']),device_ids=actual_device_ids)
    return contract,{'native_base_config_SHA256':base,'domain_config_SHA256':combined,'training_label_input_SHA256':labels,'native_ordered_input_SHA256':inputs,'label_document':document}


def check_resume_metadata(native,metadata,expected_contract):
    native['_sf3d_validate_contract'](expected_contract)
    native['_sf3d_validate_contract'](metadata['contract'])
    ns={**native,'metadata':metadata,'contract':expected_contract}
    exec(native['__metadata_checks__'],ns)
    return True
