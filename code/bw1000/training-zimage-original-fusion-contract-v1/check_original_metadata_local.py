"""Exercise the unified metadata checker against the actual original headers."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys,time
D=Path(__file__).resolve().parent;F=D.parent/'hygon-training-integration'
sys.path.insert(0,str(F/'generation/src'))
from gencuda.zimage_fusion_contract import adapter_pairs,check_metadata,validate_header
original=json.loads((D/'original-metadata.private.json').read_bytes())
shards=[];adapter=None;base_headers={};adapter_pin=None
for name,h in original['headers'].items():
    validate_header(h['tensors'],file_bytes=h['bytes'],header_bytes=h['header_bytes'])
    pin={k:h[k] for k in ('bytes','header_bytes','header_SHA256')}
    if Path(name).name.startswith('diffusion_pytorch_model-'):
        shards.append(h['tensors']);base_headers[Path(name).name]=pin
    else:adapter=h['tensors'];adapter_pin=pin
assert len(shards)==3 and adapter is not None
observed=check_metadata(shards,adapter)
expected=dict(base_tensors=521,preset_tensors=480,preset_pairs=240,preset_elements=85032960,
              preset_ranks=[32],new_rank4_modules=238,new_rank4_tensors=476,new_rank4_elements=9922560)
assert all(observed[k]==v for k,v in expected.items())
loader_path=D/'original-source/diffsynth/utils/lora/general.py'
raw=loader_path.read_bytes();assert hashlib.sha256(raw).hexdigest()=='8fc0cc07a7730eadbdd787df68cce6ae9468169f6306030337db1ffd5bf41f19'
tree=ast.parse(raw);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GeneralLoRALoader')
cls.body=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('get_name_dict','convert_state_dict')]
assert len(cls.body)==2
module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));namespace={}
exec(compile(module,str(loader_path),'exec'),namespace)
native=namespace['GeneralLoRALoader']();mapping=native.get_name_dict(adapter);converted=native.convert_state_dict(adapter)
ours=adapter_pairs(adapter)
assert set(mapping)==set(ours)
assert all(mapping[n]==(b,a) for n,(a,b) in ours.items())
assert len(converted)==480
sources={}
for name in ['examples/z_image/model_training/train.py','diffsynth/pipelines/z_image.py',
             'diffsynth/models/z_image_dit.py','diffsynth/diffusion/training_module.py',
             'diffsynth/diffusion/base_pipeline.py','diffsynth/utils/lora/general.py']:
    p=D/'original-source'/name;b=p.read_bytes()
    sources[name]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
contract=dict(schema='zimage-original-fusion-metadata-v1',source_files=sources,
              base_headers=base_headers,adapter_header=adapter_pin,expected_summary=expected,
              fusion_source_semantics=dict(preset_rank=32,preset_multiplier=1,
                checkpoint_base_storage='F32',pipeline_dtype='BF16',
                native_equation='BF16 base after load + 1 * torch.mm(B_BF16,A_BF16); then new PEFT rank4/alpha4 adapter.',
                source_only=True,actual_fusion_values_verified=False),
              qualification='Original header/source identities only; full tensor payload SHA and actual fused-base values are still required.')
out=D/'ORIGINAL_FUSION_CONTRACT.json'
with out.open('xb') as f:f.write((json.dumps(contract,indent=2)+'\n').encode())
result=dict(passed=True,actual_header_entries=1001,header_files=4,
            original_sources=sources,original_header_payload_sha256=hashlib.sha256((D/'original-metadata.private.json').read_bytes()).hexdigest(),
            original_native_name_mapper_executed=True,native_mapper_SHA256=hashlib.sha256(raw).hexdigest(),
            complete_240_target_mapping_equal=True,original_tensor_body_bytes_read=0,
            source_and_header_check_seconds=None,contract_SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),
            metadata={k:v for k,v in observed.items() if k not in ('preset_dimensions','new_target_shapes')},
            Torch_imported='torch' in sys.modules,GPU_initialized=False,actual_fused_base_verified=False,
            numerical_acceptance=False,Model_GO=False)
with (D/'original-header-unified-check-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
