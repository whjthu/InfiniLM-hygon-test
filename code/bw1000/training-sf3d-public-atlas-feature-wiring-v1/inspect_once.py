"""Actual target-input joins and frozen native cache predicates; no model/feature stub."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time

from target_join import G,BINDING,BINDING_PIN,checked,targets_for_image

D=Path(__file__).resolve().parent
OUT=G/'local-only/training-sf3d-public-atlas-feature-wiring-v1-original'
ARCHIVE=G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def calls(node):return sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)})


def main():
    begin=time.monotonic();OUT.mkdir()
    result={'schema':'SF3D_PUBLIC_ATLAS_REAL_TARGET_FEATURE_WIRING_V1','passed':False,'planned_objects':2,
        'actual_target_input_joins':0,'actual_native_preprocess_encoder_feature_or_training_calls':0,
        'Torch_or_fake_feature_module_imports':0,'actual_GETs':0,'READY':False,'phase':'frozen-source'}
    try:
        checked(ARCHIVE,{'bytes':738802,'SHA256':'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'})
        with tarfile.open(ARCHIVE,'r:gz') as t:
            train=t.extractfile('framework/generation/src/gencuda/backends/sf3d_training.py').read()
            reconstruction=t.extractfile('framework/generation/src/gencuda/backends/sf3d_reconstruction.py').read()
        if pin(train)!={'bytes':58458,'SHA256':'dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802'} or pin(reconstruction)!={'bytes':34062,'SHA256':'aa0d19927e05ffb26931f28c5f29a0f01acaa1d0bcfabe45141b6a562a2decba'}:raise ValueError('Frozen backend source differs')
        tree=ast.parse(train);rtree=ast.parse(reconstruction);run=function(tree,'run_sf3d_training')
        keys=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_SF3D_CACHE_KEYS' for x in n.targets))
        if keys!=('triplane_features','image_features','albedo','roughness','metallic'):raise ValueError('Original cache fields changed')
        validation=function(tree,'_sf3d_validate_cached_batches')
        original_key_gate=next(n.args[0] for n in ast.walk(validation) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_sf3d_require'
            and n.args and 'set(saved)' in ast.unparse(n.args[0]) and '_SF3D_CACHE_KEYS' in ast.unparse(n.args[0]))
        min_input_gate=next(n.test for n in ast.walk(run) if isinstance(n,ast.If) and 'len(input_paths) < 2' in ast.unparse(n.test))
        config_function=function(tree,'_sf3d_training_config_sha256')
        config_dict=next(n.value for n in config_function.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='values' for x in n.targets))
        config_keys=[ast.literal_eval(k) for k in config_dict.keys]
        config_without_labels=set(config_keys)=={'seed','query_points','learning_rate','weight_decay','scalar_loss_weight','max_grad_norm','warmup_steps','max_steps'}
        if not config_without_labels:raise ValueError('Expected original config hash contract changed')
        feature_stages={name:{'lines':[f.lineno,f.end_lineno],'calls':calls(f),'AST_SHA256':hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()}
            for name,f in [('_prepare_batch',function(rtree,'_prepare_batch')),('_image_features',function(tree,'_image_features')),
                ('_material_forward_loss',function(tree,'_material_forward_loss')),('_sf3d_cache_feature_spec',function(tree,'_sf3d_cache_feature_spec'))]}
        if not {'model.prepare_image','model.image_processor'}.issubset(feature_stages['_prepare_batch']['calls']) or 'model.image_estimator.model.encode_image' not in feature_stages['_image_features']['calls']:raise ValueError('Actual model-bound feature chain changed')
        binding=json.loads(checked(BINDING,BINDING_PIN));joins={};missing={};cache_rejections=0
        result['phase']='actual-two-input-SHA-split-target-joins'
        for split,purpose in [('train','train'),('dev','eval-dev')]:
            row=binding['objects'][split]
            target,identity=targets_for_image(G/row['input'],recipe='sf3d-public-pbr-UV-atlas-v1',purpose=purpose)
            # Execute exactly the original source's structure clause on real
            # target data; never execute/stub its Tensor/dtype/feature validator.
            accepted=eval(compile(ast.Expression(original_key_gate),'original-SF3D-key-clause','eval'),{'saved':target,'_SF3D_CACHE_KEYS':keys})
            if accepted:raise ValueError('Target-only bundle falsely accepted as full native cache')
            missing[split]=sorted(set(keys)-set(target));cache_rejections+=1
            if missing[split]!=['image_features','triplane_features']:raise ValueError('Actual missing native feature fields differ')
            joins[split]={'identity':identity,'target_keys':sorted(target),'albedo_shape':list(target['albedo'].shape),
                'roughness_type':type(target['roughness']).__name__,'metallic_type':type(target['metallic']).__name__,
                'accepted_as_native_cached_batch':False}
            result['actual_target_input_joins']+=1
        result['phase']='train-split-and-minimum-input-boundary'
        heldout_rejected=False
        try:targets_for_image(G/binding['objects']['dev']['input'],recipe='sf3d-public-pbr-UV-atlas-v1',purpose='train')
        except ValueError as e:heldout_rejected=str(e)=='Heldout object cannot enter training'
        if not heldout_rejected:raise ValueError('Heldout object admitted to training')
        input_paths=[G/binding['objects']['train']['input']]
        original_minimum_rejects=eval(compile(ast.Expression(min_input_gate),'original-SF3D-minimum-input-clause','eval'),{'input_paths':input_paths})
        if original_minimum_rejects is not True:raise ValueError('Actual train-only minimum input check differs')
        descriptor={'schema':'SF3D_PUBLIC_ATLAS_REAL_TARGET_INPUT_DESCRIPTOR_V1','training':[joins['train']['identity']],
            'development':[joins['dev']['identity']],'blind':[],'actual_unique_training_objects':1,'actual_unique_training_images':1,
            'native_minimum_two_inputs_satisfied':False,'label_mode':'author_PBR_uniform_UV_atlas_mean_v1',
            'target_join_hook':'Replace only target selection at original1110 within a new explicit recipe; later1127..1129 tensor construction and feature functions remain native.',
            'requires_label_identity_in_resume_contract':True,'actual_feature_widths':'UNKNOWN until real model.heads introspection',
            'training_Go':False,'READY':False}
        descriptor_raw=(json.dumps(descriptor,sort_keys=True,indent=2)+'\n').encode();(OUT/'target-input-descriptor-private.json').write_bytes(descriptor_raw)
        source_trace={'training_backend':pin(train),'reconstruction_backend':pin(reconstruction),'feature_stages':feature_stages,
            'actual_feature_producer_in_run':{'require_cuda':1050,'load_real_model':list(range(1066,1071)),
                'prepare_and_encode_scene':[1117,1118],'query_real_triplane':[1119,1120,1121],'encode_real_image':1122,
                'cache_append':list(range(1123,1132)),'labels_selected_from_original_heuristic':1110},
            'cache_save':[415,428],'cache_validate':[450,512],'cache_restore':[515,524],
            'consumer_source':{'albedo_sigmoid_decoder':[145,147],'roughness_beta_mean':[148,152],'metallic_beta_mean':[153,157],'loss':[158,164]},
            'original_cache_keys':list(keys),'original_key_clause':ast.unparse(original_key_gate),'original_minimum_input_clause':ast.unparse(min_input_gate),
            'original_config_hash_keys':config_keys,'new_target_label_identity_not_in_original_config_hash':True,
            'original_sampled_query_points':'CPU seed+17 generator -> uniform3D cube [-model.cfg.radius,+radius], default2048; no UV lookup/correspondence',
            'original_albedo_target_broadcast':'(1,3) global target broadcast over all triplane query points; per-UV labels cannot be silently treated as point labels'}
        source_raw=(json.dumps(source_trace,sort_keys=True,indent=2)+'\n').encode();(OUT/'source-call-trace.json').write_bytes(source_raw)
        result.update(passed=True,state='REAL_TARGET_INPUT_JOIN_VERIFIED_NATIVE_FEATURES_MISSING',phase='complete',
            original_key_clause_target_only_rejections=cache_rejections,missing_native_fields=missing,heldout_to_training_rejected=True,
            actual_train_only_images=1,actual_original_minimum_guard_rejects_train_only=True,original_minimum_images=2,
            synthetic_or_zero_features_created=False,original_model_or_weights_or_8RGBA_touched=False,
            current_local_torch_available=importlib.util.find_spec('torch') is not None,
            current_local_open_clip_available=importlib.util.find_spec('open_clip') is not None,
            current_local_sf3d_available=importlib.util.find_spec('sf3d') is not None,
            native_model_bound_feature_widths_known=False,no_deployed_weightfree_native_batch_preprocess=True,
            target_descriptor=pin(descriptor_raw),source_trace=pin(source_raw),actual_wall_seconds=time.monotonic()-begin,
            original_native_validator_executed=False,original_structure_and_input_clauses_executed=True,
            native_or_training_GO=False,joins=joins)
    except Exception as e:
        import traceback
        raw=(json.dumps({'phase':result['phase'],'type':type(e).__name__,'reason':str(e),'trace':traceback.format_exc()},indent=2)+'\n').encode()
        (OUT/'failure-private.json').write_bytes(raw);result.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,failure=pin(raw))
    (OUT/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['passed','state','planned_objects','actual_target_input_joins','actual_native_preprocess_encoder_feature_or_training_calls','READY']},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
