"""Project the pinned .3729 attention evidence without reading tensors or traces."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = 'results/bw1000/2026-11-26/'
SOURCE = 'code/bw1000/training-hunyuanvideo-t09-step6-native-topology-v2/'
PREFIX = 'code/bw1000/training-hunyuanvideo-prefix-math-sdpa-v1/'
PINS = {
    'terminal': (RESULTS+'training-hunyuanvideo-original3729-six-updates-step6-observer-independent.json',10713,'3223ca2d8bb0a8456f60b065b33ca9cb834da0e428ee5d3a5f61d0ba3af5d18f'),
    'window': (RESULTS+'training-hunyuanvideo-original3729-all-operator-54-blocks-recompute-cost.json',46499,'a24cc669e3f12f3c6cf32ebebe75bc11303ae2d199923cd0d94762fd1fb740c1'),
    'window_reader': ('tools/read_t09_original3729_recompute_by_block.py',12932,'2264650d7155093d89db634f9641d7c5982d6c23954d9ae8f4817d5d85d46c42'),
    'production': (SOURCE+'production_scope.py',14125,'e9847a0d48a2562671c8461a40216d425eb742be55c1ef97fb651c2d06e08169'),
    'worker': (SOURCE+'native_worker.py',5208,'438479662b65e40b2d5383dee9b7e479e33a68b942b902fd2258b1782dd1405e'),
    'commands': (SOURCE+'commands.py',3569,'ef4ab642f040f9e2e0b7187dd91e343da82865d3eeb392f03dce143cb7324041'),
    'compat_source': (SOURCE+'flash_compat.py',8637,'70fbc54b7396730a533bab5ff795ee5526a66c14b5995c16fef63911ac68c1c1'),
    'compat_spec': (SOURCE+'COMPATIBILITY.json',771,'de873e77e7749aa6b66bb374e47eb3ee5d3652e170544c3ad5ce71ce71566c4d'),
    'native_manifest': (SOURCE+'NATIVE_BINDING.json',7689,'26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7'),
    'observer': (SOURCE+'backward_observer.py',26379,'0043b4319d2a682e2d0a427e36c966b6ef5a7402419bab828000887d5c6c66f3'),
    'historic_source': ('code/bw1000/training-hunyuanvideo-unpad-arity-diagnostic-v1/SOURCE_NOTES.json',3628,'829e78b09120c2ff3662bdade85a6612318f96c6b418173cf7814f5f2315edda'),
    'prefix_math': (PREFIX+'prefix_sdpa.py',1288,'d72efcb8f77a02cb9161f56497a327a5c848430819f0d839d43bdbd45ea8a98f'),
    'prefix_runner': (PREFIX+'run_prefix_pair.py',6580,'3cff2c8202af065fc1c706522ed9a1f3501247f0f6adb462c9d0363fc9f7d564'),
    'prior_forward': ('local-only/training-hunyuanvideo-native-forward-capture-v1/terminal-independent.json',10537,'1b2f851bdee8e9803a7bbd1aa3cb1d298463944d2d3a0137604274ebb225ab25'),
    'prior_full_prefix': ('local-only/training-hunyuanvideo-native-prefix-v1/terminal-independent.json',2845,'7e50866558cbf2ebc101cd5ba0b393b479684a931bbf536e5b6bdd21b04e9a0d'),
    'prior_kv_prefix': ('local-only/training-hunyuanvideo-native-kv-prefix-v1/terminal-independent.json',4074,'1a68308914714cc0579f654b5943821ebb22fc5a766a96bb58c08c11f46043e2'),
    'synthetic_prefix': ('local-only/T09-native-prefix-math-sdpa-pair-v1/terminal-independent.json',6383,'b3058379ff1aae6e13e36afb4888a408cc20dc72a8710a2a904f5e9c08ad3a8b'),
}
TRACE = (105595897,'14840be27b10a9b1dc722f408a7be432f55c7f6ab0a13df76a8897d2e9e72dc6')
CLASSES = {'forward':108,'recompute':108,'autograd_sequence':0,'unknown':215}
ATTENTION_PATH = 'hyvideo/models/transformers/modules/attention.py'
MODEL_PATH = 'hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py'
FLASH_PATH = 'hyvideo/utils/flash_attn_no_pad.py'


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_pinned(key):
    name,size,digest = PINS[key]
    with (ROOT/name).open('rb') as file:
        raw = file.read(size+1)
    need(len(raw)==size and sha256(raw).hexdigest()==digest, 'PIN_MISMATCH_'+key)
    return json.loads(raw) if name.endswith('.json') else ast.parse(raw,filename=name)


def measure(terminal, window):
    step = terminal['step6']
    counts = step['bmm_detail']
    need(terminal['numbered_accounting']['gpu'][:4]==['1103606.3729','t09-step6-topology-fd68a293d7dd-gpu','COMPLETED','0:0']
         and terminal['profiled_steps']==[6] and terminal['updates_completed']==terminal['updates_planned']==6
         and terminal['cleanup_independently_verified'] is True and window['passed'] is True
         and window['original_step']=='1103606.3729' and window['step']==6 and window['block_count']==54
         and window['planned_blocks']==54 and counts['classes']==window['bmm_original_detail']==CLASSES
         and counts['actual_bmm_kernel_count']==sum(CLASSES.values()) and window['known_recompute_BMM']==108
         and counts['actual_block_count']==54 and step['bmm_recomputation']['valid_recompute_ranges']==54
         and counts['shape_inferred'] is False and window['raw_trace_output'] is False
         and (step['raw_trace_bytes'],step['raw_trace_SHA256'])==(window['trace_bytes'],window['trace_SHA256'])==TRACE,
         'EXACT_ORIGINAL_TRACE_AND_ATTRIBUTION_WINDOW')
    fields = ('Input Dims','Input Strides','Input type')
    need(counts['operand_fields_observed']==dict.fromkeys(fields,431), 'ALL_431_OPERAND_FIELDS_REQUIRED')
    blocks = window['blocks']
    need([b['ordinal'] for b in blocks]==list(range(54)) and all(b['block']==f'double_blocks.{b["ordinal"]}' for b in blocks),
         'ORIGINAL_54_BLOCK_MAP_REQUIRED')
    output = []
    for group in step['top_bmm_groups']:
        meta = group['operand_metadata']
        dims,strides,dtypes = (meta[key] for key in fields)
        phase, ordinal = group['phase'],group['ordinal']
        need(len(dims)==len(strides)==len(dtypes)==2 and all(len(v)==3 and all(type(x) is int and x>0 for x in v) for v in dims+strides)
             and dtypes==['float','float'] and dims[0][0]==dims[1][0] and dims[0][2]==dims[1][1]
             and ((phase=='unknown' and ordinal is None) or (phase in ('forward','recompute') and type(ordinal) is int and 0<=ordinal<54))
             and group['sequence_evidence']=='UNKNOWN' and type(group['kernel_count']) is int and group['kernel_count']>0,
             'ORIGINAL_TWO_BMM_OPERANDS_REQUIRED_WITH_UNKNOWN_SEQUENCE')
        output.append(dict(phase=phase,block=None if ordinal is None else blocks[ordinal]['block'],kernels=group['kernel_count'],
                           device=group['device'],stream=group['stream'],autograd_sequence=None,
                           operands=[dict(shape=d,stride=s,dtype=t) for d,s,t in zip(dims,strides,dtypes,strict=True)]))
    need(len(output)==8, 'BOUNDED_TOP_GROUP_PROJECTION')
    return dict(step='1103606.3729',update=6,windows=54,phases=CLASSES,total_bmm_kernels=431,
                aggregate_groups=counts['groups'],aggregate_groups_SHA256=counts['groups_SHA256'],
                operands_metadata_counts=counts['operand_fields_observed'],displayed_top_groups=output,
                full_backward_envelope=window['conservative_backward_envelope'],
                proved_recompute=window['qualified_device_recompute'],
                unchanged_device_timeline_removal_upper_bound_us=window['exclusive_all_capture_upper_bound_us'])


def source_chain(inputs):
    source = inputs['production']
    worker = inputs['worker']
    compat = inputs['compat_spec']
    manifest = inputs['native_manifest']['files']
    worker_set = {ast.unparse(n) for n in ast.walk(worker) if isinstance(n,ast.Call)}
    expected_cli = ast.dump(ast.parse("runpy.run_path(str(CLI), run_name='__main__')",mode='eval').body)
    need(any(ast.dump(n)==expected_cli for n in ast.walk(source) if isinstance(n,ast.Call)), 'NATIVE_CLI_NOT_BOUND')
    need('validate_execution_compatibility()' in worker_set and 'os.execv(bootstrap[0], bootstrap)' in worker_set,
         'ORIGINAL_HELPER_AND_PRODUCTION_BOOTSTRAP_REQUIRED')
    expected_binds = {ast.dump(ast.parse(f'f"{{{a}}}:{{{b}}}:ro"',mode='eval').body)
                      for a,b in (('NATIVE','NATIVE_ALIAS'),('BACKING','ALIAS'))}
    need(expected_binds <= {ast.dump(n) for n in ast.walk(inputs['commands']) if isinstance(n,ast.JoinedStr)},
         'ORIGINAL_READONLY_SOURCE_AND_HELPER_BINDS_REQUIRED')
    real = inputs['historic_source']['four_actual_source_pins']
    need(all({k:manifest[path][k] for k in ('bytes','SHA256')}==real[key]
             for path,key in ((MODEL_PATH,'transformer'),(ATTENTION_PATH,'attention'),(FLASH_PATH,'flash_helper')))
         and real['flash_helper']==compat['original'] and compat['modified_unpad_assignments']==4
         and compat['inverse_AST_exact'] is True and compat['attention_math_changed'] is False,
         'HELPER_ABI_OVERLAY_MUST_PRESERVE_ATTENTION_MATH')
    return dict(case='T09.sft', entry='native_worker.py -> production_scope.py -> tools/hygon_training_recipe.py',
                native_files={name:manifest[name] for name in (MODEL_PATH,ATTENTION_PATH,FLASH_PATH)},
                readonly_flash_unpad_ABI_overlay=compat,
                helper_reachability='ABI_OVERLAY_INGRESS_BOUND_BMM_MATH_CALLER_UNBOUND')


def earlier(inputs):
    prior = inputs['prior_forward']
    capture = prior['capture']
    group, = capture['sdpa_descriptor_groups']
    need(prior['actual_accounting']['gpu'][0]=='1103606.1411' and prior['passed'] is True
         and capture['actual_global_step']==6 and group['calls']==capture['sdpa_descriptor_calls']==54
         and all(group[k]['dtype']=='torch.bfloat16' for k in ('query','key','value'))
         and group['mask']['dtype']=='torch.bool', 'EXACT_HISTORIC_NATIVE_DESCRIPTOR_REQUIRED')
    synthetic = inputs['synthetic_prefix']
    need(synthetic['accounting']['gpu'][0]=='1103606.1415' and synthetic['completed_pairs']==synthetic['planned_pairs']==3
         and synthetic['numerics_passed'] is True and synthetic['formal_model_performance'] is False,
         'EXACT_SYNTHETIC_PREFIX_TERMINAL_REQUIRED')
    failures = []
    for key,step in (('prior_full_prefix','1103606.1418'),('prior_kv_prefix','1103606.1421')):
        trial = inputs[key]
        first = trial['prefix']['first_update']
        need(trial['actual_accounting']['gpu'][0]==step and trial['passed'] is False and trial['updates_completed']==0
             and first['baseline_reverted'] is True and first['groups']['unclipped_gradients']['passed'] is False,
             'EXACT_ORIGINAL_FAILED_PREFIX_TERMINAL_REQUIRED')
        failures.append(dict(step=step,updates=0,gradient_relative_l2=first['groups']['unclipped_gradients']['relative_l2'],
                             parameter_update_relative_l2=first['groups']['parameter_updates']['relative_l2']))
    return dict(historical_native=dict(step='1103606.1411',update=6,descriptor=group,
                    math_SDPA_scopes=prior['math_sdpa']['cpu_scopes'],
                    mask_values_and_query_padding='NOT_CAPTURED_BY_DESCRIPTOR',upstream_gradient='NOT_CAPTURED'),
                synthetic=dict(step='1103606.1415',pairs=3,layout=synthetic['layout'],dtype='BF16',
                    backend='forced math SDPA',mask='synthetic Boolean valid-prefix key mask',dropout=0.0,
                    scale='default',causal=False,query_padding='explicit output zero-fill',
                    upstream_gradient='separate seeded random BF16 tensor',numerics_passed=True),
                native_prefix_trials_failed=failures)


def report(inputs):
    return dict(status='BLOCKED_ACTIVE_PRODUCTION_CALL_SITE_UNBOUND', passed=False, Numeric_GO=False,
                GPU_GO=False, READY=False, actual_new_work=0,
                observed_bmm=measure(inputs['terminal'],inputs['window']), production=source_chain(inputs),
                earlier_evidence=earlier(inputs), fields={
                    'Q_K_V':'UNBOUND_FOR_3729_TWO_FP32_BMM_OPERANDS_ARE_NOT_THREE_SDPA_INPUTS',
                    'mask':'UNBOUND_FOR_3729_DTYPE_SHAPE_AND_CONTENT_SEMANTICS',
                    'scale':'UNBOUND_FOR_3729_EFFECTIVE_SCALAR_OR_DEFAULT',
                    'causal_dropout':'UNBOUND_FOR_3729_EFFECTIVE_CALL_OPTIONS',
                    'query_padding':'UNBOUND_FOR_3729_QUERY_VALIDITY_AND_OUTPUT_TREATMENT',
                    'upstream_gradient':'UNBOUND_FOR_3729_DOUT_OR_AUTOGRAD_SEQUENCE',
                    'active_call_site':'UNBOUND_FOR_3729_PROVIDER_CALLER_RANGE_AND_BMM_JOIN'},
                evidence={key:dict(path=path,bytes=size,SHA256=digest) for key,(path,size,digest) in PINS.items()})


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.parse_args()
    try:
        print(json.dumps(report({key:read_pinned(key) for key in PINS}),sort_keys=True,allow_nan=False))
    except (OSError,ValueError,TypeError,KeyError,SyntaxError) as error:
        print(json.dumps(dict(status='INVALID_OR_MISSING_EVIDENCE', error=type(error).__name__,
                              passed=False,Numeric_GO=False,GPU_GO=False,READY=False),sort_keys=True))
        return 1
    return 2


if __name__=='__main__':
    sys.exit(main())
