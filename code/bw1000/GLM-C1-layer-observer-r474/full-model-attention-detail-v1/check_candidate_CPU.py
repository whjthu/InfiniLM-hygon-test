"""New attention/169/original-epoch boundaries only; no library load or GPU."""
import copy
import json
import time
from unittest.mock import patch

import attention_evidence as detail
import budget
import ffn_evidence as frames
import host_launch as host
import native_arm
from read_terminal import attention_verdict
from slurm_accounting import is_terminal

passed, refused = [], []


def reject(name, fn):
    try:
        fn()
    except (ValueError, AssertionError):
        refused.append(name)
    else:
        raise AssertionError('accepted '+name)


def frame(count):
    names = ['CAPI.owner.begin']
    for layer in range(78):
        kind = 'shared' if layer == 3 else 'full'
        names.append(f'layer.{layer}.{kind}.begin')
        if layer in (0, 3):
            prefix = f'detail.layer.{layer}.'
            def pair(stem):
                names.extend([prefix+'attnd1.'+stem+'.begin', prefix+'attnd1.'+stem+'.return'])
            pair('prepare.forward_latent')
            names += [prefix+'attnd1.projection.tuple.M1.N8.K16.row_offset0.column_offset0.input_BF16.contiguous_checked.canonical_stride_inferred_16_1.raw_stride_UNKNOWN.kernel_provider_UNKNOWN',
                      prefix+'attnd1.projection.dispatch.cached_r809.kernel_provider_UNKNOWN']
            pair('projection.original_success_fence')
            for stem in ['fresh.expand_cached', 'attention.certified', 'attention.result_copy']:
                pair(stem)
            names.append(prefix+'attnd1.output.forward.begin')
            names += [prefix+'attnd1.output.tuple.M1.N6144.K16.row_offset0.column_offset0.input_BF16.contiguous_checked.canonical_stride_inferred_16_1.raw_stride_UNKNOWN.kernel_provider_UNKNOWN',
                      prefix+'attnd1.output.dispatch.fallback_Fp8BlockLinear.kernel_provider_UNKNOWN']
            names.append(prefix+'attnd1.output.forward.return')
            pair('output.original_stage_finish_or_stream_wait')
            names += [prefix+'FFN.begin', *(prefix+mark for mark in frames.MARKS), prefix+'FFN.end']
        names.append(f'layer.{layer}.end')
    assert len(names) <= count
    names += ['fixture.legacy-padding']*(count-len(names))
    return dict(rank=1, tokens=1, stage=0, sequence=3, row_past_count=[[0, 1024, 1]],
                full_detail_layer=0, shared_detail_layer=3, event_count=count,
                marks=[dict(name=name, owner_host_s=i*.001, gpu_from_first_s=i*.001) for i,name in enumerate(names)])


value = frame(440)
diagnostic = dict(prompt_tokens=1024, prefill_fragment_plan=[{}, {}])
assert frames.check_frame(value, diagnostic, 9, 0) == [0, 3]
assert len(detail.attention_windows(value)) == 2
passed.append('actual_full_78layer_global9_local1_440mark_reader')
bad = copy.deepcopy(value); bad['marks'].append(dict(bad['marks'][-1])); bad['event_count'] = 441
reject('441marks', lambda: frames.check_frame(bad, diagnostic, 9, 0))
bad = copy.deepcopy(value); bad['marks'][1]['name'] = 'layer.0.shared.begin'
reject('wrong_selected_layer_kind', lambda: frames.check_frame(bad, diagnostic, 9, 0))
bad = copy.deepcopy(value); bad['marks'][4]['name'] = 'detail.layer.2.attnd1.attention.selected.begin'
reject('foreign_attention_layer', lambda: detail.attention_windows(bad))
bad = copy.deepcopy(value); bad['marks'][2]['name'] = 'detail.layer.0.attnd1.prepare.forward_latent.return'
reject('unmatched_attention_return', lambda: detail.attention_windows(bad))
bad = copy.deepcopy(value); next(mark for mark in bad['marks'] if mark['name']=='layer.77.end')['name']='missing'
reject('missing78th_layer', lambda: detail.attention_windows(bad))
bad = copy.deepcopy(value)
move = next(i for i, mark in enumerate(bad['marks']) if mark['name'].startswith('detail.layer.0.attnd1.projection.tuple.'))
mark = bad['marks'].pop(move)
target = next(i for i, item in enumerate(bad['marks']) if item['name'] == 'layer.1.full.begin')+1
bad['marks'].insert(target, mark)
for i, item in enumerate(bad['marks']):
    item.update(owner_host_s=i*.001, gpu_from_first_s=i*.001)
reject('layer0_tuple_moved_into_layer1_window', lambda: detail.attention_windows(bad))

for per_rank in [0, 1, 2, 3]:
    count = per_rank*16
    verdict = attention_verdict(dict(planned_ranks=16, planned_frames=48, observed_frames=count, full_coverage=count == 48))
    assert verdict['attention_schema_valid'] is True and verdict['attention_observed_rank_frames'] == count
    assert verdict['valid_attention_detail'] is (count == 48)
    assert verdict['attention_measurement_state'] == ('MEASURED_FULL48' if count == 48 else 'UNKNOWN_INSUFFICIENT_C1_FRAMES')
passed.append('schema_valid_but0_16_32frames_UNKNOWN_only48_full')
reject('zero_frames_falsely_full', lambda: attention_verdict(dict(planned_ranks=16, planned_frames=48, observed_frames=0, full_coverage=True)))
reject('49frames_over_lifetime_bound', lambda: attention_verdict(dict(planned_ranks=16, planned_frames=48, observed_frames=49, full_coverage=False)))

for elapsed in [0, 1, 161, 1799]:
    seconds = budget.slurm_seconds(1000., 'gate', now=1000.+elapsed)
    assert seconds % 60 == 0 and elapsed+seconds <= budget.GPU_SECONDS
    assert budget.INSIDE_SECONDS-elapsed+15 < seconds
passed.append('actual_gate_minute_floor_and_inside_kill_absolute_fit')
for left, expected in [(161, 120), (100, 60)]:
    assert budget.slurm_seconds(1000., 'post', now=1000.+budget.PAIR_SECONDS-left) == expected
reject('post80s_cannot_fit_new_minute_and_reserve', lambda: budget.slurm_seconds(1000., 'post', now=1000.+budget.PAIR_SECONDS-80))
reject('expired_dispatch_epoch', lambda: budget.require_pair_capacity(1000., 1, now=61001.))
with patch.object(host, 'ORIGIN_EPOCH', 1000.), patch.object(host.time, 'time', return_value=1161.):
    argv = host.command('CPU-fixture', 'gate', __import__('pathlib').Path('fixture'))
    assert '--gres=dcu:8' in argv and '--time=957' in argv
passed.append('actual_srun_argv_uses_shrunk957_minutes_not_fixed960')
assert not is_terminal(['1103606.999','fixture','RUNNING','0:0','1'])
assert host.actual_resources(['id','label','COMPLETED','0:0','1','nodes','cpu=128,node=2,gres/dcu=16'], 'gate')
assert not host.actual_resources(['id','label','COMPLETED','0:0','1','nodes','cpu=128,node=2,gres/dcu=8'], 'gate')
passed.append('RUNNING_not_terminal_and_actual16DCU_resources')

build = dict(passed=True, step='1103606.169', new_HOST_TUs=3, new_GPU_TUs=0, GPU_cards=0,
             linked_objects=32, reused_objects=29, all3_fresh_M_before_compile=True,
             all3_MD_equal_fresh_M=True, strong_exports_preserved=280, providers=71,
             tls_storage_verified=True, Native_dlopen=False, library_bytes=1930288,
             library_SHA256=native_arm.LIBRARY[1], MD={}, fresh_M={})
for name, count in [('nov',1019),('prepare',993),('output',988)]:
    build['MD'][name] = dict(equal_fresh_M=True, count=count)
    build['fresh_M'][name] = dict(count=count, dependencies_SHA256={str(i):'CPU-fixture' for i in range(count)})
outcome = dict(passed=True, rc=0, client_timeout=False, current_step_outcome='TERMINAL', GPU_cards=0,
               epoch=1000., deadline=1600., build_SHA256=native_arm.BUILD_SHA256, build=build,
               accounting=dict(terminal_visible=True, row=['1103606.169','GLM474-attention-detail-CPU-v1','COMPLETED','0:0','63','f13r1n05','cpu=4,mem=16G,node=1']))
native_arm.verify_outcome(outcome)
passed.append('actual169_schema_contract_without_Native_load')
bad = copy.deepcopy(outcome); bad['build']['library_SHA256'] = '0'*64
reject('wrong169_DSO', lambda: native_arm.verify_outcome(bad))
bad = copy.deepcopy(outcome); bad['build']['MD']['prepare']['equal_fresh_M'] = False
reject('stale_prepare_MD', lambda: native_arm.verify_outcome(bad))
bad = copy.deepcopy(outcome); bad['accounting']['row'][0] = '1103606.143'
reject('old143_receipt', lambda: native_arm.verify_outcome(bad))
print(json.dumps(dict(passed=True, positive=passed, rejected=refused, Native_loads=0,
                     Torch_SDK_GPU_calls=0, raw_private_plan_or_weight_reads=0)))
