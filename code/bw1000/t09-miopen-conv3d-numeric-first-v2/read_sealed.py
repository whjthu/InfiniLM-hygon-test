"""Independent exact T09 receipt-chain and actual numbered pre/GPU/post readback."""
import argparse
import json
from pathlib import Path

import host
from scope import REQUEST, STAGE, require
from stage_seals import consume


def verify(manifest,epoch,intent_pin,post_pin,pre_pin):
    require(set(host.source(STAGE,manifest))==host.MEMBERS,'EXACT_ORIGINAL_CURRENT_SOURCE')
    intent=host.pinned(STAGE/'gpu-intent.json',intent_pin)
    pre=host.pinned(STAGE/'pre.json',pre_pin)
    post=host.pinned(STAGE/'post.json',post_pin)
    require(intent['request']==pre['request']==post['request']==REQUEST
            and intent['epoch']==pre['epoch']==post['epoch']==epoch and intent['pre_SHA256']==pre_pin
            and pre['phase']=='pre' and post['phase']=='post' and pre['passed'] is post['passed'] is True
            and post['independent_physical_clearance'] is True
            and pre['physical']['boot_id']==post['physical']['boot_id']
            and pre['physical']['pci']==post['physical']['pci']
            and pre['physical']['kfd_holder_count']==post['physical']['kfd_holder_count']==0,
            'EXACT_SAME_BOOT_PCI_INDEPENDENT_PHYSICAL_CLEAR')
    pre_row=host.actual_terminal('pre',pre['step'],epoch,epoch+240,True)
    gpu_row=host.actual_terminal('gpu',intent['step'],epoch,epoch+240)
    post_row=host.actual_terminal('post',post['step'],epoch,epoch+240,True)
    require(pre_row==post['pre_accounting'] and gpu_row==post['gpu_accounting']
            and pre_row[8]<=gpu_row[7] and gpu_row[8]<=post_row[7]
            and len({pre_row[0],gpu_row[0],post_row[0]})==3,
            'EXACT_INDEPENDENT_FRESH_NINE_COLUMNS_AND_PHASE_ORDER')
    original=consume(STAGE/'receipts-once',intent)
    numeric=original.get('01-numeric.json',{}).get('result')
    trace=original.get('02-aten.json',{}).get('result')
    provider=original.get('03-provider.json',{}).get('result')
    numeric_pass=bool(numeric and numeric['numeric_pass'] is True and numeric['actual_output_shape']==[1,2048,1,2,2]
                      and numeric['output_elements']==numeric['finite_elements']==8192
                      and numeric['max_absolute_error']==0 and numeric['reference_value']==65.0
                      and numeric['synchronized_single_conv3d'] is True)
    trace_pass=bool(trace and trace['ATen_route_passed'] is True and trace['correct_conv3d_nested_miopen'] is True
                    and trace['marker_count']==1 and trace['scoped_operator_counts'].get('aten::miopen_convolution')==1)
    mapped_pass=bool(provider and provider['mapped_provider_qualified'] is True
                     and provider['selection']['state']=='ONE_EXECUTABLE_EXACT_MIOPEN_GROUP'
                     and provider['exact_pin_policy']=='AUTHORIZED_PATH_SAME_MAPPED_DEV_INODE_SAME_FD_SHA_ONLY'
                     and provider['map_files_permission_probe_attempted'] is False)
    return dict(request=REQUEST,state='INDEPENDENT_ORIGINAL_SCOPE_READBACK',actual_pre_gpu_post=[pre_row,gpu_row,post_row],
                independent_physical_release=True,numeric=numeric,numeric_pass=numeric_pass,
                ATen=trace,ATen_pass=trace_pass,approved_mapped_provider_pass=mapped_pass,
                provider_selection=provider.get('selection') if provider else None,
                result_file_pins={key:value['pin'] for key,value in original.items()},
                vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED',native_open_write_syscalls='UNKNOWN_NOT_CAPTURED',
                actual_original_models_loaded_by_reader=0,formal_performance_ready=False,
                all_three_requirements_pass=numeric_pass and trace_pass and mapped_pass and gpu_row[2:4]==['COMPLETED','0:0'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',type=float,required=True)
    for key in ('manifest-sha','gpu-intent-sha','post-sha','pre-sha'):
        parser.add_argument('--'+key,required=True)
    a=parser.parse_args()
    try:
        ret=verify(a.manifest_sha,a.epoch,a.gpu_intent_sha,a.post_sha,a.pre_sha)
        print(json.dumps(ret,sort_keys=True,allow_nan=False))
    except (OSError,ValueError,TypeError,KeyError) as error:
        print(json.dumps(dict(request=REQUEST,state='READBACK_UNQUALIFIED',error_type=type(error).__name__)))
        raise SystemExit(2)
