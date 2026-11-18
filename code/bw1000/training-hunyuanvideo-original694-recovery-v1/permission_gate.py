"""A root GO alone cannot grant the action-specific UNKNOWN-request exception."""
import re
OLD_REQUEST='694070604bdb409db0eb9bf76f0e9f0c'
TARGET='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-first-step-replay-a100-v1'
CAPTURE='/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-first-update-v1/capture'
PINS={'before.pt':(7149635,'2856fae7918e67fcda1abf4b9ec0106f8fe11900132397ab71c65d9ed5f1ac46'),
      'inputs.pt':(1853149,'5ee197e410ca7a5595b98375be64d392028f466f249a222aea532f2841a1851c')}

def require(ok,reason):
    if not ok:raise ValueError(reason)

def validate(claim,approval,plan):
    require(plan['original_request_id']==OLD_REQUEST and plan['target']==TARGET
        and plan['source_capture']==CAPTURE and set(plan['members'])==set(PINS)
        and plan['target_control_name']=='.recovery-original694-once-v1'
        and plan['epoch_seconds']==300 and plan['payload_bytes']==9002784,'FIXED_TWO_MEMBER_SCOPE')
    for name,(count,digest) in PINS.items():
        row=plan['members'][name]
        require(row['bytes']==count and row['SHA256']==digest and row['source']==CAPTURE+'/'+name
            and row['target']==TARGET+'/'+name and row['tensor_decode_allowed'] is False,'FIXED_MEMBER_PIN_PATH')
    require(claim['root_GO'] is claim['CPU_GO'] is claim['transfer_GO'] is True
        and claim['GPU_GO'] is claim['Model_GO'] is False and claim['max_invocations']==1,'ROOT_TRANSFER_ADOPTION_REQUIRED')
    require(claim['explicit_user_exception_received'] is True
        and isinstance(claim['exception_record_path'],str)
        and isinstance(claim['exception_record_SHA256'],str)
        and re.fullmatch('[0-9a-f]{64}',claim['exception_record_SHA256']),'SPECIFIC_USER_EXCEPTION_PIN_REQUIRED')
    require(approval['schema']=='T09_694_EXPLICIT_USER_EXCEPTION_V1'
        and approval['user_granted'] is True and approval['allow_atomic_recovery_despite_unknown'] is True
        and approval['original_request_id']==claim['original_request_id']==plan['original_request_id']
        and approval['original_state']==claim['original_state']==plan['original_state']=='UNKNOWN'
        and approval['target']==claim['target']==plan['target']
        and approval['members']==claim['members']==plan['members']
        and approval['max_invocations']==1 and approval['epoch_seconds']==300
        and approval['no_overwrite'] is True and approval['no_target_rename'] is True,'USER_EXCEPTION_EXACT_ACTION_REQUIRED')
    return True
