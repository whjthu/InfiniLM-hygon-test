"""Bind a separately prepared one-use root decision to this exact package."""
import time

from contract import DISPATCH_REQUEST, LABEL, NODE, PARENT, PORT, REQUEST, need

def verify_authority(value, source, epoch):
    need(value['request'] == REQUEST and value['dispatch_request'] == DISPATCH_REQUEST
         and value['label'] == LABEL and str(value['parent']) == PARENT and value['node'] == NODE
         and value['port'] == PORT and value['source_manifest'] == source, 'root scope/label/pin differs')
    need(value['root_GPU_GO'] is True and value['same_business_placement_confirmed'] is True
         and value['no_gold_target_staged'] is True and value['score_GO'] is False
         and isinstance(value['reservation_SHA256'], str) and len(value['reservation_SHA256']) == 64
         and set(value['reservation_SHA256']) <= set('0123456789abcdef')
         and isinstance(value['preflight_SHA256'], str) and len(value['preflight_SHA256']) == 64
         and set(value['preflight_SHA256']) <= set('0123456789abcdef'),
         'actual root approval, reservation and asset witness required')
    need(type(value['issued_epoch']) in (int, float) and type(epoch) in (int, float)
         and -3 <= epoch - value['issued_epoch'] <= 120 and time.time() < epoch + 900,
         'root approval freshness differs')
