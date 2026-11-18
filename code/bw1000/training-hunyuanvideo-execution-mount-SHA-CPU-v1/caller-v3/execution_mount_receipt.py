"""Consume one root-qualified saved CPU wholeSHA receipt; never hash a body here."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat

WEIGHT_BYTES = 33306632192
WEIGHT_SHA = '71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86'
RELATIVE = 'training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model.safetensors'
LABEL = 'training-hunyuanvideo-execution-mount-SHA-CPU-v1'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def read_small(path, expected):
    path = Path(path)
    need(path.is_absolute() and not path.is_symlink() and path.is_file(),'QUALIFIED_RECEIPT_PATH')
    before = path.stat(); need(0 < before.st_size <= 65536,'QUALIFIED_RECEIPT_BOUND')
    with path.open('rb',buffering=0) as stream:
        raw = stream.read(65537); after = os.fstat(stream.fileno())
    key = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(key(before)==key(after)==key(path.stat()) and len(raw)==before.st_size
         and hashlib.sha256(raw).hexdigest()==expected,'QUALIFIED_RECEIPT_SHA_STABLE')
    return json.loads(raw)


def validate(receipt, authority, *, stamp, node, boot, uid):
    need(authority['root_independent_terminal_verified'] is True and authority['passed'] is True
         and authority['worker_exit_evidence']=='EXACT_SACCT_COMPLETED0_FOREGROUND_NO_CHILDREN'
         and authority['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES'],
         'ROOT_INDEPENDENT_CPU_TERMINAL_REQUIRED')
    row=authority['actual_accounting']
    need(len(row)==7 and re.fullmatch(r'1103606\.[0-9]+',row[0]) and row[1]==LABEL
         and row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r3n13','ACTUAL_CPU_ACCOUNTING')
    tres=dict(word.split('=',1) for word in row[6].split(',') if '=' in word)
    need(tres.get('cpu')=='2' and tres.get('mem') in ('4G','4096M')
         and not any(k.startswith('gres/') and v!='0' for k,v in tres.items()),'ACTUAL_TWO_CPU_4G_0GPU')
    need(receipt['schema']=='T09_EXECUTION_MOUNT_WHOLE_SHA_RECEIPT_V1' and receipt['passed'] is True
         and receipt['request_id']==authority['original_request_id'] and receipt['CPU_step']==row[0]
         and receipt['source_manifest_SHA256']==authority['CPU_source_manifest_SHA256']
         and receipt['publication']==authority['CPU_publication'] and receipt['epoch']==authority['CPU_epoch']
         and receipt['node']==node=='f13r3n13' and receipt['boot_id']==boot and receipt['uid']==uid==154459
         and receipt['parent']=='1103606','ORIGINAL_EXECUTION_NODE_BOOT_UID')
    need(receipt['relative_weight']==RELATIVE and receipt['payload_bytes_read']==WEIGHT_BYTES
         and receipt['weight_SHA256']==WEIGHT_SHA and receipt['file_identity']==stamp
         and stat.S_ISREG(receipt['file_mode'])
         and receipt['tensor_decodes']==receipt['Torch_SDK_imports']==receipt['GPU_cards']==0
         and receipt['original_replay_inputs_read'] is False and receipt['deadline']==receipt['epoch']+900
         and receipt['finished_epoch']<=receipt['work_deadline']<=receipt['epoch']+780,'EXACT_STABLE_EXECUTION_FILE')
    return dict(state='VERIFIED_RECEIVER_IDENTITY',verification='QUALIFIED_EXECUTION_MOUNT_CPU_WHOLE_SHA',
        weight_SHA256=WEIGHT_SHA,final_bytes=WEIGHT_BYTES,payload_bytes_read=0,
        earlier_CPU_payload_bytes_read=WEIGHT_BYTES,CPU_step=row[0],original_CPU_request=receipt['request_id'])


def consume(results, path, authority_path, authority_sha, stamp):
    need(isinstance(authority_sha,str) and re.fullmatch('[0-9a-f]{64}',authority_sha),'QUALIFIED_AUTHORITY_PIN_REQUIRED')
    # Host and container have fixed aliases of the same execution filesystem.
    results=Path(results)
    need(results.as_posix() in ('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results','/refactor/results')
         and Path(path)==results/RELATIVE,'EXACT_REGISTERED_EXECUTION_WEIGHT')
    authority=read_small(authority_path,authority_sha)
    declared=authority['worker_receipt']
    need(declared['relative_path']==LABEL+'/weight-receipt.json' and 0<declared['bytes']<=65536,'EXACT_SAVED_CPU_RECEIPT')
    receipt_path=results/declared['relative_path']
    receipt=read_small(receipt_path,declared['SHA256'])
    need(receipt_path.stat().st_size==declared['bytes'],'ACTUAL_SAVED_RECEIPT_BYTES')
    node=socket.gethostname().split('.',1)[0]
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return validate(receipt,authority,stamp=stamp,node=node,boot=boot,uid=os.getuid())
