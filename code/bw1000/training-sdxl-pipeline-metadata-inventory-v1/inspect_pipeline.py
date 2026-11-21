"""Explicit-root SDXL metadata inventory. Never open model tensor bodies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

METADATA = ('model_index.json','scheduler/scheduler_config.json',
            'tokenizer/tokenizer_config.json','tokenizer/vocab.json','tokenizer/merges.txt',
            'tokenizer_2/tokenizer_config.json','tokenizer_2/vocab.json','tokenizer_2/merges.txt',
            'text_encoder/config.json','text_encoder_2/config.json','vae/config.json','unet/config.json')
WEIGHTS = {
    'text_encoder':('model.safetensors','pytorch_model.bin'),
    'text_encoder_2':('model.safetensors','pytorch_model.bin'),
    'vae':('diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin'),
    'unet':('diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin'),
}


def stamp(s):
    return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def inspect(root,remaining):
    """No directory glob, cache scan, download, model import, or weight open."""
    answer={'root':str(root),'metadata':{},'weight_stats':{},'model_body_reads':0,
            'weight_body_SHA_verified':False,'native_tokenizer_VAE_verified':False,
            'shard_index_bodies_and_members_verified':False,
            'training_READY':False,'actual_updates':0}
    if not root.is_absolute():raise ValueError('Explicit absolute model root required')
    for name in METADATA:
        path=root/name
        try:
            a=path.lstat()
            if not stat.S_ISREG(a.st_mode) or path.is_symlink() or path.resolve(strict=True)!=path:
                answer['metadata'][name]={'state':'ALIASED_OR_NONREGULAR_NOT_READ'};continue
            if not 0<a.st_size<=2*1024**2 or a.st_size>remaining[0]:
                answer['metadata'][name]={'state':'METADATA_CAP_NOT_READ','bytes':a.st_size};continue
            with path.open('rb') as f:
                opened=os.fstat(f.fileno());raw=f.read(a.st_size+1);after=os.fstat(f.fileno())
            remaining[0]-=len(raw)
            if (stamp(a)!=stamp(path.lstat()) or stamp(opened)!=stamp(after)
                    or (a.st_dev,a.st_ino,a.st_size)!=(opened.st_dev,opened.st_ino,opened.st_size)
                    or len(raw)!=a.st_size):raise ValueError('Metadata changed during read')
            parsed=json.loads(raw) if name.endswith('.json') else None
            # No vocabulary values/text or model configuration body is exported.
            fields={k:parsed[k] for k in ('_class_name','architectures','tokenizer_class')
                    if type(parsed) is dict and k in parsed}
            answer['metadata'][name]={'state':'BODY_HASHED','bytes':len(raw),
                                     'SHA256':hashlib.sha256(raw).hexdigest(),'class_fields':fields}
        except FileNotFoundError:
            answer['metadata'][name]={'state':'ABSENT_AT_OBSERVATION'}
        except PermissionError as e:
            answer['metadata'][name]={'state':'PERMISSION_UNKNOWN','errno':e.errno}
        except ValueError as e:
            answer['metadata'][name]={'state':'INVALID_OR_CHANGED_METADATA','error_type':type(e).__name__}
    for component,names in WEIGHTS.items():
        states={}
        for name in (*names,*(n+'.index.json' for n in names)):
            path=root/component/name
            try:
                a=path.lstat()
                states[name]={'state':'REGULAR_STAT_ONLY' if stat.S_ISREG(a.st_mode) else 'NONREGULAR_NOT_READ',
                              **stamp(a)}
            except FileNotFoundError:states[name]={'state':'ABSENT_AT_OBSERVATION'}
            except PermissionError as e:states[name]={'state':'PERMISSION_UNKNOWN','errno':e.errno}
        answer['weight_stats'][component]=states
    answer['metadata_complete']=all(x['state']=='BODY_HASHED' for x in answer['metadata'].values())
    answer['weight_or_index_entries_present']=all(any(v['state']=='REGULAR_STAT_ONLY' and v['bytes']>0
        for v in entries.values()) for entries in answer['weight_stats'].values())
    answer['state']='METADATA_AND_WEIGHT_ENTRY_STATS_PRESENT_NOT_BODY_QUALIFIED' if (
        answer['metadata_complete'] and answer['weight_or_index_entries_present']) else 'PIPELINE_PREREQUISITES_INCOMPLETE_OR_UNKNOWN'
    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if len(args.root)>5 or len(set(args.root))!=len(args.root):raise ValueError('At most5 distinct explicit roots')
    remaining=[8*1024**2]
    result={'schema':'SDXL_EXPLICIT_PIPELINE_METADATA_INVENTORY_V1',
            'roots':[inspect(path,remaining) for path in args.root],
            'metadata_body_bytes_read':8*1024**2-remaining[0],
            'model_tensor_body_reads':0,'current_hardware_access':False,
            'actual_updates':0,'training_READY':False,'formal17':False}
    with args.output.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'roots':len(args.root),'metadata_body_bytes_read':result['metadata_body_bytes_read'],
                      'model_tensor_body_reads':0,'training_READY':False},sort_keys=True))


if __name__=='__main__':main()
