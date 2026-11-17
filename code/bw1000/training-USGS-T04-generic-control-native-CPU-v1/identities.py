"""Choose generic control by frozen image identities only; never inspect answers."""
import hashlib
import json
from pathlib import PurePosixPath


def need(ok,message):
    if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()

def select_generic45(metadata,seed):
    roles=metadata['ordered_image_roles'];need(len(roles)==64 and len(set(roles))==64,'Expected64 ordered unique source roles')
    lookup={x['name']:x for x in metadata['files']};selected=[]
    dataset=lookup['original.cuda.jsonl'];need(dataset['bytes']==24129 and dataset['SHA256']=='0e476394e3b0117d90cfad8f89ccfafe1800917692e86efed84eaf182c453811','Frozen original JSONL identity differs')
    need(len({lookup['datasets/'+role]['SHA256'] for role in roles})==64,'Original64 repeated content')
    for ordinal,role in enumerate(roles):
        pin=lookup['datasets/'+role];need(PurePosixPath(role).is_absolute() is False and '..' not in PurePosixPath(role).parts and pin['source'].endswith('/datasets/'+role),'Image identity role/source differs')
        key=sha((seed+'\n'+role+'\n'+pin['SHA256']).encode())
        selected.append(dict(original_ordinal=ordinal,relative_role=role,image_bytes=pin['bytes'],image_SHA256=pin['SHA256'],source=pin['source'],selection_key_SHA256=key))
    selected=sorted(selected,key=lambda x:(x['selection_key_SHA256'],x['original_ordinal']))[:45]
    return dict(schema='IMAGE_IDENTITY_ONLY_CONTROL45_V1',selected_count=45,source_count=64,selection_seed=seed,source_dataset=dataset,canonical_original_rows_SHA256=metadata['canonical_rows_SHA256'],
        answer_or_model_selected=False,selected=selected,caption_payload_read=False,images_read=False,formal_COCO_CIDEr_holdout_eligibility=False,
        note='Original64 contain COCO val2017 training inputs; this control must not be evaluated on reused original64 as an independent COCO holdout.')
