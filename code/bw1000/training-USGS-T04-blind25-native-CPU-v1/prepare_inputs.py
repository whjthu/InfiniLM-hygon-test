"""Root-after-publication private blind25 view; never opens labels or train/dev."""
import argparse
import json
from pathlib import Path,PurePosixPath
from common import D,canonical,need,pinned,sha,verify_source


def validate_rows(rows,media):
    need(len(rows)==25 and len({r['id'] for r in rows})==25,'ALL25_UNIQUE_BLIND_USER_INPUTS')
    for row in rows:
        need(set(row)=={'id','chart_group','source_group','messages','images'}
             and len(row['messages'])==1 and row['messages'][0]['role']=='user'
             and len(row['images'])==1 and row['images'][0] in media,'BLIND_USER_ONLY_NO_GOLD')
    need(len({r['source_group'] for r in rows})==3 and {r['images'][0] for r in rows}==set(media),'FROZEN3GROUP5MEDIA')
    return rows


def prepare(dataset,output,target_data,remaining):
    verify_source();plan=json.loads((D/'PLAN.json').read_bytes())
    need(json.loads((D/'CLAIM.json').read_bytes())['root_GO'] is True,'ROOT_PUBLICATION_BEFORE_BLIND_INPUT_READ')
    need(not output.exists() and target_data.is_absolute(),'NEW_PRIVATE_VIEW')
    manifest=json.loads(pinned(dataset/'manifest-private.json',plan['dataset_manifest']))
    need(manifest['outputs']['blind/inputs.jsonl']==plan['blind_inputs'],'FROZEN_BLIND_INPUT_IDENTITY')
    raw=pinned(dataset/'blind/inputs.jsonl',plan['blind_inputs']);rows=[json.loads(x) for x in raw.splitlines()]
    validate_rows(rows,plan['media'])
    view=[]
    for row in rows:
        remaining();need(set(row)=={'id','chart_group','source_group','messages','images'}
            and len(row['messages'])==1 and row['messages'][0]['role']=='user' and len(row['images'])==1,
            'NO_ASSISTANT_LABEL_OR_NEW_PROMPT')
        name=row['images'][0];need(name in plan['media'] and PurePosixPath(name).parts[:2]==('blind','media'),'ONLY_FIVE_BLIND_MEDIA')
        view.append(dict(id=row['id'],messages=row['messages'],images=[str(target_data/name)]))
    output.mkdir(mode=0o700)
    for name,pin in plan['media'].items():
        remaining();body=pinned(dataset/name,pin,1024**2)
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    body=b''.join(canonical(r) for r in view);(output/'blind-swift.jsonl').write_bytes(body)
    binding=dict(passed=True,state='BLIND25_USER_ONLY_PRIVATE_VIEW',target_data=str(target_data),
        dataset_manifest_SHA256=plan['dataset_manifest']['SHA256'],dataset_audit_SHA256=plan['dataset_audit']['SHA256'],
        source_inputs=plan['blind_inputs'],ordered_source_id_SHA256=[sha(r['id'].encode()) for r in rows],files={'blind-swift.jsonl':dict(bytes=len(body),SHA256=sha(body),rows=25)},
        media=plan['media'],blind_gold_read=False,train_or_dev_data_read=False,literal_special_tokens_added=False,
        processor_or_model_called=False,planned_native_rows=25,source_groups=3)
    (output/'data-binding-private.json').write_bytes(canonical(binding));return binding


if __name__=='__main__':
    raise SystemExit('Use published submit_once only: preparation belongs to its original300s epoch.')
