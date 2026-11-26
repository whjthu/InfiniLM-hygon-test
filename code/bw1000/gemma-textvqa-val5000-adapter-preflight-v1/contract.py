"""Separate the Gemma arm identity from the answerless official Val5000 inputs."""

from hashlib import sha256
import json
from pathlib import Path
import re


ARMS=('A100','Hygon','Hygon_before')
QIDS,IMAGES=5000,3166
ANN='4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'
IDENTITIES='c3e42949ca5d35b655995e4991d8f73a12f12f8da7d1428881f1258d3e0b1886'
SCORER='89b796e59a542068f6b09b5d4ef8be1747a31a03c3d2bf3df3919c0486eb405b'
MMF='e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a'
CP={'A100':'d6e5a097d491043db0dc069b70dc38c013e651e7d1f6d7d65d46c112c5958d98',
    'Hygon':'897b5aadef83c6c4cdab76d2a3d9a24c83e3eb47c6e97a48e7f71e8e24b3515c',
    'Hygon_before':'e6b624e3342df4d84751be5b65c2722137ff17109f2dfb18db8c5fd4c916b98c'}
A100_WEIGHT=(119280712,CP['A100'])
A100_CONFIG=(1127,'04ee0d564d613af9cf104e034a44b6ac52f6f9559e2bf7b611ceb50b02c0ed28')
INITIAL_PT=(119382791,'da110892d3f9cac70993d14e8ea1dfb39bae64631b08051007a276f94e067de7')
HYGON_CONFIG='3453c728728e4eefb7704b62a4e727278f4ebdf9a94f98b67ba146c8be9a16bc'
STRUCTURAL='7bd0b8e46b164ef179ac59a2525322dfb50b6802e3a277c45113a4e9bc6cc8a1'
CAP=32
SHA=re.compile('[0-9a-f]{64}')
IMAGE=re.compile('[0-9a-f]{16}')
FAILURES={'INPUT_UNAVAILABLE','NATIVE_ERROR','EMPTY_RESPONSE','GENERATION_CAP','ARM_AUTHORITY_MISSING',
          'ACTIVATION_NOT_PROVED','EXECUTION_NOT_REACHED','DEADLINE'}


def need(ok,code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return sha256(raw).hexdigest()


def packed(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def unique(pairs):
    result=dict(pairs)
    need(len(result)==len(pairs),'DUPLICATE_JSON_KEY')
    return result


def document(raw):
    return json.loads(raw,object_pairs_hook=unique)


def identities(raw):
    need(sha(raw)==IDENTITIES,'EXACT_OFFICIAL_ORDERED_VAL_IDENTITIES')
    value=document(raw)
    need(set(value)=={'schema','annotation_SHA256','split','version','rows'} and
         (value['schema'],value['annotation_SHA256'],value['split'],value['version'])==
         ('TEXTVQA_V051_VAL_IDENTITY_ONLY_V1',ANN,'val','0.5.1'),'EXACT_ANSWERLESS_IDENTITY_SCHEMA')
    return validate_identities(value['rows'])


def validate_identities(rows):
    seen={}
    for row in rows:
        need(set(row)=={'question_id','image_id'} and type(row['question_id']) is int and row['question_id']>=0
             and row['question_id'] not in seen and IMAGE.fullmatch(row['image_id']),'VALID_UNIQUE_OFFICIAL_IDENTITY')
        seen[row['question_id']]=row['image_id']
    need(len(seen)==QIDS and len(set(seen.values()))==IMAGES,'FROZEN_5000_3166_DENOMINATORS')
    return seen


def input_plan(identity, questions, media):
    """Validate question/image references independently of any checkpoint or runtime."""
    need(set(questions)=={'schema','annotation_SHA256','identity_SHA256','rows'} and
         (questions['schema'],questions['annotation_SHA256'],questions['identity_SHA256'])==
         ('TEXTVQA_V051_VAL_QUESTIONS_WITHOUT_ANSWERS_V1',ANN,IDENTITIES),'EXACT_ANSWERLESS_QUESTION_PROJECTION')
    need(set(media)=={'schema','identity_SHA256','annotations_SHA256','original_64_content_overlap','rows'} and
         (media['schema'],media['identity_SHA256'],media['annotations_SHA256'])==
         ('TEXTVQA_V051_VAL_PRESENT_MEDIA_FOR_INFERENCE_V1',IDENTITIES,ANN) and
         type(media['original_64_content_overlap']) is int and media['original_64_content_overlap']==0,
         'EXACT_READBACK_MEDIA_WITH_TRAINING_CONTENT_DISJOINT')
    by_img={}
    for item in media['rows']:
        img=item['image_id']
        need(set(item)=={'image_id','relative_path','bytes','SHA256'} and img not in by_img and IMAGE.fullmatch(img)
             and item['relative_path'] in (img+'.jpg',img+'.png',img+'.jpeg') and SHA.fullmatch(item['SHA256'])
             and type(item['bytes']) is int and 0<item['bytes']<=64*1024**2,'CANONICAL_PINNED_MEDIA_ROW')
        by_img[img]=item
    need(set(by_img)==set(identity.values()),'ALL_3166_IMAGES_REQUIRED_BEFORE_EXECUTION')
    seen=[];out=[]
    for row in questions['rows']:
        qid=row['question_id']
        need(set(row)=={'question_id','image_id','question'} and type(qid) is int and qid in identity
             and row['image_id']==identity[qid] and type(row['question']) is str and 0<len(row['question'])<=4096
             and row['question'].strip() and '<image>' not in row['question'],'FROZEN_QID_IMAGE_AND_ANSWERLESS_TEXT')
        seen.append(qid)
        out.append((qid,row['question'],by_img[row['image_id']]))
    need(seen==list(identity),'ALL_5000_OFFICIAL_ORDERED_QUESTIONS_REQUIRED')
    return out


def checkpoint_binding(arm, raw=None):
    need(arm in ARMS,'UNKNOWN_OR_BARE_BASE_ARM')
    if arm=='A100':
        return {'weight':A100_WEIGHT,'config':A100_CONFIG,'checkpoint_SHA256':CP[arm],'format':'safetensors'}
    need(raw is not None and sha(raw)==CP['Hygon'],'EXACT_HYGON_ORIGINAL_CHECKPOINT25_MANIFEST')
    value=document(raw)
    files=value['files']
    need(value['step']==25 and value['request_id']=='247fa3682a114542ac3640fbc726aaae' and value['required_members']==len(files)==14
         and files['adapter_config.json']['SHA256']==HYGON_CONFIG,'EXACT_HYGON_CP25_ORIGINAL_MEMBERS')
    def pair(item):
        need(type(item['bytes']) is int and item['bytes']>0 and SHA.fullmatch(item['SHA256']),'ORIGINAL_FILE_PIN_REQUIRED')
        return item['bytes'],item['SHA256']
    return {'weight':INITIAL_PT if arm=='Hygon_before' else pair(files['adapter_model.safetensors']),
            'config':pair(files['adapter_config.json']),'checkpoint_SHA256':CP[arm],
            'format':'target_initial_pt' if arm=='Hygon_before' else 'safetensors'}


def compare_layer_execution(layer, adapter, kwargs):
    need(getattr(layer,'disable_adapters',None) is False and getattr(layer,'merged',None) is False,
         'ACTUAL_LORA_LAYER_DISABLED_OR_MERGED')
    names=kwargs.get('adapter_names')
    if names is None:
        names=getattr(layer,'active_adapters',None)
    need(isinstance(names,(list,tuple)) and list(names)==[adapter]
         and adapter in layer.lora_A and adapter in layer.lora_B,'ACTUAL_LAYER_ROUTE_NOT_REQUESTED_ADAPTER')


def skipped(qid,reason):
    need(reason in FAILURES,'FIXED_FAILURE_REASON_REQUIRED')
    return dict(question_id=qid,status='failed',reason=reason)


def complete_records(identity, records, fallback):
    need(fallback in FAILURES and set(records)<=set(identity),'WRONG_PREDICTION_QID_OR_FALLBACK')
    result=[]
    for qid in identity:
        row=records.get(qid,skipped(qid,fallback))
        need(type(row['question_id']) is int and row['question_id']==qid,'PREDICTION_ID_MISMATCH')
        if row['status']=='scored':
            need(set(row)=={'question_id','status','answer'} and type(row['answer']) is str
                 and 0<len(row['answer'])<=4096 and row['answer'].strip(),'BOUNDED_NATIVE_ANSWER_REQUIRED')
        else:
            need(row==skipped(qid,row['reason']),'INVALID_EXPLICIT_FAILURE_DISPOSITION')
        result.append(row)
    return result


def scorer_manifest(plan_sha, arms):
    need(SHA.fullmatch(plan_sha) and set(arms)==set(ARMS),'ALL_THREE_FROZEN_FORMAL_ARMS')
    for arm,item in arms.items():
        need(set(item)=={'path','SHA256','checkpoint_SHA256'} and item['checkpoint_SHA256']==CP[arm]
             and SHA.fullmatch(item['SHA256']) and isinstance(item['path'],str) and item['path'],
             'PREDICTION_ARM_CHECKPOINT_BINDING')
    return dict(sample_plan_SHA256=plan_sha,arms=arms)
