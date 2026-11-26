"""Check answerless TextVQA retention evidence for the original Gemma domain CP25."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
QIDS, IMAGES, TRAIN = 5000,3166,64
ANN = '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'
MMF = 'e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a'
CP25 = '897b5aadef83c6c4cdab76d2a3d9a24c83e3eb47c6e97a48e7f71e8e24b3515c'
BEFORE = 'e6b624e3342df4d84751be5b65c2722137ff17109f2dfb18db8c5fd4c916b98c'
A100_ADAPTER = 'd6e5a097d491043db0dc069b70dc38c013e651e7d1f6d7d65d46c112c5958d98'
SHA = re.compile(r'[a-f0-9]{64}')
IMAGE = re.compile(r'[a-f0-9]{16}')
PINS = {
    'domain_terminal': ('results/bw1000/2026-11-22/training-gemma-usgs72-v2-original851-eight-rank-25-updates-terminal.json',2889,'53e393df525443d7c5c44b1dee8633f30a389610d9a72dcf988d094be6d260d2'),
    'historical_membership': ('results/bw1000/2026-11-17/training-qwen-generic-control-public-holdout-membership.json',2439,'92839e244bb3bea0bb743014cd61e4d09eff258bbfd3d76314baf41a1fd09b08'),
    'compact_original_membership': ('local-only/training-qwen-generic-public-holdout-v1/public-membership-receipt.json',1788,'849485c7733838bea652508017f37297d9c2ed5dd6151d5f664e3f147a82884d'),
    'historical_checker': ('local-only/training-qwen-generic-public-holdout-v1/check_public_membership.py',8187,'e85af98b501f6867e89ea51a089b99663a30e1dfb4e5b2fcf806eb89d42ec426'),
    'original_training_binding': ('code/bw1000/training-vlm-original-assets-v1/original-bindings.json',20663,'5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822'),
    'training_input_manifest': ('code/bw1000/training-gemma-native-inputs-HIP-CPU-v1/ORIGINAL_INPUT_FILES.json',22227,'85b1cc6b6ec40d76d26f7950c7d2e6d7927af654e32f29e3e86149675787afcf'),
    'a100_terminal': ('results/bw1000/2026-11-11/training-gemma-original-native25-two-updates-and-three-checkpoints-independent.json',164859,'75f1fb844cb4d2ab0d5dba948ab862db597bf7ff912c277266afc2dd78d6e795'),
    'scorer': ('tools/textvqa_effect.py',8965,'89b796e59a542068f6b09b5d4ef8be1747a31a03c3d2bf3df3919c0486eb405b'),
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def unique_object(items):
    value = dict(items)
    need(len(items)==len(value), 'DUPLICATE_JSON_OBJECT_KEY')
    return value


def metadata(path, cap, expected, length=None):
    need(path.is_file() and not path.is_symlink() and SHA.fullmatch(expected), 'EXACT_METADATA_PIN_REQUIRED')
    with path.open('rb') as stream:
        raw = stream.read(cap+1)
    need(0<len(raw)<=cap and (length is None or len(raw)==length) and digest(raw)==expected,
         'BOUNDED_METADATA_DIGEST_MISMATCH')
    return raw


def fixed(key):
    name,size,sha = PINS[key]
    raw = metadata(ROOT/name,size,sha,size)
    if key=='scorer':
        tree = ast.parse(raw)
        return {n.targets[0].id:n.value.value for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    if key=='historical_checker':
        return ast.parse(raw)
    return json.loads(raw,object_pairs_hook=unique_object)


def training_hashes(manifest, binding):
    pins = manifest['files']
    media = [row for row in pins if row['name'].endswith('.jpg')]
    original = binding['common']['image_identity']
    by_name = {row['name']:row for row in media}
    need(len(pins)==67 and len(media)==TRAIN and len(manifest['ordered_image_roles'])==TRAIN
         and len(set(manifest['ordered_image_roles']))==TRAIN
         and len(by_name)==len(original)==TRAIN
         and all(type(row['bytes']) is int and row['bytes']>0 and SHA.fullmatch(row['SHA256']) for row in media)
         and manifest['ordered_image_roles']==[row['relative_role'] for row in original]
         and all(row['ordered_index']==i and (by_name['datasets/'+row['relative_role']]['bytes'],
                     by_name['datasets/'+row['relative_role']]['SHA256'])==(row['bytes'],row['sha256'])
                 for i,row in enumerate(original)),
         'EXACT_ORIGINAL_64_TRAIN_JPEG_MANIFEST')
    return {row['SHA256'] for row in media}


def projection(supplied):
    expected = {'schema':'TEXTVQA_V051_VAL_IDENTITY_ONLY_V1','annotation_SHA256':ANN,'split':'val','version':'0.5.1'}
    need(set(supplied)=={*expected,'rows'} and all(supplied[k]==v for k,v in expected.items()),
         'WRONG_OR_NON_IDENTITY_ONLY_ANNOTATION_PROJECTION')
    qids,images=set(),set()
    for row in supplied['rows']:
        need(set(row)=={'question_id','image_id'} and type(row['question_id']) is int
             and row['question_id']>=0 and row['question_id'] not in qids
             and isinstance(row['image_id'],str) and IMAGE.fullmatch(row['image_id']), 'BAD_OR_DUPLICATE_IDENTITY_ROW')
        qids.add(row['question_id']); images.add(row['image_id'])
    need(len(qids)==QIDS and len(images)==IMAGES, 'WRONG_FROZEN_QUESTION_OR_IMAGE_DENOMINATOR')
    return qids,images


def media_evidence(supplied, projection_sha, images, train):
    expected = {'schema':'TEXTVQA_V051_VAL_MEDIA_READBACK_V1','annotation_SHA256':ANN,
                'identities_SHA256':projection_sha,'all_local_asset_bytes_verified':True}
    need(set(supplied)=={*expected,'original_readback_SHA256','asset_manifest_SHA256','rows'}
         and all(supplied[k]==v for k,v in expected.items()) and supplied['all_local_asset_bytes_verified'] is True
         and SHA.fullmatch(supplied['original_readback_SHA256']) and SHA.fullmatch(supplied['asset_manifest_SHA256']),
         'EXACT_MEDIA_READBACK_RECEIPT_REQUIRED')
    by_id={}
    for row in supplied['rows']:
        need(set(row)=={'image_id','bytes','SHA256','readable'} and row['image_id'] in images
             and row['image_id'] not in by_id and row['readable'] is True
             and type(row['bytes']) is int and row['bytes']>0 and SHA.fullmatch(row['SHA256']),
             'UNAVAILABLE_OR_INVALID_FROZEN_IMAGE_ID')
        by_id[row['image_id']]=row['SHA256']
    need(by_id.keys()==images, 'MEDIA_COVERAGE_MISSING_FROZEN_IMAGES')
    shared=[sha for sha in by_id.values() if sha in train]
    return dict(images_confirmed_available_in_pinned_receipt=len(by_id),
                validation_unique_encoded_media_hashes=len(set(by_id.values())),
                overlapping_validation_image_ids=len(shared),overlapping_distinct_encoded_hashes=len(set(shared)),
                exact_content_disjoint=not shared,decoded_pixel_or_near_duplicate_disjoint=None)


def report(inputs, identity=None, identity_sha=None, media=None, media_sha=None):
    domain=inputs['domain_terminal']
    need(domain['passed'] is True and domain['actual_updates']==25 and domain['actual_steps'][1]=='1103606.851'
         and domain['checkpoint_manifests']['25']=={'bytes':2977,'SHA256':CP25}
         and domain['fresh_initial_named_SHA256']==BEFORE, 'WRONG_ORIGINAL_CP25_OR_BEFORE')
    a100=inputs['a100_terminal']
    reference=a100['source_reference']
    cp, = [row for row in reference['checkpoints'] if row['global_step']==25]
    need(a100['passed'] is a100['independent_original_CPU_reader_terminal_verified'] is True
         and reference['original25_completed_ranks']==8 and cp['optimizer_step_min']==cp['optimizer_step_max']==25
         and cp['required_state_files']['adapter_model.safetensors']=={'bytes':119280712,'SHA256':A100_ADAPTER},
         'WRONG_FORMAL_A100_SOURCE_CHECKPOINT')
    historical=inputs['historical_membership']['textvqa_0.5.1']['val']
    compact=inputs['compact_original_membership']
    need(historical==compact['textvqa_0.5.1']['val'] and historical['source_SHA256']==ANN
         and historical['questions']==QIDS and historical['openimages_ids']==IMAGES
         and historical['training_images_with_flickr_source']==TRAIN
         and historical['openimages_matching_any_training_flickr_photo_id']==0
         and historical['content_SHA_intersection_measured'] is False and compact['source_images_downloaded'] is False,
         'WRONG_HISTORIC_AGGREGATE_OR_SOURCE_ID_SCOPE')
    aggregate=inputs['historical_membership']
    checker, = [fn for fn in inputs['historical_checker'].body if isinstance(fn,ast.FunctionDef) and fn.name=='textvqa']
    need(aggregate['checker_SHA256']==PINS['historical_checker'][2]
         and aggregate['original_binding_SHA256']==compact['original_binding_SHA256']==PINS['original_training_binding'][2]
         and any(ast.unparse(node)=='len(qids) != len(rows)' for node in ast.walk(checker)),
         'HISTORICAL_UNIQUE_QUESTION_OR_TRAINING_ID_BINDING_MISSING')
    need(inputs['scorer']['ANNOTATIONS_SHA256']==ANN and inputs['scorer']['MMF_SHA256']==MMF,
         'FROZEN_OFFICIAL_TEXTVQA_SCORER_MISMATCH')
    train=training_hashes(inputs['training_input_manifest'],inputs['original_training_binding'])
    blockers=[]
    media_result=None
    rechecked=None
    need(media is None or identity is not None, 'MEDIA_WITHOUT_EXACT_IDENTITY_PROJECTION')
    if identity is None:
        blockers.append('MISSING_AUTHORIZED_5000_QID_TO_3166_IMAGE_ID_ANSWERLESS_PROJECTION')
    else:
        qids,images=projection(identity)
        rechecked=dict(unique_question_ids=len(qids),mapped_image_ids=len(images),SHA256=identity_sha)
    if media is None:
        blockers.append('MISSING_PINNED_3166_MEDIA_BYTE_READBACK_AND_AVAILABILITY_RECEIPT')
    else:
        media_result=media_evidence(media,identity_sha,images,train)
        media_result['receipt_SHA256']=media_sha
        if not media_result['exact_content_disjoint']:
            blockers.append('VAL_MEDIA_ENCODED_CONTENT_OVERLAPS_ORIGINAL_64_TRAINING_IMAGES')
    ready_inputs=not blockers
    blockers.append('MISSING_OFFICIAL_A100_SAME_5000_QID_GENERATION_AND_SCORING_BINDING')
    return dict(scope='INDEPENDENT_GEMMA_USGS_DOMAIN_CP25_RETENTION_INPUT_ADMISSION',source_step='1103606.851',
                R119_derived=False, status='INPUT_EVIDENCE_ONLY' if ready_inputs else 'INPUT_ADMISSION_BLOCKED',
                planned_questions=QIDS,planned_images=IMAGES,original_training_images=TRAIN,
                original_training_unique_encoded_hashes=len(train),
                historical_report=dict(unique_question_ids=historical['questions'],unique_image_ids=historical['openimages_ids'],
                                       flickr_source_id_overlap=0,content_overlap_measured=False),
                current_identity_recheck=rechecked,current_media_and_exact_encoded_content_overlap=media_result,
                checkpoints=dict(Hygon={'kind':'checkpoint_manifest','SHA256':CP25},
                                 Hygon_before={'kind':'target_created_named_parameters','SHA256':BEFORE},
                                 A100={'kind':'original_native_CP25_adapter_file','SHA256':A100_ADAPTER,
                                       'same_TextVQA_comparator_evidence':None}),
                annotations_source_SHA256=ANN, scorer_MMF_SHA256=MMF,
                input_admission_passed=ready_inputs, formal_retention_passed=False, GPU_GO=False,
                READY=False, actual_GPU=0, annotation_answers_opened=False, raw_media_opened=False,
                blockers=blockers, evidence={key:dict(path=name,bytes=size,SHA256=sha)
                                            for key,(name,size,sha) in PINS.items()})


def optional(path, pin, basename):
    need((path is None)==(pin is None), 'SUPPLY_BOTH_EXACT_METADATA_PATH_AND_PIN')
    if path is None:
        return None
    need(path.name==basename, 'ONLY_THE_NAMED_ANSWERLESS_METADATA_ARTIFACT_IS_ALLOWED')
    return json.loads(metadata(path,1<<20,pin),object_pairs_hook=unique_object)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identities',type=Path)
    parser.add_argument('--identities-sha256')
    parser.add_argument('--media-receipt',type=Path)
    parser.add_argument('--media-sha256')
    args=parser.parse_args()
    try:
        inputs={key:fixed(key) for key in PINS}
        identity=optional(args.identities,args.identities_sha256,'TEXTVQA_VAL_051_IDENTITIES.json')
        media=optional(args.media_receipt,args.media_sha256,'TEXTVQA_VAL_051_MEDIA_READBACK.json')
        result=report(inputs,identity,args.identities_sha256,media,args.media_sha256)
    except (OSError,ValueError,TypeError,KeyError,SyntaxError) as error:
        print(json.dumps(dict(status='INVALID_OR_INACCESSIBLE_METADATA', error_type=type(error).__name__,
                              error_code=str(error) if type(error) is ValueError and re.fullmatch('[A-Z0-9_]+',str(error)) else None,
                              input_admission_passed=False, formal_retention_passed=False, GPU_GO=False,READY=False)))
        return 1
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['input_admission_passed'] else 2


if __name__=='__main__':
    sys.exit(main())
