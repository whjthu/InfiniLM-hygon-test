"""Synthetic roster/identity boundaries only; no real captions/media/scorer/network."""
import argparse
import copy
import json
from pathlib import Path

from prepare_caption_dev import select,require_new_sha


def check():
    protected=dict(ids={1},photos={'999'},image_SHA256={'a'*64})
    licenses=[dict(id=4,url='http://creativecommons.org/licenses/by/2.0/')]
    def image(i,w,h,**extra):
        value=dict(id=i,file_name=f'{i:012d}.jpg',coco_url=f'http://images.cocodataset.org/train2017/{i:012d}.jpg',
            flickr_url=f'http://farm1.staticflickr.com/1/{90000000000+i}_abcdef.jpg',license=4,width=w,height=h)
        value.update(extra);return value
    images=[image(1000+i,640,480) for i in range(12)]+[image(2000+i,480,640) for i in range(8)]+[image(3000+i,500,500) for i in range(8)]
    selected,receipt=select(images,licenses,protected)
    reversed_selected,reversed_receipt=select(list(reversed(images)),licenses,protected)
    assert selected==reversed_selected and receipt==reversed_receipt
    assert len(selected)==16 and receipt['selected_by_stratum']=={'landscape':8,'portrait':4,'square':4}
    cases=[]
    def reject(name,fn):
        try:fn()
        except ValueError:cases.append(name)
        else:raise AssertionError('accepted '+name)
    reject('duplicate_COCO_id',lambda:select(images+[copy.deepcopy(images[0])],licenses,protected))
    reject('license_URL_changed',lambda:select(images,[dict(id=4,url='https://example.invalid/no-license')],protected))
    reject('original64_SHA',lambda:require_new_sha('a'*64,protected))
    reject('invalid_SHA',lambda:require_new_sha('no-hash',protected))
    require_new_sha('b'*64,protected)
    extras=[image(1,640,480),image(4000,640,480,flickr_url='http://farm1.staticflickr.com/1/999_abcdef.jpg'),
            image(4001,640,480,license=1),image(4002,0,480),
            image(4003,640,480,coco_url='http://images.cocodataset.org/val2017/000000004003.jpg'),
            image(4004,640,480,flickr_url='https://unknown.example/photo')]
    _,blocked=select(images+extras,licenses,protected)
    for key in ['original64_COCO_ID_overlap','original64_Flickr_group_overlap','outside_frozen_CC_BY2_image_policy',
                'invalid_dimensions','wrong_train_membership_or_URL','source_group_unknown']:
        assert blocked['rejected'][key]==1;cases.append(key)
    duplicates=copy.deepcopy(images)
    duplicates[1]['flickr_url']=duplicates[0]['flickr_url']
    output,stats=select(duplicates,licenses,protected)
    assert len({row['flickr_photo_id'] for row in output})==len(output)
    assert stats['rejected']['candidate_source_group_duplicate']==1
    short,shortstats=select(images[:3],licenses,protected)
    assert len(short)==3 and shortstats['planned']==16 and shortstats['missing_slots']==13
    # With labels absent from select's API, altered/empty reference text cannot alter the roster.
    assert receipt['selection_uses_captions_or_model_outputs'] is False
    return dict(state='LOCAL_SYNTHETIC_SELECTION_AND_EXCLUSION_BOUNDARIES_PASS',
        synthetic_pool_images=len(images),synthetic_selected=16,normal_order_invariance=True,
        rejected_or_excluded_cases=cases,rejected_or_excluded_count=len(cases),
        duplicate_source_group_cases=1,shortage_missing_slots=13,
        real_train2017_images_selected=0,real_images_read=0,real_captions_selected=0,
        CIDEr_evaluated=False,network_calls=0,GPU_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();value=check()
    with a.output.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=value['state'],cases=value['rejected_or_excluded_count'],real_selected=0,CIDEr_evaluated=False)))
