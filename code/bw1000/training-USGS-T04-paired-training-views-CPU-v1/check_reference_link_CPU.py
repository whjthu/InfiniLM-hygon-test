"""New reference-link boundaries only; do not rerun old exposure/native tests."""
from copy import deepcopy
import hashlib
import json
from views_cpu import references


def run():
    rows=[];native=[];field=dict(SHA256='1'*64,finite=True,elements=20)
    for i in range(45):
        identifier='synthetic'+str(i);rows.append(dict(id=identifier,messages=[dict(role='user',content='synthetic'),dict(role='assistant',content='FINAL: synthetic')],images=['/synthetic/train/media/one.png']))
        native.append(dict(split='train',ordinal=i,source_id_SHA256=hashlib.sha256(identifier.encode()).hexdigest(),full_input_retained=True,labels_None=False,supervised_EOS_count=1,native_EOS_ids=[2],
            full_length=20,first_supervised_index=10,encoded_fields=dict(input_ids=field,labels=field),collated_fields={k:field for k in ['input_ids','labels','attention_mask','pixel_values']}))
    binding=dict(target_data='/synthetic',media={'train/media/one.png':dict(SHA256='2'*64)})
    refs=references(rows,native,binding);assert len(refs)==45 and refs[0]['identity_candidate_only'] is True and refs[0]['actual_GPU_rank_cursor_or_padding_match_qualified'] is False
    rejects=[]
    def check(name,r,n):
        try:references(r,n,binding)
        except ValueError:rejects.append(name);return
        raise AssertionError(name)
    x=deepcopy(native);x[0]['source_id_SHA256']='0'*64;check('wrong_native_source',rows,x)
    x=deepcopy(native);x[0]['full_input_retained']=False;check('truncated_reference',rows,x)
    x=deepcopy(rows);x[0]['messages'][1]['content']='synthetic';check('format_mismatch',x,native)
    x=deepcopy(rows);x[0]['images']=['/synthetic/dev/media/one.png'];check('dev_image_in_train',x,native)
    return dict(passed=True,normal=1,rejected=len(rejects),rejected_cases=rejects,actual_source_rows_read=0,actual_views_generated=0,old_exposure_tests_rerun=0,new_encodes=0,model_or_GPU_calls=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
