"""Only new control format/order/schema boundaries; synthetic, no raw caption source."""
from copy import deepcopy
import json
from prepare_inputs import transform_selected


def run():
    rows=[dict(messages=[dict(role='user',content='<image> synthetic prompt '+str(i)),dict(role='assistant',content='synthetic caption '+str(i))],images=['/synthetic/source/'+str(i)+'.jpg']) for i in range(64)]
    selected=[dict(original_ordinal=i,source=rows[i]['images'][0],image_SHA256=('%064x'%i)) for i in range(45)]
    result,ledger=transform_selected(rows,selected,'/synthetic/new/data')
    assert len(result)==len(ledger)==45 and result[0]['messages'][1]['content']=='FINAL: synthetic caption 0' and result[-1]['messages'][0]['content'].endswith('44')
    assert rows[0]['messages'][1]['content']=='synthetic caption 0'
    source=deepcopy(rows);source[0]['messages'][1]['content']='FINAL: original';one,_=transform_selected(source,selected,'/synthetic/new/data');assert one[0]['messages'][1]['content']=='FINAL: original'
    rejects=[]
    def bad(name,r,s):
        try:transform_selected(r,s,'/synthetic/new/data')
        except ValueError:rejects.append(name);return
        raise AssertionError(name)
    bad('not64',rows[:-1],selected);bad('not45',rows,selected[:-1])
    altered=deepcopy(selected);altered[1]['original_ordinal']=0;bad('repeated_ordinal',rows,altered)
    altered=deepcopy(selected);altered[1]['image_SHA256']=altered[0]['image_SHA256'];bad('repeated_content',rows,altered)
    altered=deepcopy(rows);altered[0]['images']=['/wrong.jpg'];bad('wrong_image_association',altered,selected)
    altered=deepcopy(rows);altered[0]['messages'][1]['role']='user';bad('not_assistant',altered,selected)
    altered=deepcopy(rows);altered[0]['messages'][1]['content']='';bad('empty_caption',altered,selected)
    return dict(passed=True,normal=3,rejected=len(rejects),rejected_cases=rejects,actual_original_source_caption_reads=0,actual_image_reads=0,actual_native_calls=0,old_wrapper_or_science_tests_rerun=0,HTTP_GETs=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
