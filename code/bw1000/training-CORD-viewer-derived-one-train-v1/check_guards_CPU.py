"""New one-row safety/schema guards only, no network/Pillow/model or real data."""
import copy
from email.message import Message
import json
from pathlib import Path
from once import asset_gate,canonical,contract_gate,headers_gate,need,row_gate


def run():
    c=json.loads((Path(__file__).parent/'CONTRACT.json').read_bytes());contract_gate(c)
    src='https://datasets-server.huggingface.co/assets/'+c['dataset']+'/--/'+c['revision']+'/--/default/train/795/image/image.jpg'
    gold=dict(meta=dict(version='2.0.0',split='train',image_id=123,image_size=dict(width=20,height=10)),gt_parse=dict(menu=dict(nm='SYNTHETIC'),total=dict(total_price='1.00')))
    row=dict(row_idx=795,truncated_cells=[],row=dict(image=dict(src=src,width=20,height=10),ground_truth=canonical(gold).decode()))
    payload=dict(partial=False,num_rows_total=800,rows=[row],features=[dict(name='image',type=dict(_type='Image')),dict(name='ground_truth',type=dict(_type='Value',dtype='string'))])
    g,s,d=row_gate(payload,c);need(d==(20,10) and g['meta']['image_id']==123,'Row ordinal incorrectly substituted as actual image_id')
    signed=src+'?Expires=4102444800&Signature=SYNTHETIC&Key-Pair-Id=SYNTHETIC';asset_gate(signed,payload['rows'][0]['row']['image'],gold,c)
    h=Message();h['Content-Type']='application/json';h['Content-Length']='500';h['X-Revision']=c['revision'];need(headers_gate(200,h,'row',c)==500,'Good header')
    rejected=[]
    def bad(n,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):rejected.append(n);return
        raise AssertionError('Accepted '+n)
    missing=Message();missing['Content-Type']='application/json';bad('no_revision_0media',lambda:headers_gate(200,missing,'row',c))
    wrong=Message();wrong['Content-Type']='application/json';wrong['X-Revision']='0'*40;bad('wrong_revision_0media',lambda:headers_gate(200,wrong,'row',c))
    duplicate=Message();duplicate['Content-Type']='application/json';duplicate['X-Revision']=c['revision'];duplicate['X-Revision']=c['revision'];bad('duplicate_revision',lambda:headers_gate(200,duplicate,'row',c))
    bad('redirect_no_body',lambda:headers_gate(302,h,'row',c))
    over=Message();over['Content-Type']='application/json';over['Content-Length']=str(c['row_cap_bytes']+1);over['X-Revision']=c['revision'];bad('row_cap_before_body',lambda:headers_gate(200,over,'row',c))
    x=copy.deepcopy(payload);x['rows'][0]['row_idx']=794;bad('different_row',lambda:row_gate(x,c))
    x2=copy.deepcopy(payload);x2['rows'][0]['truncated_cells']=['ground_truth'];bad('truncated_gt',lambda:row_gate(x2,c))
    x3=copy.deepcopy(payload);x3['rows'][0]['row']['image']['width']=19;bad('viewer_resize_before_media',lambda:row_gate(x3,c))
    bad('other_domain',lambda:asset_gate(src.replace('datasets-server.huggingface.co','example.org'),row['row']['image'],gold,c))
    bad('different_cache_revision',lambda:asset_gate(src.replace(c['revision'],'0'*40),row['row']['image'],gold,c))
    bad('thumbnail_path',lambda:asset_gate(src.replace('image.jpg','thumbnail.jpg'),row['row']['image'],gold,c))
    bad('expired_signed_asset',lambda:asset_gate(src+'?Expires=1&Signature=SYNTHETIC&Key-Pair-Id=SYNTHETIC',row['row']['image'],gold,c))
    bad('unexpected_query',lambda:asset_gate(src+'?token=SYNTHETIC',row['row']['image'],gold,c))
    x4=copy.deepcopy(c);x4['ordinal']=0;bad('selection_seed_drift',lambda:contract_gate(x4))
    return dict(passed=True,normal_contracts=4,rejected_count=len(rejected),rejected=rejected,actual_row_GETs=0,actual_media_GETs=0,
        real_gt_read=0,model_calls=0,old_tests_repeated=False,pixel_decode_tested=False,Dataset_GO=False,training_GO=False,GPU_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
