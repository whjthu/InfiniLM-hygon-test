"""Synthetic join failure boundaries and actual exact common-interface AST only."""
import copy
import hashlib
import json
from pathlib import Path

from framework_contract import inspect_sources
from prepare_train import PLAN,PLAN_SHA,bounded,canonical,join_train,js,need


def run():
    G=Path(__file__).resolve().parents[3]
    common=G/'code/bw1000/training-qwen-target-unified-framework-a68952c-v1/deltas/tools/hygon_training_recipe.py'
    vlm=G/'code/bw1000/training-qwen-target-native-CPU-v1/deltas/tools/hygon_vlm_recipe.py'
    interface=inspect_sources(bounded(common,65536),bounded(vlm,65536))
    plan=js(bounded(PLAN,16384,PLAN_SHA));inputs=[];labels=[];charts=[]
    for country in plan['splits']['train']['countries']:
        for indicator in plan['indicators']:
            chart='wdi:'+country+':'+indicator+':2000-2006';image='train/media/'+str(len(charts))+'.png'
            charts.append(dict(id=chart,split='train',status='QUALIFIED',country_group=country,indicator=indicator,
                image=dict(name=image,bytes=1,SHA256=hashlib.sha256(chart.encode()).hexdigest())))
            for kind in ('lookup','change','extrema'):
                for i in range(3):
                    identifier=chart+':'+kind+':'+str(i);year=kind=='extrema' and i==2
                    labels.append(dict(id=identifier,data_status='QUALIFIED',chart_group=chart,country_group=country,indicator=indicator,
                        kind=kind,answer_type='year' if year else 'decimal2',answer='2000' if year else '-1.00',question='Synthetic fixture only, no official fact.'))
                    inputs.append(dict(id=identifier,messages=[dict(role='user',content='<image>\n'+labels[-1]['question'])],images=[image]))
    root='/refactor/results/NEW-WDI-NAMESPACE/derived';rows,stats=join_train(inputs,labels,charts,plan,root)
    need(len(rows)==216 and stats['unique_images']==24 and all(set(r)=={'messages','images'} and [m['role'] for m in r['messages']]==['user','assistant'] and r['messages'][1]['content'].startswith('FINAL: ') for r in rows),'Common Swift output schema')
    rejected=[]
    def bad(name,i,l,c,r=root):
        try:join_train(i,l,c,plan,r)
        except (ValueError,KeyError):rejected.append(name);return
        raise AssertionError('Accepted '+name)
    bad('missing_training_slot',inputs[:-1],labels,charts)
    x=copy.deepcopy(inputs);x[1]=x[0];bad('duplicate_input_id',x,labels,charts)
    x=copy.deepcopy(charts);x[0]['split']='blind';bad('blind_chart_group',inputs,labels,x)
    x=copy.deepcopy(charts);x[0]['image']['name']='dev/media/0.png';bad('dev_media_path',inputs,labels,x)
    x=copy.deepcopy(labels);x[0]['data_status']='MISSING_SOURCE_VALUES';bad('failed_gold_no_backfill',inputs,x,charts)
    x=copy.deepcopy(inputs);x[0]['answer']='leaked';bad('input_hidden_gold',x,labels,charts)
    x=copy.deepcopy(inputs);x[0]['messages'].append(dict(role='assistant',content='leak'));bad('preexisting_assistant',x,labels,charts)
    x=copy.deepcopy(labels);x[0]['answer']='<|im_end|>';bad('gold_special_token',inputs,x,charts)
    x=copy.deepcopy(charts);x[0]['image']['SHA256']=x[1]['image']['SHA256'];bad('duplicate_media_content',inputs,labels,x)
    bad('target_blind_namespace',inputs,labels,charts,'/refactor/results/blind/derived')
    return dict(passed=True,synthetic_rows=216,synthetic_media_groups=24,normal_contracts=1,rejected_count=len(rejected),rejected=rejected,
        actual_framework_source_AST=interface,real_WDI_inputs_or_labels_read=0,blind_opened=False,real_media_read=0,
        native_encode_calls=0,training_updates=0,HTTP=0,Dataset_GO=False,training_GO=False,GPU_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
