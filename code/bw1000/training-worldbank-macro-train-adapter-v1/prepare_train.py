"""Train-only source join for the common Swift dataset argument, never a training loop."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import stat

from framework_contract import inspect_sources

G=Path(__file__).resolve().parents[3]
PLAN=G/'code/bw1000/training-worldbank-macro-chart-v1/PLAN.json'
PLAN_SHA='c49d86cb991e9b3d10d08a0b953e63d44349f4e76fc6f16d1f855b676c97f2ec'


def need(value,why):
    if not value:raise ValueError(why)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()


def js(raw):
    def pairs(items):need(len(items)==len({k for k,v in items}),'Duplicate JSON key');return dict(items)
    def nonfinite(value):raise ValueError('Nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)


def bounded(path,cap,expected=None):
    need(path.is_file() and not path.is_symlink() and not getattr(path,'is_junction',lambda:False)(),'Nonregular source')
    before=path.stat();need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'Source cap')
    raw=path.read_bytes();after=path.stat()
    need((before.st_size,before.st_ino,before.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns),'Source changed')
    need(expected is None or sha(raw)==expected,'Source identity mismatch');return raw


def join_train(inputs,labels,charts,plan,target_root):
    root=PurePosixPath(target_root)
    need(root.is_absolute() and '..' not in root.parts and root.name=='derived' and not {'dev','blind'}.intersection(root.parts),'Unsafe target derived root')
    need(len(inputs)==len(labels)==216 and len({x['id'] for x in inputs})==len({x['id'] for x in labels})==216,'Full216 train slots required; no partial selection')
    need(len(charts)==24 and all(c['split']=='train' and c['status']=='QUALIFIED' for c in charts),'Full24 qualified train charts required')
    by_chart={c['id']:c for c in charts};need(len(by_chart)==24,'Duplicate train chart')
    wanted=set()
    for country in plan['splits']['train']['countries']:
        for indicator in plan['indicators']:
            wanted.add('wdi:'+country+':'+indicator+':2000-2006')
    need(set(by_chart)==wanted,'Chart group outside frozen training countries/years')
    gold={x['id']:x for x in labels};need(set(gold)=={x['id'] for x in inputs},'Input/gold ID membership differs')
    rows=[];ids=[];images={};counts={}
    # Preserve frozen builder input order, never shuffle/filter by answer values.
    for source in inputs:
        need(set(source)=={'id','messages','images'},'Input has hidden labels/extra fields')
        label=gold[source['id']];chart=by_chart[label['chart_group']]
        need(label['data_status']=='QUALIFIED' and label['country_group']==chart['country_group'] and label['indicator']==chart['indicator'],'Unqualified/foreign gold')
        need(source['images']==[chart['image']['name']] and len(source['messages'])==1,'Input/media binding differs')
        user=source['messages'][0];need(set(user)=={'role','content'} and user['role']=='user' and user['content']=='<image>\n'+label['question'] and user['content'].count('<image>')==1,'User question identity differs')
        relative=PurePosixPath(chart['image']['name'])
        need(not relative.is_absolute() and '..' not in relative.parts and relative.parts[:2]==('train','media') and relative.suffix=='.png','Nontrain image path')
        answer=label['answer'];pattern=r'[0-9]{4}' if label['answer_type']=='year' else r'-?(?:0|[1-9][0-9]*)\.[0-9]{2}'
        need(label['answer_type'] in ('year','decimal2') and isinstance(answer,str) and len(answer)<=32 and re.fullmatch(pattern,answer) is not None,'Gold format differs')
        kind,index=source['id'].rsplit(':',2)[-2:]
        need(kind in ('lookup','change','extrema') and index in ('0','1','2') and source['id']==chart['id']+':'+kind+':'+index and label['kind']==kind,'Question slot identity')
        need(label['answer_type']==('year' if kind=='extrema' and index=='2' else 'decimal2'),'Answer type/slot mismatch')
        # Literal assistant text only; EOS token and loss masks belong to the existing native template.
        rows.append(dict(messages=[dict(user),dict(role='assistant',content='FINAL: '+answer)],images=[str(root/relative)]))
        ids.append(source['id']);images[relative.as_posix()]=chart['image'];counts[kind]=counts.get(kind,0)+1
    need(len(images)==24 and len({p['SHA256'] for p in images.values()})==24 and counts==dict(lookup=72,change=72,extrema=72),'Training coverage/duplicate content differs')
    return rows,dict(rows=216,unique_images=24,question_kind_counts=counts,ordered_source_ids_SHA256=sha(canonical(ids)),
        assistant_format='FINAL: <frozen answer>',literal_special_tokens_added=False,train_only=True,
        development_or_blind_gold_read=False,mask_and_EOS_owner='existing native qwen3_vl template/collator; actual WDI qualification pending')


def main(args):
    need(not args.output.exists(),'Output namespace already exists');args.output.mkdir()
    receipt=dict(passed=False,planned_train_rows=216,planned_train_images=24,Dataset_GO=False,training_GO=False,GPU_GO=False,actual_updates=0)
    try:
        interface=inspect_sources(bounded(args.common_source,65536),bounded(args.vlm_source,65536))
        proof=js(bounded(args.reader_result,4194304,args.reader_sha))
        need(proof['reader_passed'] is True and proof['producer_passed'] is True and proof['known_terminal'] is True
            and proof['state']=='TRAIN_DEV_SOURCE_LABEL_AND_PASSIVE_MEDIA_CHECKED_BLIND_UNOPENED'
            and proof['blind_media_or_gold_opened'] is False,'Independent source reader not qualified')
        root=args.run.resolve();need(root.name!='blind' and root.name!='dev','Nontrain source root')
        def get(name,cap):
            pin=proof['original_files'][name];return bounded(root/name,cap,pin['SHA256'])
        plan=js(bounded(PLAN,16384,PLAN_SHA));manifest=js(get('derived/manifest-private.json',2097152))
        inputs=[js(x) for x in get('derived/train/inputs.jsonl',2097152).splitlines()]
        labels=[js(x) for x in get('derived/train/labels-private.jsonl',2097152).splitlines()]
        charts=[c for c in manifest['charts'] if c['split']=='train']
        rows,stats=join_train(inputs,labels,charts,plan,args.target_derived_root)
        for c in charts:
            path=root/'derived'/c['image']['name'];need(path.resolve().is_relative_to(root/'derived/train/media'),'Image path escaped training media')
            raw=get('derived/'+c['image']['name'],393216);need(len(raw)==c['image']['bytes'] and sha(raw)==c['image']['SHA256'],'Actual image identity differs')
        data=b''.join(canonical(x) for x in rows);(args.output/'train-swift.jsonl').write_bytes(data)
        receipt.update(passed=True,state='TRAIN_ONLY_SWIFT_DATASET_PREPARED_NATIVE_NOT_RUN',**stats,dataset_SHA256=sha(data),dataset_bytes=len(data),
            independent_reader_SHA256=args.reader_sha,framework_interface=interface,
            source_train_inputs_SHA256=proof['original_files']['derived/train/inputs.jsonl']['SHA256'],
            source_train_labels_SHA256=proof['original_files']['derived/train/labels-private.jsonl']['SHA256'],
            source_train_image_pins={c['image']['name']:c['image'] for c in charts},
            target_derived_root=args.target_derived_root,current_target_framework_installed=False,
            separate_domain_recipe='qwen3vl.wdi-macro-chart-sft-v1',formal_T01_case_modified=False,
            max_length=4096,max_pixels=262144,template='qwen3_vl',model_type='qwen3_vl',
            native_token_lengths_mask_EOS_qualified=False,model_base_SHA256=None,initial_trainables_SHA256=None,
            initial_trainables_scope='fresh WDI recipe; same explicit base/initial LoRA state must be frozen before any candidate training',
            formal_8card_BF16_batch_precision_changed=False,performance_reference='new separate domain recipe required',
            pending=['target common-framework/current callback closure','native full216 encode/collator/mask/EOS','base/initial trainables identity','numerical first update','continuous25/full-state fresh-process resume','paired domain/generalization and blind effects'])
    except Exception as error:receipt.update(state='FAILED_NO_PARTIAL_TRAINING_SELECTION',error_type=type(error).__name__,error=str(error))
    (args.output/'receipt-private.json').write_bytes(canonical(receipt))
    print(json.dumps(dict(passed=receipt['passed'],state=receipt['state'],planned_train_rows=216,actual_updates=0)))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('run','output','reader-result','common-source','vlm-source'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reader-sha',required=True);p.add_argument('--target-derived-root',required=True)
    a=p.parse_args();need(re.fullmatch('[0-9a-f]{64}',a.reader_sha) is not None,'Bad reader SHA');raise SystemExit(main(a))
