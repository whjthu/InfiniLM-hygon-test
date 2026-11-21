"""Consume actual source images once; preserve split/failed denominators."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from native_reward import load_native, need, pin, score_batch4

D = Path(__file__).resolve().parent
G = D.parents[2]
DATA = G / 'local-only/training-zimage-USGS-style18-dataset-v1-original/dataset'
OUT = G / 'local-only/training-zimage-flow-grpo-real18-reward-input-v1-original'


def read_fixed(path, expected):
    need(not path.is_symlink(), 'Unexpected aliased source')
    with path.open('rb') as f:
        before=os.fstat(f.fileno());raw=f.read(expected['bytes']+1);after=os.fstat(f.fileno())
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(ident(before)==ident(after)==ident(path.stat()), 'Source changed during read')
    need(pin(raw)==expected, 'Frozen source identity differs')
    return raw


def save(name, value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    started=time.monotonic();OUT.mkdir()
    plan=json.loads((D/'PLAN.json').read_bytes())
    save('intent.json',{'request_id':plan['request_id'],'wall_start':time.time(),'plan':pin((D/'PLAN.json').read_bytes()),
        'script':pin(Path(__file__).read_bytes()),'native_adapter':pin((D/'native_reward.py').read_bytes())})
    r={'schema':'ZIMAGE_FLOW_GRPO_REAL_REFERENCE_IMAGE_REWARD_INPUT_LOCAL_V1','request_id':plan['request_id'],
       'passed':False,'planned_reference_images':18,'completed_reference_images':0,'failed_reference_images':0,
       'actual_original_reward_image_scores':0,'actual_flip_diagnostic_scores':0,'actual_policy_rollouts':0,
       'actual_optimizer_updates':0,'actual_model_loads':0,'actual_GPU_or_Slurm_or_SSH_or_HTTP_calls':0,
       'training_GO':False,'GPU_GO':False,'Model_GO':False,'formal17_GO':False,'READY':False,'splits':{}}
    try:
        import numpy as np
        from PIL import Image, ImageOps
        native, binding=load_native(G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz')
        r['native_function']=binding
        b=json.loads(read_fixed(DATA/'data-binding-private.json',plan['data_binding']))
        need(b['QA_or_gold_read'] is False and b['blind_body_read'] is False,'Original input provenance differs')
        groups={};events={};pixels={};first4=[];first4scores=[]
        for split,count in (('train',12),('dev',6)):
            selected=[x for x in b['images'] if x['split']==split]
            need(len(selected)==count,'Full source split denominator differs')
            item={'planned':count,'completed':0,'failed':0,'rows':[]};r['splits'][split]=item
            groups[split]=set();events[split]=set();pixels[split]=set();scores=[];flipdiff=[]
            for index,x in enumerate(selected):
                entry={'ordinal':index,'passed':False,'image':x['dataset_relative']};item['rows'].append(entry)
                try:
                    need(time.monotonic()-started<45,'Single local input audit budget elapsed')
                    relative=Path(x['dataset_relative']);need(not relative.is_absolute() and '..' not in relative.parts
                         and relative.parts[0]==split,'Image outside its frozen source split')
                    raw=read_fixed(DATA/relative,x['source'])
                    with Image.open(io.BytesIO(raw)) as image:
                        need(image.format=='PNG' and list(image.size)==x['source_dimensions'] and image.mode in ('RGB','RGBA'),
                             'Actual frozen full-source RGB geometry differs')
                        source_mode=image.mode;image.load();im=image.convert('RGB')
                    rawpixels=np.asarray(im).tobytes();need(hashlib.sha256(rawpixels).hexdigest()==x['RGB_pixels_SHA256'],
                                                           'Original RGB pixel identity differs')
                    score=float(native([im],tuple(plan['reward_target_rgb']))[0]);r['actual_original_reward_image_scores']+=1
                    mirrored=ImageOps.mirror(im)
                    fs=float(native([mirrored],tuple(plan['reward_target_rgb']))[0]);r['actual_flip_diagnostic_scores']+=1
                    need(np.isfinite([score,fs]).all(),'Real native reward nonfinite')
                    diff=abs(score-fs);scores.append(score);flipdiff.append(diff)
                    entry.update(passed=True,source=pin(raw),width=im.width,height=im.height,source_mode=source_mode,
                                 reward_input_mode='RGB',reward=score,
                                 flip_reward=fs,absolute_flip_difference=diff)
                    item['completed']+=1;r['completed_reference_images']+=1
                    groups[split].add(x['group']);events[split].update(x['event_ids']);pixels[split].add(x['RGB_pixels_SHA256'])
                    if split=='train' and len(first4)<4:first4.append(im);first4scores.append(score)
                except Exception as e:
                    entry.update(error_type=type(e).__name__,reason=str(e)[:256]);item['failed']+=1;r['failed_reference_images']+=1
            item['source_groups']=len(groups[split]);item['events']=len(events[split]);item['actual_dimensions']=sorted({(x.get('width'),x.get('height')) for x in item['rows'] if x['passed']})
            if scores:
                item['reward_statistics']={'count':len(scores),'mean':float(np.mean(scores)),'std_population':float(np.std(scores)),
                     'min':min(scores),'max':max(scores),'flip_max_absolute_difference':max(flipdiff),
                     'flip_equal_within_1e-7':sum(x<=1e-7 for x in flipdiff)}
        need(not groups['train']&groups['dev'] and not events['train']&events['dev'] and not pixels['train']&pixels['dev'],
             'Train/dev group, event or pixel collision')
        r['cross_split_group_event_pixel_overlap']=0
        need(len(first4)==4,'Four real references required for input ABI diagnostic')
        batch4=score_batch4(native,first4,plan['reward_target_rgb']);r['actual_reference_only_batch4_scores']=4
        need(np.array_equal(batch4,np.asarray(first4scores)),'Native list input consistency differs')
        rejected=0
        for count in (0,3,5):
            try:score_batch4(native,([first4[0]]*count),plan['reward_target_rgb'])
            except ValueError:rejected+=1
        need(rejected==3,'Partial/padded engineering batch cardinalities must reject')
        r['reference_batch4_ABI_diagnostic']={'passed':True,'actual_policy_rollouts':0,'accepted_real_reference_images':4,
                                            'wrong_cardinality_rejected':3,'not_policy_group_reward_variance':True}
        need(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'Unexpected Torch import')
        r.update(passed=r['completed_reference_images']==18 and r['failed_reference_images']==0,
                 target_rgb=plan['reward_target_rgb'],Torch_imported=False,blind_inputs_labels_media_read=False,
                 existing_SFT_CSV_or_resize_or_VAE_rerun=False,
                 state='REAL_NATIVE_CPU_REWARD_INPUT_PASS_DOMAIN_SEMANTIC_SCORE_NOT_QUALIFIED')
    except Exception as e:
        r.update(error_type=type(e).__name__,reason=str(e)[:512],state='FAILED_OR_PARTIAL_INPUT_AUDIT')
    r['missing_reference_images']=18-r['completed_reference_images']-r['failed_reference_images']
    r['wall_seconds']=time.monotonic()-started
    save('result.json',r)
    print(json.dumps({k:r[k] for k in ('passed','state','completed_reference_images','failed_reference_images','missing_reference_images','actual_policy_rollouts','actual_optimizer_updates','READY')}))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
