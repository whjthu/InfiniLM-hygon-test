"""Execute real18 through the original transform and audited official PIL route."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from pil_branch import build,need

D=Path(__file__).resolve().parent
G=D.parents[2]
SOURCE=G/'local-only/training-zimage-style18-official-PIL-transform-v1-source-reviewed-original'
DATA=G/'local-only/training-zimage-USGS-style18-dataset-v1-original/dataset'
OUT=G/'local-only/training-zimage-style18-official-PIL-transform-v1-input-original'


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def read(path,expected):
    with path.open('rb') as f:
        before=os.fstat(f.fileno());b=f.read(expected['bytes']+1);after=os.fstat(f.fileno())
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(ident(before)==ident(after)==ident(path.stat()) and pin(b)==expected,'Frozen input bytes/identity differ')
    return b


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    start=time.monotonic();OUT.mkdir()
    save('intent.json',{'request_id':'6dac1ccbbd7747a7a227b15d2862f78a','wall_start':time.time(),
         'scripts':{n:pin((D/n).read_bytes()) for n in ['pil_branch.py','check_transform_once.py']},
         'planned_train':12,'planned_dev':6,'blind':0,'new_HTTP':0,'seconds_max_per_input_guard':45})
    r={'schema':'ZIMAGE_REAL18_OFFICIAL_PIL_BRANCH_512_TO256_LOCAL_V1','passed':False,'planned':18,'completed':0,'failed':0,
       'actual_native_PIL_resize':0,'actual_native_center_crop':0,'GPU_or_model_or_optimizer_calls':0,'READY':False,'splits':{}}
    try:
        import numpy as np
        from PIL import Image
        transform,trace,proof,invoke=build(SOURCE,G/'code/bw1000/training-zimage-target-native-meta-CPU-v1/upstream/diffsynth/core/data/operators.py')
        r['source_projection']=proof
        b=json.loads(read(DATA/'data-binding-private.json',{'bytes':14457,'SHA256':'2152981e81882985de57a9419a34f895440d7965cc3b277c762f15304bc51c35'}))
        sets={};all_refs=[]
        for split,count in [('train',12),('dev',6)]:
            entries=[x for x in b['images'] if x['split']==split];need(len(entries)==count,'Split denominator differs')
            (OUT/split).mkdir();stats={'planned':count,'completed':0,'failed':0,'rows':[]};r['splits'][split]=stats
            sets[split]={'groups':set(),'events':set(),'pixels':set()}
            for index,x in enumerate(entries):
                row={'ordinal':index,'passed':False,'source_path':x['dataset_relative']};stats['rows'].append(row)
                try:
                    need(time.monotonic()-start<45,'Local input audit budget elapsed')
                    relative=Path(x['dataset_relative']);need(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]==split,'Input escapes source split')
                    data=read(DATA/relative,x['source'])
                    with Image.open(io.BytesIO(data)) as im:
                        need(list(im.size)==[512,512] and im.format=='PNG','Frozen source geometry differs')
                        im.load();image=im.convert('RGB')
                    original_pixels=np.asarray(image).tobytes();need(hashlib.sha256(original_pixels).hexdigest()==x['RGB_pixels_SHA256'],'Original fullRGB pixels differ')
                    before=len(trace);output=transform(image)
                    need(trace[before:]==[{'name':'resize','input_size':[512,512],'output_size':[256,256]},
                                         {'name':'center_crop','input_size':[256,256],'output_size':[256,256]}],'Actual native routing differs')
                    need(output.size==(256,256) and output.mode=='RGB','Native transform output geometry differs')
                    pixels=np.asarray(output);need(pixels.shape==(256,256,3) and pixels.dtype==np.uint8,'FullRGB256 schema differs')
                    name=split+'/'+f'{index:03d}.png';output.save(OUT/name,format='PNG')
                    pixels_sha=hashlib.sha256(pixels.tobytes()).hexdigest()
                    row.update(passed=True,source=pin(data),reference_path=name,reference_png=pin((OUT/name).read_bytes()),
                               RGB256_pixels_SHA256=pixels_sha,source_width=512,source_height=512,width=256,height=256)
                    sets[split]['groups'].add(x['group']);sets[split]['events'].update(x['event_ids']);sets[split]['pixels'].add(pixels_sha)
                    stats['completed']+=1;r['completed']+=1;r['actual_native_PIL_resize']+=1;r['actual_native_center_crop']+=1
                    all_refs.append(row)
                except Exception as e:
                    row.update(error_type=type(e).__name__,reason=str(e)[:256]);stats['failed']+=1;r['failed']+=1
            stats.update(source_groups=len(sets[split]['groups']),source_events=len(sets[split]['events']))
        need(all(not sets['train'][key]&sets['dev'][key] for key in ['groups','events','pixels']),'Cross-split collision')
        rejected=False
        try:invoke('resize',np.zeros((2,2,3),dtype=np.uint8),[1,1])
        except ValueError:rejected=True
        need(rejected,'Non-PIL input must reject rather than impersonating a Tensor')
        need(not any(n=='torch' or n.startswith('torch.') or n=='torchvision' or n.startswith('torchvision.') for n in sys.modules),'Unexpected framework package import')
        r.update(passed=r['completed']==18 and not r['failed'],cross_split_group_event_output_pixels_overlap=0,
                 non_PIL_rejection_test=True,Torch_or_installed_TorchVision_imported=False,blind_input_or_gold_read=False,
                 source_resized_or_rewritten=False,reference_outputs_do_not_replace_original_SFT_source=True,
                 target_installed_torchvision_version_or_source_qualified=False,full_native_dataset_or_tokenizer_or_VAE_qualified=False,
                 state='REAL18_LOCAL_OFFICIAL_PIL_MATH_BRANCH_PASS_TARGET_PROVIDER_UNBOUND')
        save('RGB256-reference-binding-private.json',{'schema':'ZIMAGE_OFFICIAL_PIL_RGB256_REFERENCE_IDENTITIES_V1',
             'source_binding_SHA256':'2152981e81882985de57a9419a34f895440d7965cc3b277c762f15304bc51c35',
             'target_comparison_expected_pixel_hash_only':True,'do_not_stage_as_formal_or_original_inputs':True,
             'rows':all_refs,'target_GO':False,'training_GO':False})
    except Exception as e:r.update(error_type=type(e).__name__,reason=str(e)[:512],state='FAILED_OR_PARTIAL_INPUT_TRANSFORM')
    r.update(missing=18-r['completed']-r['failed'],wall_seconds=time.monotonic()-start)
    save('result.json',r)
    print(json.dumps({k:r[k] for k in ['passed','state','completed','failed','missing','actual_native_PIL_resize','actual_native_center_crop','READY']}))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
