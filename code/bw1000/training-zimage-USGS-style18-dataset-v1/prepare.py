"""Build T07 CSVs from original PNGs; no model, Torch, gold or blind media."""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import time

G = Path(__file__).resolve().parents[3]
SOURCE = G/'local-only/training-USGS-multievent-render144-v2-original/dataset'
SOURCE_SHA = '4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc'
UPSTREAM = G/'code/bw1000/training-zimage-target-native-meta-CPU-v1/upstream/diffsynth/core/data'
PINS = {'unified_dataset.py': (4972, 'b6e0729fbd00deee8b0d03dd463df43ce7e653ecea3f75c18741d4cc03ac027f'),
        'operators.py': (7735, 'c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4')}
CAPTION = ('A scientific earthquake report infographic on a white background. '
           'Two stacked bar charts show six chronological event categories: '
           'blue reported magnitude numbers above green reported depth in kilometres. '
           'Axes, event labels, numeric annotations and a data-source footer are visible.')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(raw):
    return {'bytes': len(raw), 'SHA256': hashlib.sha256(raw).hexdigest()}


def put(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return pin(raw)


def native_parts(Image):
    """Use unmodified exact loader/routing methods; don't import DiffSynth/Torch."""
    trees = {}
    for name, (length, digest) in PINS.items():
        raw = (UPSTREAM/name).read_bytes()
        need(pin(raw) == {'bytes': length, 'SHA256': digest}, 'original T07 source differs')
        trees[name] = ast.parse(raw)
    names = {'DataProcessingPipeline', 'DataProcessingOperator', 'LoadImage', 'ToAbsolutePath'}
    body = [x for x in trees['operators.py'].body if isinstance(x, ast.ClassDef) and x.name in names]
    need({x.name for x in body} == names, 'native image-loader methods missing')
    original = next(x for x in trees['unified_dataset.py'].body if isinstance(x, ast.ClassDef) and x.name=='UnifiedDataset')
    methods = [x for x in original.body if isinstance(x, ast.FunctionDef) and x.name in ('__getitem__', '__len__')]
    need(len(methods)==2, 'native dataset routing methods missing')
    body.append(ast.ClassDef(name='NativeRouting', bases=[], keywords=[], body=methods, decorator_list=[]))
    crop = next(x for x in trees['operators.py'].body if isinstance(x, ast.ClassDef) and x.name=='ImageCropAndResize')
    dim = [x for x in crop.body if isinstance(x, ast.FunctionDef) and x.name=='get_height_width']
    body.append(ast.ClassDef(name='NativeDimensions', bases=[], keywords=[], body=dim, decorator_list=[]))
    ns = {'Image': Image, 'os': os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'pinned-T07-data-methods', 'exec'), ns)
    return ns


def recipe_overlay(target_data, train_manifest_sha):
    root = PurePosixPath(target_data)
    need(root.is_absolute() and '..' not in root.parts and len(train_manifest_sha)==64, 'exact Linux data root required')
    # Only new dataset fields; all native model/fusion/optimizer arguments stay external.
    return {'dataset_root': str(root/'train'), 'dataset_metadata': str(root/'train/metadata.csv'),
            'dataset_manifest': str(root/'train/manifest.json'), 'dataset_sha': train_manifest_sha}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--gate',required=True)
    p.add_argument('--deadline',type=float,required=True); p.add_argument('--target-data',required=True)
    a=p.parse_args()
    while not Path(a.gate).exists():
        need(time.time()<a.deadline, 'deadline before Job admission'); time.sleep(.01)
    gate=json.loads(Path(a.gate).read_bytes())
    need(gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648, 'actual Job limits differ')
    from PIL import Image
    import importlib.metadata as md
    need(md.version('Pillow')=='13.0.0', 'pinned local Pillow differs')
    out=Path(a.output); out.mkdir(exist_ok=False)
    r={'schema':'ZIMAGE_USGS_STYLE18_CSV_LOCAL_V1','passed':False,'planned_images':18,'completed_images':0,
       'split_planned':{'train':12,'dev':6},'split_completed':{'train':0,'dev':0},
       'blind_media_read':False,'QA_or_gold_read':False,'HTTP':0,'GPU':0,'optimizer_updates':0,
       'native_Torchvision_resize_executed':False,'native_tokenizer_or_VAE_executed':False,
       'target_READY':False,'training_GO':False,'GPU_GO':False,'original_T07_failures_preserved':True}
    try:
        raw=(SOURCE/'manifest-private.json').read_bytes()
        need(pin(raw)=={'bytes':14864,'SHA256':SOURCE_SHA}, 'frozen source manifest differs')
        source=json.loads(raw)
        need(source['Software']=='training-USGS-multievent-render144-v2', 'source renderer differs')
        groups={s:set() for s in ('train','dev','blind')}; events={s:set() for s in groups}
        for c in source['charts']:
            groups[c['split']].add(c['group']); events[c['split']].update(c['event_ids'])
        for x,y in (('train','dev'),('train','blind'),('dev','blind')):
            need(not groups[x]&groups[y] and not events[x]&events[y], 'source groups or events overlap')
        selected=[c for c in source['charts'] if c['split'] in ('train','dev')]
        need(len(selected)==18, 'planned media denominator differs')
        ns=native_parts(Image)
        body_seen=set(); pixels_seen=set(); image_records=[]; csv_pins={}; manifests={}
        for split, count in r['split_planned'].items():
            root=out/split; (root/'images').mkdir(parents=True)
            rows=[]; bindings=[]
            for c in [v for v in selected if v['split']==split]:
                need(time.time()<a.deadline-3, 'original epoch exhausted')
                rel=PurePosixPath(c['image'])
                need(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]==split, 'source path escape')
                src=SOURCE/str(rel); need(not src.is_symlink(), 'source image alias')
                original=src.read_bytes(); need(pin(original)==source['outputs'][c['image']], 'original image bytes differ')
                need(pin(original)['SHA256'] not in body_seen,'duplicate image body'); body_seen.add(pin(original)['SHA256'])
                dest=root/'images'/rel.name
                with dest.open('xb') as f: f.write(original); f.flush(); os.fsync(f.fileno())
                # Exact native ToAbsolutePath + LoadImage (RGB conversion) on these real files.
                path=ns['ToAbsolutePath'](str(root))('images/'+rel.name)
                rgb=ns['LoadImage']()(path)
                need(rgb.mode=='RGB' and rgb.size==(512,512),'native decoded RGB geometry differs')
                need(rgb.info.get('Software')==source['Software'],'image software identity differs')
                pixels=rgb.tobytes(); need(min(pixels)<max(pixels),'blank image')
                pixel_sha=pin(pixels)['SHA256']; need(pixel_sha not in pixels_seen,'duplicate RGB pixels'); pixels_seen.add(pixel_sha)
                dims=ns['NativeDimensions'](); dims.height=dims.width=256
                dims.max_pixels=1920*1080; dims.height_division_factor=dims.width_division_factor=16
                need(dims.get_height_width(rgb)==(256,256),'native target dimensions differ')
                rows.append({'image':'images/'+rel.name,'prompt':CAPTION})
                bindings.append({'image':'images/'+rel.name,**pin(original)})
                image_records.append({'split':split,'source_relative':c['image'],'source':pin(original),
                    'dataset_relative':split+'/images/'+rel.name,'RGB_pixels_SHA256':pixel_sha,
                    'source_dimensions':[512,512],'native_target_dimensions':[256,256],
                    'group':c['group'],'event_ids':c['event_ids']})
                r['completed_images']+=1; r['split_completed'][split]+=1
                rgb.close()
            need(len(rows)==count,'incomplete split')
            stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=['image','prompt'],lineterminator='\n')
            writer.writeheader(); writer.writerows(rows); encoded=stream.getvalue().encode('utf-8')
            with (root/'metadata.csv').open('xb') as f: f.write(encoded); f.flush(); os.fsync(f.fileno())
            csv_pins[split]=pin(encoded)
            with (root/'metadata.csv').open(encoding='utf-8',newline='') as f:
                reader=csv.DictReader(f); need(reader.fieldnames==['image','prompt'],'CSV fields differ'); parsed=list(reader)
            need(parsed==rows,'CSV roundtrip differs')
            routing=ns['NativeRouting'](); routing.data=parsed; routing.repeat=1; routing.max_data_items=None
            routing.load_from_cache=False; routing.data_file_keys=('image',); routing.special_operator_map={}
            routing.main_data_operator=lambda value,base=root:ns['LoadImage']()(ns['ToAbsolutePath'](str(base))(value))
            need(len(routing)==count,'native row count differs')
            for i in range(count):
                value=routing[i]; need(value['prompt']==CAPTION and value['image'].size==(512,512),'native CSV row or image differs')
                value['image'].close()
            manifests[split]=put(root/'manifest.json',{'schema':'ZIMAGE_NEW_STYLE_DATASET_V1','rows':count,
                'metadata':csv_pins[split],'images':bindings,'source_manifest_SHA256':SOURCE_SHA,
                'prompt_origin':'locally authored visual-style caption; not USGS/COCO labels',
                'native_output_dimensions':[256,256],'native_resize_pending':True})
        overlay=recipe_overlay(a.target_data,manifests['train']['SHA256'])
        put(out/'training-data-overlay.json',overlay)
        binding=put(out/'data-binding-private.json',{'images':image_records,'metadata':csv_pins,'manifests':manifests,
            'source_manifest_SHA256':SOURCE_SHA,'target_data':a.target_data,'actual_target_files_exist':False,
            'train_dev_blind_groups':{k:len(v) for k,v in groups.items()},'train_dev_blind_events':{k:len(v) for k,v in events.items()},
            'caption_SHA256':pin(CAPTION.encode())['SHA256'],'blind_body_read':False,'QA_or_gold_read':False})
        r.update(passed=True,state='T07_NEW_CSV_AND_ORIGINAL_PNG_LOADING_PASS',unique_source_images=len(body_seen),
            unique_RGB_pixel_hashes=len(pixels_seen),source_manifest_SHA256=SOURCE_SHA,
            actual_CSV_rows=18,native_LoadImage_calls=36,native_dataset_getitem_calls=18,
            native_target_dimensions_calls=18,source_images_unchanged=True,
            data_binding=binding,metadata=csv_pins,train_manifest=manifests['train'],
            limitations=['512px original PNG preserved; target native RGB/BILINEAR/center_crop256 not executed',
                'one visual-style caption; not numerical chart-answer training or effect acceptance',
                'not original COCO20 recipe; new recipe has separate performance/effect references',
                'Turbo adapter fusion, target model/provider/device/optimizer and recovery remain unqualified'])
    except Exception as e:
        r.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,reason=str(e)[:256])
    put(out/'result.json',r)
    print(json.dumps({k:r[k] for k in ('passed','state','planned_images','completed_images','split_completed','target_READY')},sort_keys=True))
    return 0 if r['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
