"""Reproduce the exact existing CSV and prove Gemma image mapping; read no images."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path

PINS = {
    'code/bw1000/training-zimage-USGS-style18-dataset-v1/prepare.py':
        'c8985dc708dcab5e7c15ffa9b5d738ee2294fb79adccdd082720b0db35ae168b',
    'code/bw1000/training-gemma-render72-target-data-stage-v1/INPUTS.json':
        '3bc8a516403ad5c17dd377efe037529a9e88370f9fdf606edd08795025cfec88',
    'coordination/training/zimage-style18-target-train-inputs-v1.json':
        'a06c29c8af3d56ed8192c39f63063db82bf67a14b75ba1fb2913202155d1ea09',
}
GEMMA_MEDIA = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-render72-native-CPU-v1/data/train/media'


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def prepare(repo,output):
    raw={}
    for name,sha in PINS.items():
        raw[name]=(repo/name).read_bytes()
        if hashlib.sha256(raw[name]).hexdigest()!=sha:raise ValueError('Published source metadata differs')
    original=ast.parse(raw[next(iter(PINS))])
    assignment=next(n for n in original.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CAPTION' for t in n.targets))
    caption=ast.literal_eval(assignment.value)
    gemma=json.loads(raw[list(PINS)[1]])['files']
    binding=json.loads(raw[list(PINS)[2]])
    rows=[];mapping=[]
    for i,row in enumerate(binding['rows']):
        relative=f'train/media/{i:03d}.png'
        matched=[p for p in gemma if p['target_relative']=='data/'+relative]
        if (len(matched)!=1 or row['ordinal']!=i or row['path']!=f'images/{i:03d}.png'
                or any(matched[0][k]!=row['image'][k] for k in ('bytes','SHA256'))
                or hashlib.sha256(caption.encode()).hexdigest()!=row['prompt_SHA256']):
            raise ValueError('Gemma/Z-Image train mapping differs')
        rows.append({'image':row['path'],'prompt':caption})
        mapping.append({'ordinal':i,'source':GEMMA_MEDIA+f'/{i:03d}.png',
                        'target_relative':row['path'],'image':row['image']})
    if len(mapping)!=12:raise ValueError('Expected12 train images')
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=['image','prompt'],lineterminator='\n')
    writer.writeheader();writer.writerows(rows);encoded=stream.getvalue().encode()
    if pin(encoded)!=binding['CSV']:raise ValueError('Reproduced frozen CSV bytes differ')
    output.mkdir(mode=0o700)
    with (output/'metadata.csv').open('xb') as f:f.write(encoded)
    result={'state':'EXACT_EXISTING_CSV_AND_PUBLISHED_GEMMA12_MAPPING_VERIFIED',
            'CSV':pin(encoded),'train_images':12,'mapping':mapping,
            'source_metadata_pins':{k:pin(v) for k,v in raw.items()},
            'image_bodies_read':0,'original_private_CSV_read':False,'new_source_samples':0,
            'current_target_image_hashes_observed':False,'development_or_blind_read':False,
            'native_calls':0,'optimizer_updates':0,'training_READY':False}
    with (output/'prepare-result.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=prepare(args.repo,args.output)
    print(json.dumps({k:r[k] for k in ('state','CSV','train_images','image_bodies_read','training_READY')},sort_keys=True))


if __name__=='__main__':main()
