"""Consume the real T07 CSV constructor path without image/model/device imports."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

G=Path(__file__).resolve().parents[3]
DATA=G/'local-only/training-zimage-USGS-style18-dataset-v1-original/dataset'
OUT=G/'local-only/training-zimage-USGS-style18-pandas-constructor-v1-original'
NATIVE=G/'code/bw1000/training-zimage-target-native-meta-CPU-v1/upstream/diffsynth/core/data'
REQUEST='c5fa3bf486bb4b77b8d8c96fbead28d0'


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def need(ok,why):
    if not ok:raise ValueError(why)


def checked(path,wanted):
    need(not path.is_symlink(),'unexpected aliased metadata')
    with path.open('rb') as f:raw=f.read(wanted['bytes']+1)
    need(pin(raw)==wanted,'frozen metadata/source differs')
    return raw


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    need(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'unexpected Torch already imported')
    import pandas
    need(pandas.__version__=='3.0.1','actual local pandas version differs')
    native=checked(NATIVE/'unified_dataset.py',dict(bytes=4972,SHA256='b6e0729fbd00deee8b0d03dd463df43ce7e653ecea3f75c18741d4cc03ac027f'))
    operators=checked(NATIVE/'operators.py',dict(bytes=7735,SHA256='c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4'))
    trees=[ast.parse(native),ast.parse(operators)]
    original=next(n for n in trees[0].body if isinstance(n,ast.ClassDef) and n.name=='UnifiedDataset')
    selected=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','load_metadata','__getitem__','__len__')]
    need(len(selected)==4,'native constructor/loader/routing methods missing')
    parents=[n for n in trees[1].body if isinstance(n,ast.ClassDef) and n.name in ('DataProcessingOperator','LoadTorchPickle')]
    need(len(parents)==2,'native unused cache operator initializer missing')
    # Only the Tensor dataset base is omitted. All four function bodies and the
    # unused cache operator initializer stay exact; no cache path may be invoked.
    isolated=ast.ClassDef(name='NativeCSVConstructor',bases=[],keywords=[],body=selected,decorator_list=[])
    ns={'pandas':pandas,'json':json,'os':__import__('os')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=parents+[isolated],type_ignores=[])),'pinned-native-T07-CSV-constructor','exec'),ns)
    binding=checked(DATA/'data-binding-private.json',dict(bytes=14457,SHA256='2152981e81882985de57a9419a34f895440d7965cc3b277c762f15304bc51c35'))
    b=json.loads(binding);results={};prompt_hashes=set()
    for split,count in (('train',12),('dev',6)):
        need(time.monotonic()-start<30,'local read-only budget exhausted')
        path=DATA/split/'metadata.csv';raw=checked(path,b['metadata'][split])
        standard=list(csv.DictReader(io.StringIO(raw.decode('utf-8'),newline='')))
        # This calls original __init__ -> original pandas.read_csv/iloc/to_dict.
        # data_file_keys=() intentionally leaves image paths as metadata strings.
        native_csv=ns['NativeCSVConstructor'](base_path=str(DATA/split),metadata_path=str(path),repeat=1,data_file_keys=())
        need(not native_csv.load_from_cache and native_csv.cached_data==[],'unexpected cache path selected')
        need(len(native_csv)==count and native_csv.data==standard,'pandas source rows differ from exact CSV')
        expected=[Path(r['dataset_relative']).relative_to(split).as_posix() for r in b['images'] if r['split']==split]
        need([r['image'] for r in native_csv.data]==expected,'native metadata changes image identities/order')
        for ordinal in range(count):
            row=native_csv[ordinal]
            need(set(row)=={'image','prompt'} and all(type(v) is str and v for v in row.values()),'unexpected CSV dtype/missing text')
            need(row==standard[ordinal] and row is not native_csv.data[ordinal],'native item copy semantics differ')
            prompt_hashes.add(pin(row['prompt'].encode())['SHA256'])
        results[split]={'rows':count,'CSV':pin(raw),'native_metadata_rows_equal_standard_CSV':True,
            'all_image_and_prompt_fields_python_str':True,'cached_items_loaded':0,'image_operators_called':0}
    need(len(prompt_hashes)==1 and not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'caption identity or no-Torch boundary differs')
    r={'schema':'ZIMAGE_STYLE18_REAL_PANDAS_NATIVE_CSV_CONSTRUCTOR_LOCAL_V1','request_id':REQUEST,'passed':True,
       'split_results':results,'actual_native_constructor_calls':2,'actual_native_pandas_load_metadata_calls':2,
       'actual_metadata_rows':18,'actual_new_image_body_reads':0,'actual_new_native_image_encodes':0,
       'source_functions_AST_SHA256':pin(ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False).encode())['SHA256'],
       'pinned_source_files':{'unified_dataset.py':pin(native),'operators.py':pin(operators)},
       'actual_local_pandas':pandas.__version__,'target_pandas_version_qualification_inherited':False,
       'Torch_imported':False,'Torchvision_resize_executed':False,'model_or_VAE_loaded':False,
       'actual_optimizer_updates':0,'HTTP':0,'SSH':0,'Slurm':0,'GPU':0,
       'dev_gold_or_blind_body_read':False,'original_T07_and_frozen_style18_files_modified':False,
       'original_launch_bytes_present':False,'required_original_launch_SHA256':'f0bc54e478ad0e8b5d9e0b7fe632a172e5aa6d2e9d6fa166b6d09d280ecdc89d',
       'READY':False,'READY_increment':0,'training_GO':False,'GPU_GO':False,
       'wall_seconds':time.monotonic()-start,
       'limits':['Isolated original function execution with Tensor dataset base omitted; not full native package import',
                 'Local pandas3.0.1 proves current CSV dtype/quoting behavior only; target runtime remains separate',
                 'Original T07 launch wrapper bytes and native BILINEAR/center_crop256 still missing',
                 'Fusion/model/initial/provider/device and numerical/recovery/effect qualification unchanged']}
    (OUT/'result.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:r[k] for k in ('passed','actual_native_constructor_calls','actual_metadata_rows','actual_new_image_body_reads','READY','wall_seconds')},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
