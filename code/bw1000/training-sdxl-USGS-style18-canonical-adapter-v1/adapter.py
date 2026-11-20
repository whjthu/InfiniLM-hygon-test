"""New domain config for the exact frozen common SDXL backend; never trains here."""
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
G = D.parents[2]
RUN = G/'local-only/training-sdxl-USGS-style18-local-v1-original'
PINS = {
    'audit.json': (965, '4ac64a359841ae76f317de8da555a2964fd61f9e13134fb09eb6106df68ef072'),
    'dataset/media-binding-private.json': (17506, '710323b657ded1817a6f3ca01e5ae15f9e55be34c6915952bd9de42f53385659'),
    'dataset/train/metadata.jsonl': (3816, 'c32929f7156ec343b49df969e17bfe177672b44f6491e7114a96f04f98300870'),
    'dataset/dev/metadata.jsonl': (1908, '6dac37f488250744f779e9f40484097664cac23d645629ae22c9563ea3b28119'),
}


def load_checked(relative):
    raw = (RUN/relative).read_bytes()
    expected_size, expected_sha = PINS[relative]
    if len(raw)!=expected_size or hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('frozen style18 data metadata differs')
    return raw


def build_config(train_data_dir=None):
    audit=json.loads(load_checked('audit.json'))
    binding=json.loads(load_checked('dataset/media-binding-private.json'))
    if audit['passed'] is not True or audit['split_images']!={'train':12,'dev':6}:
        raise ValueError('actual local data prerequisite is incomplete')
    for split,count in [('train',12),('dev',6)]:
        rows=[json.loads(line) for line in load_checked('dataset/'+split+'/metadata.jsonl').splitlines()]
        if len(rows)!=count or any(set(row)!={'file_name','text'} for row in rows):
            raise ValueError('exact ImageFolder metadata schema differs')
        for row in rows:
            if Path(row['file_name']).name!=row['file_name'] or not row['text']:
                raise ValueError('invalid file_name or text')
            records=[r for r in binding['images'] if r['transformed_relative']==split+'/'+row['file_name']]
            if len(records)!=1 or records[0]['caption_SHA256']!=hashlib.sha256(row['text'].encode()).hexdigest():
                raise ValueError('caption does not bind actual media')
    # Explicit new recipe, not an overwrite of the formal T06 original recipe.
    train = RUN/'dataset/train' if train_data_dir is None else Path(train_data_dir)
    if not train.is_absolute():
        raise ValueError('training directory must be an explicit absolute path')
    return dict(train_data_dir=str(train), max_train_steps=25, resolution=256,
                train_batch_size=1, learning_rate=1e-4, lora_rank=4,
                seed=20261121, dataloader_num_workers=2, gradient_checkpointing=False,
                checkpointing_steps=10, checkpoints_total_limit=3,
                extra_checkpoint_steps='20,25', warmup_steps=5)


def run_via_common_backend(backend, *, weights, output_dir, source_root, train_data_dir=None):
    """Root must supply a qualified real module/model/trainer/runtime before use."""
    filename=Path(backend.run_sdxl_training.__code__.co_filename)
    raw=filename.read_bytes()
    if hashlib.sha256(raw).hexdigest() not in {
        '1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7',
        '65814732b3d2eb4bec29198c2594bbfbd0e84ceb364592ac542a23c133c11ffb',
    }:
        raise ValueError('actual common SDXL backend provider differs')
    return backend.run_sdxl_training(Path(weights), build_config(train_data_dir),
                                     Path(output_dir), Path(source_root))
