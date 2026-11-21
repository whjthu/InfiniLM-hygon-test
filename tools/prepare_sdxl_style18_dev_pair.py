"""Prepare paired generation inputs from the frozen SDXL development split."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


MEDIA_SHA = '710323b657ded1817a6f3ca01e5ae15f9e55be34c6915952bd9de42f53385659'
ARMS = ('base_before', 'domain_adapter_after')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(root, relative, pin):
    path = root / relative
    require(path.resolve().parent == root.resolve() / 'dev', 'development file escaped its split')
    raw = path.read_bytes()
    require(pin == {'bytes': len(raw), 'SHA256': sha(raw)}, 'development file differs from its pin')
    return raw


def plan(root):
    media_raw = (root / 'media-binding-private.json').read_bytes()
    require(sha(media_raw) == MEDIA_SHA, 'frozen media binding mismatch')
    binding = json.loads(media_raw)
    train, dev = {}, {}
    for row in binding['images']:
        parts = PurePosixPath(row['transformed_relative']).parts
        require(len(parts) == 2 and parts[0] in ('train', 'dev'), 'unexpected split or file path')
        split = train if parts[0] == 'train' else dev
        require(parts[1] not in split, 'duplicate media entry')
        split[parts[1]] = row
    require(len(train) == 12 and len(dev) == 6, 'unexpected split counts')
    for key in ('group', 'RGB_pixels_SHA256'):
        require({row[key] for row in train.values()}.isdisjoint(row[key] for row in dev.values()),
                'training and development overlap')
    train_events = {event for row in train.values() for event in row['event_ids']}
    require(train_events.isdisjoint(event for row in dev.values() for event in row['event_ids']),
            'training and development event overlap')

    metadata = read_pinned(root, 'dev/metadata.jsonl', binding['metadata']['dev/metadata.jsonl'])
    prompts = [json.loads(line) for line in metadata.splitlines()]
    require(len(prompts) == len(dev) and {row['file_name'] for row in prompts} == set(dev),
            'development prompts do not cover the media view')
    samples = []
    for index, row in enumerate(prompts):
        name, prompt = row['file_name'], row['text']
        image = dev[name]
        prompt_sha = sha(prompt.encode('utf-8'))
        require(prompt_sha == image['caption_SHA256'], 'development prompt mismatch')
        read_pinned(root, 'dev/' + name, image['transformed'])
        for replicate in range(2):
            identity = sha(f'SDXL_STYLE18_DEV_PAIR_V1/{MEDIA_SHA}/{index}/{replicate}'.encode('ascii'))
            samples.append(dict(sample_id=identity, seed=int(identity[:15], 16), prompt=prompt,
                                prompt_SHA256=prompt_sha, development_reference=image['transformed']))
    require(len({row['seed'] for row in samples}) == len(samples), 'generation seeds repeat')
    return dict(schema='SDXL_STYLE18_DEV_PAIR_PLAN_V1', arms=ARMS, media_SHA256=MEDIA_SHA,
                samples=samples, width=256, height=256, images_as_generation_input=False,
                paired_initial_latents_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('private_plan', type=Path)
    args = parser.parse_args()
    result = plan(args.data)
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with args.private_plan.open('xb') as file:
        file.write(raw)
    result_samples = result['samples']
    print(json.dumps(dict(schema=result['schema'], plan_SHA256=sha(raw), plan_bytes=len(raw),
                          verified_development_images=6, development_prompt_variants=len({
                              row['prompt_SHA256'] for row in result_samples}),
                          planned_pairs=len(result_samples), distinct_seeds=len(result_samples),
                          planned_generations=len(ARMS) * len(result_samples),
                          actual_generations=0, actual_optimizer_updates=0, ready_training_added=0,
                          blind_files_read=0, train_media_files_read=0, metric_status='NOT_RUN')))


if __name__ == '__main__':
    main()
