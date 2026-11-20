"""Local derived-image gate only. No network, Torch, QA labels or blind bodies."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

D = Path(__file__).resolve().parent
G = D.parents[2]
SOURCE = G / 'local-only/training-USGS-multievent-render144-v2-original/dataset'
MANIFEST_SHA = '4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc'
CAPTION = ('A scientific earthquake report infographic on a white background. '
           'Two stacked bar charts show six chronological event categories: '
           'blue reported magnitude numbers above green reported depth in kilometres. '
           'Axes, event labels, numeric annotations and a data-source footer are visible.')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(raw):
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def write(path, value):
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--gate', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    gate = Path(args.gate)
    while not gate.exists():
        need(time.time() < args.deadline, 'deadline before gate')
        time.sleep(.01)
    admission = json.loads(gate.read_bytes())
    need(admission['CPU_count'] == 2 and admission['memory_limit_bytes'] == 2147483648, 'Job limits differ')
    import importlib.metadata as md
    from PIL import Image
    need(md.version('Pillow') == '13.0.0', 'qualified Pillow differs')
    need(time.time() < args.deadline, 'deadline before source')
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    result = dict(schema='SDXL_USGS_STYLE18_LOCAL_V1', passed=False, planned_images=18,
                  completed_images=0, planned_split_images={'train': 12, 'dev': 6},
                  actual_split_images={'train': 0, 'dev': 0}, blind_body_read=False,
                  QA_or_gold_read=False, HTTP=0, native=0, model=0, optimizer_updates=0,
                  Dataset_GO=False, training_GO=False, GPU_GO=False, target_READY=False)
    records = []
    try:
        raw = (SOURCE / 'manifest-private.json').read_bytes()
        need(len(raw) == 14864 and pin(raw)['SHA256'] == MANIFEST_SHA, 'source manifest differs')
        manifest = json.loads(raw)
        need(manifest['Software'] == 'training-USGS-multievent-render144-v2', 'source software differs')
        groups = {s: set() for s in ('train', 'dev', 'blind')}
        events = {s: set() for s in groups}
        for chart in manifest['charts']:
            s = chart['split']
            need(s in groups, 'unknown split')
            groups[s].add(chart['group'])
            events[s].update(chart['event_ids'])
        for a, b in [('train', 'dev'), ('train', 'blind'), ('dev', 'blind')]:
            need(not groups[a] & groups[b] and not events[a] & events[b], 'source split overlaps')
        selected = [c for c in manifest['charts'] if c['split'] in ('train', 'dev')]
        need(len(selected) == 18, 'full image denominator differs')
        source_seen, pixel_seen = set(), set()
        for split in ('train', 'dev'):
            (output / split).mkdir()
        metadata = {'train': [], 'dev': []}
        for chart in selected:
            need(time.time() < args.deadline - 3, 'source processing deadline')
            relative = Path(chart['image'])
            need(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == chart['split'], 'image escape')
            source = SOURCE / relative
            need(source.is_file() and not source.is_symlink(), 'source not plain image')
            raw = source.read_bytes()
            need(pin(raw) == manifest['outputs'][chart['image']], 'source image differs')
            need(pin(raw)['SHA256'] not in source_seen, 'duplicate source body')
            source_seen.add(pin(raw)['SHA256'])
            with Image.open(io.BytesIO(raw)) as image:
                need(image.format == 'PNG' and image.size == (512, 512), 'source geometry differs')
                need(image.info.get('Software') == 'training-USGS-multievent-render144-v2', 'source PNG metadata differs')
                image.load()  # Full actual PNG decoding, not passive-header qualification.
                rgb = image.convert('RGB')
                small = rgb.resize((256, 256), resample=Image.Resampling.BICUBIC)
                pixels = small.tobytes()
                need(len(pixels) == 196608 and min(pixels) < max(pixels), 'blank or wrong RGB image')
                pixel_sha = pin(pixels)['SHA256']
                need(pixel_sha not in pixel_seen, 'duplicate transformed pixels')
                pixel_seen.add(pixel_sha)
                name = relative.name
                destination = output / chart['split'] / name
                small.save(destination, format='PNG', optimize=False, compress_level=6)
            transformed = destination.read_bytes()
            with Image.open(io.BytesIO(transformed)) as check:
                check.load()
                need(check.mode == 'RGB' and check.size == (256, 256) and check.tobytes() == pixels, 'saved RGB differs')
            metadata[chart['split']].append({'file_name': name, 'text': CAPTION})
            records.append(dict(source_relative=chart['image'], source=pin(raw), transformed_relative=chart['split']+'/'+name,
                                transformed=pin(transformed), RGB_pixels_SHA256=pixel_sha, width=256, height=256,
                                group=chart['group'], event_ids=chart['event_ids'], caption_SHA256=pin(CAPTION.encode())['SHA256'],
                                normalized_pixel_min=2*min(pixels)/255-1, normalized_pixel_max=2*max(pixels)/255-1))
            result['completed_images'] += 1
            result['actual_split_images'][chart['split']] += 1
            write(output / ('progress-%02d.json' % result['completed_images']), dict(completed=result['completed_images'], planned=18))
        pins = {}
        for split, rows in metadata.items():
            raw = b''.join(json.dumps(r, ensure_ascii=True, sort_keys=True).encode()+b'\n' for r in rows)
            (output / split / 'metadata.jsonl').write_bytes(raw)
            pins[split+'/metadata.jsonl'] = pin(raw)
        need(result['actual_split_images'] == result['planned_split_images'], 'split denominator incomplete')
        write(output / 'media-binding-private.json', dict(source_manifest_SHA256=MANIFEST_SHA, images=records, metadata=pins,
              caption_origin='locally authored generic visual-style description; not a USGS or COCO annotation',
              QA_answers_or_numeric_generation_targets=False))
        result.update(passed=True, state='LOCAL_STYLE_MEDIA_DECODE_AND_IMAGEFOLDER_PREPARATION_PASS',
                      unique_decoded_source_images=len(source_seen), unique_transformed_pixels=len(pixel_seen),
                      source_groups={s: len(v) for s,v in groups.items()}, source_events={s:len(v) for s,v in events.items()},
                      source_manifest_SHA256=MANIFEST_SHA, pillow=md.version('Pillow'),
                      transformed_bytes=sum(r['transformed']['bytes'] for r in records),
                      limitations=['256px text readability unqualified', 'one shared caption limits semantic diversity',
                                   'project SDXL canonical backend imagefolder binding not yet qualified',
                                   'existing Hygon SDXL numerical-update FAIL retained', 'no target deployment or optimizer update'])
    except Exception as error:
        result.update(state='FAILED_PARTIAL_PRESERVED', error_type=type(error).__name__, error=str(error)[:256])
    write(output / 'result.json', result)
    print(json.dumps({k:result[k] for k in ('passed','state','planned_images','completed_images','actual_split_images','blind_body_read','QA_or_gold_read','target_READY')}, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
