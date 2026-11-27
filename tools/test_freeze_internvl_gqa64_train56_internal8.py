"""Fixtures and an optional train-only replay of the original .4369/.4370 evidence."""
import argparse
import copy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import unittest

import freeze_internvl_gqa64_train56_internal8 as split


REPO = Path(__file__).resolve().parents[1]
TARGET = PurePosixPath(split.ORIGINAL / 'data')


class SplitFixture(unittest.TestCase):
    def setUp(self):
        self.images = {}
        self.lineage = []
        self.rows = []
        self.media = {}
        for index in range(1, 65):
            name = f'{index}.jpg'
            raw = f'synthetic-image-{index}'.encode()
            self.images[name] = raw
            self.media[str(index)] = dict(path=name, source_group=f'flickr:{index}', license='fixture',
                                          source_url='fixture', SHA256=split.digest(raw))
            self.lineage.append(dict(self.media[str(index)], question_id=str(index), image_id=str(index)))
            self.rows.append(dict(images=[str(TARGET/name)], messages=[
                {'role':'user', 'content':f'<image>\nSynthetic question {index}?'}, {'role':'assistant', 'content':'fixture'}]))

    def records(self):
        return split.split_records(
            b''.join(map(split.encoded, self.rows)), self.lineage, self.media, target_data=TARGET,
            read_media=self.images.__getitem__, staged_images={item['path']:item['SHA256'] for item in self.lineage},
            identity=lambda raw: (int(raw.rsplit(b'-', 1)[1]), 1.05**int(raw.rsplit(b'-', 1)[1])))

    def test_two_proposed_updates_preserve_exact_denominators(self):
        records = self.records()
        train, internal, exposures, slots = split.exposure_plan(records)
        self.assertEqual((len(train), len(internal), len(exposures), len(slots)), (56, 8, 64, 64))
        self.assertEqual({x['ordinal'] for x in exposures}, {x['ordinal'] for x in train})
        self.assertEqual(len({slot['source_ordinal'] for slot in slots if slot['repeat']}),8)
        self.assertEqual({x['ordinal'] for x in internal},
                         {x['ordinal'] for x in split.exposure_plan(records[::-1])[1]})
        for step in (1,2):
            for rank in range(8):
                self.assertEqual([s['accumulation'] for s in slots if s['update']==step and s['rank']==rank],
                                 [1,2,3,4])

    def test_duplicate_group_and_question_rejected(self):
        for key in ('source_group', 'question_id'):
            with self.subTest(key=key):
                self.setUp()
                self.lineage[1][key] = self.lineage[0][key]
                if key == 'source_group':
                    self.media['2'][key] = self.lineage[0][key]
                with self.assertRaisesRegex(ValueError, 'DUPLICATE_'+key.upper()):
                    self.records()

    def test_current_media_and_cross_row_identity(self):
        self.images['1.jpg'] = b'changed'
        with self.assertRaisesRegex(ValueError, 'CURRENT_JPEG_SHA'):
            self.records()
        self.setUp()
        self.rows[0]['images'] = copy.deepcopy(self.rows[1]['images'])
        with self.assertRaisesRegex(ValueError, 'STAGED_ROW_IMAGE_DRIFT'):
            self.records()

    def test_near_duplicates_and_strict_json(self):
        self.assertTrue(split.near_duplicate((0x100,1.0),(0x104,1.019)))
        self.assertFalse(split.near_duplicate((0x100,1.0),(0x104,1.03)))
        with self.assertRaisesRegex(ValueError, 'KNOWN_NEAR_DUPLICATE'):
            split.split_records(b''.join(map(split.encoded, self.rows)), self.lineage, self.media,
                                target_data=TARGET, read_media=self.images.__getitem__, identity=lambda raw:(1,1))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON_KEY'):
            split.decode(b'{"images": [], "images": []}')


def actual_replay(output=None):
    existing = REPO/'local-only/internvl-gqa-stanford-train-compressed-member-v1'
    source = existing/'actual-gqa-relation-64-frozen-input-v2'
    evidence = {name:split.decode(split.read_regular(source/name, pin[0], wanted=pin))
                for name,pin in split.EVIDENCE_PINS.items()}
    original = split.read_regular(source/'train.jsonl', 22615, wanted=(
        22615,'0ccbe304ad1d7e686bcfcf8b81bf17ef43bab4c4fcaeef5410fc7010bf170752'))
    rows = [split.decode(line) for line in original.splitlines()]
    for row, entry in zip(rows,evidence['training-lineage.json']['lineage'],strict=True):
        assert PureWindowsPath(row['images'][0]).name == entry['path']
        row['images'] = [str(TARGET/entry['path'])]
    staged = ('\n'.join(json.dumps(row,sort_keys=True,ensure_ascii=False) for row in rows)+'\n').encode()
    assert (len(staged),split.digest(staged)) == split.TRAIN_PIN
    archived = split.read_regular(existing/'actual-gqa64-logit-probe-caac5aebc8e04fef9fd88f12ab69d952/result.json',
        53927,wanted=(53927,'9ebeb6d576844c590613f60ec62608550f1d0c96973977e45e41f09f58860a17'))
    fingerprint = json.dumps(split.decode(archived)['result']).encode()
    assert (len(fingerprint),split.digest(fingerprint)) == split.FINGERPRINT_PIN
    result, _ = split.assemble(staged,evidence,fingerprint,
        REPO/'code/bw1000/training-domain-pilot-v1/internvl_gqa_training_roster.py',
        lambda name:split.read_regular(source/'media'/name, 4*1024*1024),target_data=TARGET)
    result['local_original_to_target_row_binding_reconstructed_exactly'] = True
    result['actual_target_execution'] = False
    raw = split.encoded(result)
    if output is not None:
        with output.open('xb') as stream:
            stream.write(raw)
    print(raw.decode(),end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.actual:
        actual_replay(args.receipt)
    else:
        unittest.main(argv=[__file__])
