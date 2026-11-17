"""New source-only USGS join boundary checks, no real labels/media/providers read."""
import copy
import json
from prepare_inputs import join


def fixture(split):
    rows = []; labels = []; media = {}
    for image in range(9 if split == 'train' else 2):
        name = split + '/media/' + str(image).zfill(3) + '.png'; media[name] = {}
        for q in range(5):
            key = split + ':' + str(image) + ':' + str(q); question = 'Fixture displayed value?'
            rows.append(dict(id=key, chart_group='c' + str(image), source_group='g' + str(image),
                             messages=[dict(role='user', content='<image>\n' + question)], images=[name]))
            labels.append(dict(id=key, chart_id='c' + str(image), group='g' + str(image),
                               question=question, data_status='QUALIFIED', answer_type='decimal2', answer='-1.23'))
    return rows, labels, media


def main():
    train, gold, media = fixture('train'); dev, _, dmedia = fixture('dev')
    target = '/home/whj/nvidia-baseline/alignment-assets/USGS-fixture/data'
    actual = join(train, gold, media, target, 'train')
    assert len(actual) == 45 and all(r['messages'][1]['content'] == 'FINAL: -1.23' for r in actual)
    actual_dev = join(dev, None, dmedia, target, 'dev')
    assert len(actual_dev) == 10 and all(len(r['messages']) == 1 for r in actual_dev)
    rejected = []
    def bad(name, call):
        try: call()
        except ValueError: rejected.append(name); return
        raise AssertionError(name)
    bad('partial train denominator', lambda: join(train[:-1], gold, media, target, 'train'))
    bad('dev gold', lambda: join(dev, [], dmedia, target, 'dev'))
    bad('blind split', lambda: join(train, gold, media, target, 'blind'))
    bad('bad target', lambda: join(train, gold, media, '/root/blind/data', 'train'))
    changed = copy.deepcopy(train); changed[0]['images'] = ['blind/media/000.png']
    bad('foreign image', lambda: join(changed, gold, media, target, 'train'))
    changed_gold = copy.deepcopy(gold); changed_gold[0]['question'] += ' modified'
    bad('misjoined question', lambda: join(train, changed_gold, media, target, 'train'))
    return dict(passed=True, normal=2, rejected=len(rejected), rejected_cases=rejected,
                real_domain_rows_read=False, real_gold_read=False, Torch_import=False,
                native_processor_run=False, GPU=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
