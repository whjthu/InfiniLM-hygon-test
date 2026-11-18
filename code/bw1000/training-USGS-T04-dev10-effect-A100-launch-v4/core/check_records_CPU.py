"""Synthetic new raw-EOS/FINAL bridge boundaries; never encodes actual dev data."""
import importlib.util
import json
from pathlib import Path
import sys
from effect_contract import CAP, eos_observation, scoring_prediction, require_ready

D = Path(__file__).resolve().parent


def main():
    cases = []
    def check(name, condition):
        cases.append(dict(name=name, passed=bool(condition)))
    check('raw_pad_equal_EOS_final_kept', eos_observation([31, 2], [2])['natural_EOS'] is True)
    check('stop_without_raw_EOS_rejected', eos_observation([31, 32], [2])['natural_EOS'] is False)
    check('cap_even_with_final_EOS_rejected', eos_observation([31] * (CAP-1) + [2], [2]) == dict(
        natural_EOS=False, cap_hit=True, generated_tokens=CAP,
        generated_ids_SHA256=eos_observation([31] * (CAP-1) + [2], [2])['generated_ids_SHA256'], last_token_is_EOS=True))
    check('padding_after_EOS_not_natural', eos_observation([31, 2, 2], [2])['natural_EOS'] is False)
    for name, value in [('bare', '1.25'), ('FINAL', 'FINAL: 1.25'), ('ties', 'FINAL: E00 E02')]:
        check(name + '_canonicalized', scoring_prediction(dict(text=value))['text'] in ['1.25', 'E00 E02'])
    for name, value in [('repeated', 'FINAL: FINAL: 1.25'), ('explanation', 'The answer is 1.25'),
                        ('multiline', 'FINAL: 1.25\n'), ('extra', 'FINAL: 1.25 extra')]:
        check(name + '_rejected', scoring_prediction(dict(text=value))['text'] == 'INVALID_FINAL_FORMAT')
    check('strictFINAL_separately_counted', scoring_prediction(dict(text='FINAL: 1.25'))['strict_FINAL_format'] is True
          and scoring_prediction(dict(text='1.25'))['strict_FINAL_format'] is False)
    try:
        eos_observation([31, True], [2]); rejected = False
    except ValueError:
        rejected = True
    check('typed_token_bool_rejected', rejected)
    try:
        require_ready(dict(root_GO=True, GPU_GO=True, generic25=None)); rejected = False
    except ValueError as e:
        rejected = str(e) == 'GENERIC25_NOT_READY_NO_ACTUAL_ARTIFACT'
    check('unknown_generic25_before_model_import', rejected)
    sys.path.insert(0, str(D / 'frozen_scorer'))
    spec = importlib.util.spec_from_file_location('frozen_scorer_fixture', D / 'frozen_scorer/score_once.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    labels = [dict(id='synthetic-%d' % i, answer='1.25', data_status='QUALIFIED', answer_type='decimal2',
                   kind='lookup', chart_id='synthetic') for i in range(10)]
    prediction = scoring_prediction(dict(id='synthetic-0', status='COMPLETED', text='FINAL: 1.25', natural_EOS=True, cap_hit=False))
    result = module.score(labels, [prediction])
    check('published_scorer_bridge_full10_missing9', result['denominator'] == 10 and result['correct'] == 1 and result['failures'] == 9)
    print(json.dumps(dict(passed=all(x['passed'] for x in cases), cases=cases, actual_dev_read=False,
        actual_encode=0, actual_model=0, GPU=0, synthetic_only=True), sort_keys=True))
    return 0 if all(x['passed'] for x in cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
