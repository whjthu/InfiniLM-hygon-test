"""Share the checked UTC terminal query between the release producer and keeper."""

import re

from utc_accounting import collect_exact, compare, query_argv, validate


def history_argv(parent):
    if not re.fullmatch(r'[1-9][0-9]{0,9}',parent):
        raise ValueError('EXACT_PARENT_JOB_REQUIRED')
    args=query_argv([parent+'.0'])
    args[args.index('-j')+1]=parent
    return args


def utc_control_argv(argv):
    if argv[0] in ('scontrol','squeue'):
        return [*history_argv('1')[:4],*argv]
    return argv


def keeper_accounting(proof, original, clock, collect=collect_exact):
    expected=validate(proof['UTC_accounting'])
    holder,post=proof['holder_accounting'],proof['post_accounting']
    steps=[*original,holder[0],post[0]]
    if expected['requested_steps']!=steps or expected['rows']!=[[step,*original[step]] for step in original]+[holder,post]:
        raise ValueError('R120_RELEASE_ROWS_NOT_EXACT_UTC_PROJECTION')

    clock.left(reserve=10)
    observed=collect(steps)
    if not compare(expected,observed)['passed']:
        raise ValueError('R120_EXACT_LIVE_UTC_ACCOUNTING_DIFFERS')
    return observed
