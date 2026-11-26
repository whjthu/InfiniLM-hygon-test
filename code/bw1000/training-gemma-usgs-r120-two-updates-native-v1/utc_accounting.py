"""Read exact numbered Slurm terminals in UTC and compare lossless nine-column evidence."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys


SCHEMA = 'SLURM_SACCT_EXACT_NUMBERED_TERMINALS_UTC_V1'
TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
COLUMNS = ('JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES','Start','End')
FORMAT = ','.join('JobName%120' if column=='JobName' else column for column in COLUMNS)
HANDLE = re.compile(r'[1-9][0-9]{0,9}\.[0-9]{1,10}')
STAMP = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
STATE = re.compile(r'(?:COMPLETED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|CANCELLED(?: by [0-9]+)?)')
CAP = 32768


def require(condition, code):
    if not condition:
        raise ValueError(code)


def handles(steps):
    values = tuple(steps)
    require(1<=len(values)<=16 and all(isinstance(v,str) and HANDLE.fullmatch(v) for v in values)
            and len(values)==len(set(values)), 'EXACT_UNIQUE_NUMBERED_STEPS_REQUIRED')
    return values


def query_argv(steps):
    return ['env','TZ=UTC','LC_ALL=C','SLURM_TIME_FORMAT='+TIME_FORMAT,'sacct','-nP',
            '-j',','.join(handles(steps)),'--format',FORMAT]


def utc_epoch(value):
    require(isinstance(value,str) and STAMP.fullmatch(value), 'EXACT_UTC_TIMESTAMP_REQUIRED')
    try:
        return int(datetime.strptime(value,TIME_FORMAT).replace(tzinfo=timezone.utc).timestamp())
    except ValueError as error:
        raise ValueError('INVALID_UTC_CALENDAR_TIMESTAMP') from error


def exact_rows(rows, steps):
    steps = handles(steps)
    require(isinstance(rows,list) and len(rows)==len(steps), 'EXACT_ACCOUNTING_ROW_COUNT')
    by_step={}
    for row in rows:
        require(isinstance(row,list) and len(row)==len(COLUMNS) and all(isinstance(v,str) for v in row),
                'EXACT_NINE_COLUMNS_REQUIRED')
        step,name,state,code,elapsed,nodes,tres,start,end=row
        require(step in steps and step not in by_step, 'MISSING_EXTRA_OR_DUPLICATE_STEP')
        require(0<len(name)<=120 and all(32<=ord(c)<127 for c in name) and not name.endswith('+')
                and STATE.fullmatch(state) and re.fullmatch(r'[0-9]{1,10}:[0-9]{1,10}',code)
                and re.fullmatch(r'[0-9]{1,10}',elapsed)
                and re.fullmatch(r'[A-Za-z0-9_.,\[\]-]{1,256}',nodes) and nodes not in ('Unknown','None','N/A')
                and re.fullmatch(r'[A-Za-z0-9_/.:=,\-]{1,512}',tres), 'NONTERMINAL_OR_TRUNCATED_ACCOUNTING')
        beginning,finished=map(utc_epoch,(start,end))
        require(0<beginning<=finished and abs(finished-beginning-int(elapsed))<=1, 'ELAPSED_TIME_INCONSISTENT')
        by_step[step]=row
    require(by_step.keys()==set(steps),'EXACT_ACCOUNTING_STEP_SET')
    return [by_step[step] for step in steps]


def collect_exact(steps, run=subprocess.run):
    """Production and independent readers should both use this same TZ=UTC command."""
    steps = handles(steps)
    collected = run(query_argv(steps), stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
    require(collected.returncode==0 and 0<len(collected.stdout)<=CAP and not collected.stderr,
            'BOUNDED_UNAMBIGUOUS_SACCT_QUERY_REQUIRED')
    rows = [line.split('|') for line in collected.stdout.decode('ascii').splitlines()]
    return evidence(exact_rows(rows,steps),steps)


def canonical_rows_digest(rows):
    raw = json.dumps(rows,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode('ascii')
    return sha256(raw).hexdigest()


def evidence(rows, steps):
    steps = handles(steps)
    values = exact_rows(rows,steps)
    return dict(schema=SCHEMA,TZ='UTC',SLURM_TIME_FORMAT=TIME_FORMAT,columns=list(COLUMNS),
                requested_steps=list(steps),rows=values,canonical_rows_SHA256=canonical_rows_digest(values))


def validate(document):
    require(isinstance(document,dict) and set(document)=={'schema','TZ','SLURM_TIME_FORMAT','columns',
            'requested_steps','rows','canonical_rows_SHA256'} and document['schema']==SCHEMA
            and document['TZ']=='UTC' and document['SLURM_TIME_FORMAT']==TIME_FORMAT
            and document['columns']==list(COLUMNS) and isinstance(document['requested_steps'],list),
            'EXPLICIT_UTC_NINE_COLUMN_EVIDENCE_REQUIRED')
    require(document==evidence(document['rows'],document['requested_steps']), 'EXACT_CANONICAL_ROWS_HASH_OR_ORDER_MISMATCH')
    return document


def compare(expected, observed):
    old, current=validate(expected),validate(observed)
    require(set(old['requested_steps'])==set(current['requested_steps']), 'ACCOUNTING_STEP_SET_DIFFERS')
    actual={row[0]:row for row in current['rows']}
    differences = sorted({column for row in old['rows'] for i,column in enumerate(COLUMNS) if row[i]!=actual[row[0]][i]})
    # UTC provenance is required at both producers. There is no offset correction.
    return dict(passed=not differences, different_columns=differences, compared_steps=len(actual),
                timestamp_offset_repair_applied=False, READY=False)


def read_pinned(path, expected):
    require(re.fullmatch(r'[0-9a-f]{64}',expected) and path.is_file() and not path.is_symlink(), 'FILE_SHA256_REQUIRED')
    with path.open('rb') as file:
        raw=file.read(CAP+1)
    require(0<len(raw)<=CAP and sha256(raw).hexdigest()==expected, 'EXACT_ENVELOPE_SHA256_MISMATCH')
    def unique(pairs):
        result=dict(pairs)
        require(len(result)==len(pairs),'DUPLICATE_ENVELOPE_KEY')
        return result
    return json.loads(raw,object_pairs_hook=unique)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    read=commands.add_parser('collect',help='issue one read-only sacct query for exact numbered steps')
    read.add_argument('step',nargs='+')
    check=commands.add_parser('compare',help='compare two locally pinned UTC envelopes without invoking Slurm')
    for kind in ('expected','observed'):
        check.add_argument('--'+kind,type=Path,required=True)
        check.add_argument('--'+kind+'-sha256',required=True)
    args=parser.parse_args()
    try:
        if args.mode=='collect':
            print(json.dumps(collect_exact(args.step),sort_keys=True))
            return 0
        result=compare(read_pinned(args.expected,args.expected_sha256),read_pinned(args.observed,args.observed_sha256))
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as error:
        code=str(error)
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,
                              error_code=code if type(error) is ValueError and re.fullmatch('[A-Z0-9_]+',code) else None)))
        return 1
    print(json.dumps(result,sort_keys=True))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    sys.exit(main())
