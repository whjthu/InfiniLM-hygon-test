"""Bind the original MiniCPM 88-input CPU audit to fresh Slurm accounting."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
from types import SimpleNamespace

from read_training_slurm_membership import allocated_dcu


REPO = Path(__file__).resolve().parents[1]
LABEL = 'training-minicpm-usgs-native88-CPU-v1'
RUN = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results') / LABEL
PACKAGE = REPO / 'code/bw1000' / LABEL
PARENT, NODE = '1103606', 'f13r1n19'
REQUEST = '7f9a0e2809bf4f53a7294fdc3dc74225'
MANIFEST = 'c9c3e4c1b8cd44c069e83a9accff75acfcea22defe31148cef52584772d7aef9'
CLAIM = '03939c48bfc7f5bafbaedf1cbe20be3c7944cd1c47f28f08f0a6b0ab99baec5d'
READER = '5dd62673bc3e430b18954759a73be2c06376371b7121a30fa4003e6bd517c1ea'
COLUMNS = ['JobIDRaw','JobName%100','State','ExitCode','ElapsedRaw','NodeList','AllocTRES']
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED',
            'BOOT_FAIL','DEADLINE','REVOKED'}


class ReadDeadline(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_raw(path, limit=2*1024**2, pin=None):
    need(path.is_file() and not path.is_symlink(), 'missing or linked original file: '+path.name)
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and (pin is None or sha256(raw).hexdigest() == pin),
         'original metadata exceeds limit or SHA-256 differs: '+path.name)
    return raw


def accounting(raw):
    need(0 < len(raw) <= 2*1024**2, 'original accounting exceeds limit or is empty')
    rows = [line.split('|') for line in raw.decode().splitlines() if line.strip()]
    need(all(len(row) == len(COLUMNS) for row in rows), 'original accounting column layout differs')
    need(sum(row[0] == PARENT for row in rows) == 1, 'original accounting parent is missing')
    matched = [row for row in rows if row[1] == LABEL]
    need(len(matched) <= 1, 'original CPU88 label appears more than once')
    if not matched:
        return None
    row = matched[0]
    need(re.fullmatch(PARENT+r'\.\d+', row[0]) and row[5] == NODE and row[2].split(),
         'original CPU88 Slurm step or location differs')
    return row


def source_reader(package=PACKAGE):
    source = json.loads(read_raw(package/'SOURCE_FILES.json',32768,MANIFEST))
    member = source['files']['read_terminal.py']
    need(member['SHA256'] == READER and source['files']['CLAIM.json']['SHA256'] == CLAIM,
         'original metadata reader or claim pin changed')
    file = package/'read_terminal.py'
    raw = read_raw(file,32768,READER)
    need(len(raw) == member['bytes'], 'original metadata reader size differs')
    namespace = {'__name__':'frozen_minicpm_native88_metadata_reader'}
    exec(compile(raw,str(file),'exec'),namespace)
    return namespace['read']


def verify(root, raw, publication, epoch, *, load_reader=source_reader):
    need(re.fullmatch(r'[0-9a-f]{40}', publication) and math.isfinite(epoch) and epoch > 0,
         'required original publication or epoch differs')
    row = accounting(raw)
    terminal = bool(row and row[2].split()[0] in TERMINAL)
    bound = dict(columns=COLUMNS,row=row,terminal=terminal,raw_SHA256=sha256(raw).hexdigest())
    result = dict(state='ORIGINAL_ACCOUNTING_UNKNOWN',passed=False,request=REQUEST,publication=publication,
                  planned_rows=88,actual_verified_rows=0,independent_accounting=bound,
                  training_READY=False,original_replay_allowed=False)
    if not terminal:
        return result
    result['state'] = 'ORIGINAL_TERMINAL_UNQUALIFIED'
    if row[2:4] != ['COMPLETED','0:0']:
        return result
    try:
        values = dict(item.split('=',1) for item in row[6].split(','))
        need(allocated_dcu(row[6]) == 0 and values.get('cpu') == '8' and values.get('node') == '1'
             and values.get('mem') in ('12G','12288M') and row[4].isdecimal() and int(row[4]) <= 240,
             'original CPU88 resource or elapsed record differs')
        outcome = json.loads(read_raw(root/'outcome.json'))
        need(outcome['request'] == REQUEST and outcome['accounting_columns'] == ','.join(COLUMNS)
             and outcome['publication'] == publication and outcome['epoch'] == epoch and outcome['deadline'] == epoch+300
             and outcome['terminal']['terminal_visible'] is True and outcome['terminal']['row'] == row,
             'original controller does not bind the live accounting row')
        reader = load_reader()
        context = SimpleNamespace(request=REQUEST,publication=publication,epoch=epoch,
                                  source_manifest_sha256=MANIFEST,claim_sha256=CLAIM)
        proof = reader(context)
        need(proof['state'] == 'MINICPM_USGS88_NATIVE_CPU_INDEPENDENT_METADATA_PASS' and proof['passed'] is True
             and proof['request'] == REQUEST and proof['publication'] == publication and proof['actual_CPU_step'] == row[0]
             and proof['actual_step_elapsed_seconds'] == int(row[4]) and proof['completed_rows'] == 88
             and proof['actual_GPU_allocation'] == 0 and proof['actual_optimizer_updates'] == 0,
             'original independent metadata result differs')
        result.update(proof,actual_verified_rows=88,independent_accounting=bound,original_replay_allowed=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['reader_error'] = f'{type(error).__name__}: {str(error)[:240]}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication',required=True)
    parser.add_argument('--epoch',required=True,type=float)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    need(not args.output.resolve().is_relative_to(RUN.resolve()),'write the independent receipt outside the original run')
    def expired(*_):
        raise ReadDeadline('original CPU88 independent reading budget expired')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,60)
    raw = subprocess.check_output(['sacct','-n','-P','-j',PARENT,'-S','2026-11-19T00:00:00','-o',','.join(COLUMNS)],timeout=10)
    report = verify(RUN,raw,args.publication,args.epoch)
    with args.output.open('xb') as stream:
        body=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(state=report['state'],passed=report['passed'],request=REQUEST,
                          verified_rows=report['actual_verified_rows'],receipt_SHA256=sha256(body).hexdigest())),flush=True)
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
