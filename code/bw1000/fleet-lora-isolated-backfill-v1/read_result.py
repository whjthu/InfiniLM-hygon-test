"""Consume only receipts for the claimed fleet run and its two exact scheduler labels."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from qualified_sampler import aggregate, node_map


def read(path, limit=16384):
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError('RECEIPT_SIZE')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def valid_pair(result):
    checks = [result['comparison_relative_l2'], *result['changed_input_relative_l2']]
    if len(checks) != 4 or any(len(row) != 4 or
                               any(not math.isfinite(x) or not 0 <= x <= .005 for x in row)
                               for row in checks):
        return False
    arms = result['measurement']
    return set(arms) == {'autograd', 'reuse', 'graph'} and all(
        len(rows) == 4 and all(
            math.isfinite(row['wall_seconds']) and row['wall_seconds'] >= 6
            and type(row['complete_forward_backward_calls']) is int
            and row['complete_forward_backward_calls'] > 0
            and row['complete_forward_backward_calls'] % 8 == 0 for row in rows)
        for rows in arms.values())


def main(root, request, final):
    intent, _ = read(root/'intent.json')
    if intent is None or intent['request'] != request:
        raise ValueError('ORIGINAL_INTENT_REQUIRED')

    data = dict(request=request, epoch=intent['epoch'], observed_epoch=time.time(), steps={},
                accountable_active_cards=0, first_verified_cards=0, final_cards=0, numeric_pairs=0,
                successful_cards=0, early_exit_cards=0, missing_final_nodes=0, errors={})
    columns = 'JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
    steps = {}
    for parent, label in intent['labels'].items():
        raw = subprocess.check_output(['sacct','-n','-P','-j',parent,'--format='+columns], timeout=15)
        rows = [line.split('|') for line in raw.decode().splitlines() if line.split('|')[1:2] == [label]]
        if len(rows) > 1 or (rows and not re.fullmatch(parent+r'\.\d+',rows[0][0])):
            raise ValueError('EXACT_STEP')
        data['steps'][parent] = rows
        if rows:
            steps[parent] = rows[0]
    if final and (len(steps) != 2 or any(r[2] in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED')
                                        for r in steps.values())):
        raise ValueError('ACTUAL_TERMINAL_ACCOUNTING_REQUIRED')

    owners = node_map(json.loads((root/'ledger.json').read_bytes()))
    pins, samples, hashes = {}, {}, {}
    for node, (parent, _) in owners.items():
        if parent not in steps:
            continue
        directory = root/'nodes'/node
        step = steps[parent][0]
        node_pids = set()
        def checked(name):
            row, sha = read(directory/name)
            if row is not None:
                if (row['node'], row['step'], row['launch_epoch']) != (node, step, intent['epoch']):
                    raise ValueError('RECEIPT_IDENTITY')
                hashes[node+'/'+name] = sha
            return row
        inventory = checked('inventory.json')
        if inventory:
            pins[node] = sorted(inventory['PCI_KFD'])
            report = checked('busy.json')
            if report:
                samples[node] = report
        terminal = checked('terminal.json')
        data['missing_final_nodes'] += int(terminal is None)
        for card in range(8):
            active = checked(f'active-{card}.json')
            done = checked(f'card-{card}.json')
            if active:
                if active['card'] != card or active['pid'] in node_pids:
                    raise ValueError('INDEPENDENT_CARD_PROCESS_REQUIRED')
                node_pids.add(active['pid'])
                data['first_verified_cards'] += 1
                if done is None and terminal is None and steps[parent][2] == 'RUNNING':
                    data['accountable_active_cards'] += 1
            if done:
                data['final_cards'] += 1
                if done['card'] != card or (active and done['pid'] != active['pid']):
                    raise ValueError('CARD_IDENTITY')
                if done['error']:
                    error = done['error']['stage']
                    data['errors'][error] = data['errors'].get(error, 0)+1
                valid = sum(valid_pair(result) for result in done['results'])
                data['numeric_pairs'] += valid
                if not done['error'] and valid == len(done['results']) == 2 and active:
                    data['successful_cards'] += 1
            elif terminal and str(card) in terminal['exits']:
                data['early_exit_cards'] += 1
    data['busy_observed_nodes'] = len(samples)
    if len(pins) == 80:
        data['busy'] = aggregate(owners,pins,samples,intent['epoch']+120)
    data['receipt_pins_SHA256'] = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    data['planned_cards'] = 640
    data['planned_card_shape_pairs'] = 1280
    data['formal_training_updates'] = 0
    data['complete'] = (len(steps) == 2 and all(r[2:4] == ['COMPLETED','0:0'] for r in steps.values())
                        and data['missing_final_nodes'] == 0 and data['successful_cards'] == 640
                        and data['numeric_pairs'] == 1280)
    print(json.dumps(data, allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'terminal')
