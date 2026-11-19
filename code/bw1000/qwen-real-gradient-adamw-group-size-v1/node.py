"""Run the real-gradient comparison once per allocated physical device."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    cards=range(8)
    processes={}
    try:
        for card in cards:
            with (ROOT/f'worker-{card}.log').open('xb') as log:
                processes[card]=subprocess.Popen([sys.executable,'-B',str(ROOT/'worker.py'),str(card)],
                                          stdout=log,stderr=subprocess.STDOUT)
        exits={card: process.wait() for card,process in processes.items()}
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
    receipts={}
    for card in cards:
        path=ROOT/f'device-{card}'/'result.json'
        if path.is_file():
            receipts[card]=json.loads(path.read_bytes())
    expected=set(json.loads((ROOT/'lease-before.json').read_bytes())['pci'])
    actual={row['identity']['PCI'] for row in receipts.values() if row.get('identity')}
    passed=(len(receipts)==8 and actual==expected and all(exits[i]==0 and receipts[i]['passed'] for i in cards))
    with (ROOT/'result.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(exits=exits,card_receipts=receipts,passed=passed),stream,allow_nan=False)
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
