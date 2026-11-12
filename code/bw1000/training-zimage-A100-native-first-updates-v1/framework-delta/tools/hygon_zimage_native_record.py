"""Unified entry for a bound original native recording or fresh resume phase."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_native_record import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--mode',choices=('continuous','resume20'),required=True);p.add_argument('--checkpoint')
    a=p.parse_args()
    result=run(json.loads(Path(a.binding).read_bytes()),a.output,mode=a.mode,epoch=a.epoch,deadline=a.deadline,checkpoint=a.checkpoint)
    print(json.dumps(dict(passed=result['passed'],mode=a.mode,actual_optimizer_updates=result['actual_optimizer_updates'],formal_performance=False,Model_GO=False)))

if __name__=='__main__':main()
