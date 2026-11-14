"""Qualify the actual SDK policy on observed shapes before native model loads."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_flash2_policy_probe import run
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True);p.add_argument('--epoch',required=True,type=float);p.add_argument('--deadline',required=True,type=float);a=p.parse_args()
 binding=json.loads(Path(a.binding).read_bytes());sys.path.insert(0,binding['source_root'])
 print(json.dumps(run(a.output,epoch=a.epoch,deadline=a.deadline)))
