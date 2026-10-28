"""Pure CPU source/contract entry: no Slurm, profile generation or device work."""
import argparse,hashlib,json,pathlib,runpy,sys
D=pathlib.Path(__file__).resolve().parent.parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args()
 assert hashlib.sha256((D/'service-source-manifest-r194.json').read_bytes()).hexdigest()==a.manifest_sha
 runpy.run_path(str(D/'CPU_contract_r194.py'),run_name='__main__')
if __name__=='__main__':main()
