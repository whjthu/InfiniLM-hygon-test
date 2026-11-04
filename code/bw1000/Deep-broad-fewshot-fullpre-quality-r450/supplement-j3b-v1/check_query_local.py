"""Necessary new helper only: recording + real owned stdout/stderr/timeout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import query_receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=[]
    with tempfile.TemporaryDirectory(prefix='Deep450-j3b-query-')as folder:
        root=Path(folder)
        class Recorded:
            pid=0;returncode=0
            def communicate(self,timeout=None):return('recorded out\n','recorded err\n')
            def poll(self):return 0
        called=[]
        def leaf(argv,**kwargs):called.append((argv,kwargs));return Recorded()
        path=root/'recording.json';got=query_receipt.run(['recording-only'],path,{'recording':True},popen=leaf)
        row=json.loads(path.read_bytes());assert got=='recorded out\n'and len(called)==1 and row['timeout_s']==25 and row['phase']=='returned'
        cases.append(dict(name='new_helper_recording_default25',passed=True,receipt=row))
        path=root/'echo.json';got=query_receipt.run([sys.executable,'-B','-c','import sys;print("owned stdout");print("owned stderr",file=sys.stderr)'],path,{'self_owned_CPU':True})
        row=json.loads(path.read_bytes());assert got=='owned stdout\n'and row['stderr']=='owned stderr\n'and row['returncode']==0
        cases.append(dict(name='new_helper_actual_self_owned_echo',passed=True,receipt=row))
        path=root/'timeout.json'
        try:query_receipt.run([sys.executable,'-B','-u','-c','import time;print("before-owned-timeout",flush=True);time.sleep(5)'],path,{'self_owned_CPU':True,'test_short_timeout_not_production':True},timeout=.3)
        except subprocess.TimeoutExpired:pass
        else:raise AssertionError('Owned timeout was not raised')
        row=json.loads(path.read_bytes());assert row['timeout']and row['phase']=='TIMEOUT'and row['returncode']is not None and 'before-owned-timeout'in row['stdout']
        cases.append(dict(name='new_helper_actual_self_owned_timeout',passed=True,receipt=row))
    out=dict(passed=True,completed=True,cases=cases,Slurm=0,GPU=0,Native=False,SDK=False,Model=False,old_CPU_suites_replayed=False,production_command_timeout_unchanged=25)
    a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=True,cases=len(cases),GPU=0,Slurm=0)))


if __name__=='__main__':main()
