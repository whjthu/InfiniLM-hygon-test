"""Exact .4100 observation and terminal-only admission for the n14 revisit."""

from datetime import datetime, timezone
import os
import re


PARENT='1103606'
NODE='f13r3n14'
HANDLE='1103606.4100'
LABEL='training80-long-busy-d895426c188d40dd9593ccad27c90cf1-1103606'
NODELIST='f12r2n[10-13],f13r1n[01-05,07-15,17-20],f13r2n[01-11],f13r3n[13-14],f13r4n[05-09]'
TRES={'cpu=40','mem=10G','node=40'}
FORMAT='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
TIME='%Y-%m-%dT%H:%M:%S'


def need(ok,code):
    if not ok:raise ValueError(code)


def observer_accounting(read,nodes):
    lines=read(['env','TZ=UTC','SLURM_TIME_FORMAT='+TIME,'sacct','-nP','-j',HANDLE,'--format',FORMAT])
    need(len(lines)==1,'N14_EXACT_OBSERVER_ONE_ACCOUNTING_ROW')
    row=lines[0].split('|')
    need(len(row)==9 and row[0:2]==[HANDLE,LABEL] and row[3]=='0:0' and row[4].isdecimal()
         and len(row[6].split(','))==3 and set(row[6].split(','))==TRES and nodes(row[5])==nodes(NODELIST)
         and NODE in nodes(row[5]) and len(nodes(row[5]))==40,'N14_EXACT_4100_IDENTITY_ZERO_GRES_AND_MEMBERSHIP')
    return row


def observer_policy(read,nodes,read_only,now):
    row=observer_accounting(read,nodes)
    def epoch(value):
        return datetime.strptime(value,TIME).replace(tzinfo=timezone.utc).timestamp()
    if row[2]=='COMPLETED':
        start,end=map(epoch,row[7:9])
        need(0<start<=end<=now+1 and abs(end-start-int(row[4]))<=1,'N14_EXACT_4100_FINITE_UTC_TERMINAL')
        return None
    need(row[2]=='RUNNING' and read_only is True and epoch(row[7])<=now+1,
         'N14_4100_MUST_BE_TERMINAL_BEFORE_PRODUCTION')
    return row


def allowed_controller(item,observer,nodes):
    need(observer is not None and item['StepId']==HANDLE and item.get('Name',item.get('JobName'))==LABEL
         and item['State']=='RUNNING' and nodes(item['NodeList'])==nodes(NODELIST)
         and len(item.get('TRES',item.get('AllocTRES','')).split(','))==3
         and set(item.get('TRES',item.get('AllocTRES','')).split(','))==TRES,
         'N14_CONTROLLER_ZERO_GRES_OBSERVER_ONLY')
    return True


def allowed_queue(handle,spec,observer,nodes):
    need(observer is not None and handle==HANDLE and nodes(spec)==nodes(NODELIST),
         'N14_QUEUE_ZERO_GRES_OBSERVER_ONLY')
    return True


def main():
    """Root-only read-only HOST projection, independent of submission."""
    import argparse
    import json
    import subprocess
    import time
    from placement import live_views
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read-only-observer',action='store_true')
    args=parser.parse_args()
    need(os.getuid()==154459,'EXACT_PROJECT_UID')
    def query(argv):
        adjusted=argv if argv[0]=='env' else ['env','TZ=UTC',*argv]
        out=subprocess.run(adjusted,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=8,check=False)
        need(out.returncode==0 and len(out.stdout)<=2<<20 and len(out.stderr)<=65536,'BOUNDED_HOST_DUAL_VIEW')
        return out.stdout.decode('ascii').splitlines()
    now=time.time()
    result=live_views(query,os.getuid(),now,read_only_observer=args.read_only_observer)
    print(json.dumps(dict(passed=result,request_scope='T09_REPEAT3_N14_DUAL_VIEW',node=NODE,parent=PARENT,
                          read_only_observer=args.read_only_observer,actual_GPU=0,observed_epoch=now)))


if __name__=='__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    main()
