"""Exact observed vendor-version/sentinel contract; no process or request-terminal inference."""
import re

VERSION='24.05.8-3.1.3-668-20260723'
PARENT='1103606'

def need(ok,reason):
    if not ok:raise ValueError(reason)

def fields(line):
    result={}
    for word in line.split():
        if '=' not in word:continue
        key,value=word.split('=',1)
        need(key not in result,'DUPLICATE_STEP_FIELD')
        result[key]=value
    return result

def versions(scontrol,srun,server_lines):
    need(scontrol==['slurm '+VERSION] and srun==['slurm '+VERSION],'EXACT_CLIENT_VERSION')
    selected=[line for line in server_lines if line.split('=',1)[0].strip()=='SLURM_VERSION']
    need(len(selected)==1 and selected[0].split('=',1)[1].strip()==VERSION,'EXACT_SERVER_VERSION')
    return dict(scontrol=VERSION,srun=VERSION,server=VERSION,vendor_binary_equivalence_proven=False,
        original_scontrol=scontrol,original_srun=srun,original_server_version_lines=selected)

def classify(row,parent_hosts,step_hosts,uid):
    step=row.get('StepId','')
    need(len(parent_hosts)==len(set(parent_hosts))==40,'PARENT_NODESET40')
    if re.fullmatch(PARENT+r'\.[0-9]+',step):
        need(bool(row.get('NodeList')),'UNKNOWN_ACTIVE_STEP')
        return 'NUMBERED_STEP_REQUIRES_NODE_CONFLICT_CHECK'
    need(step==PARENT+'.TDB','UNKNOWN_ACTIVE_STEP')
    need(row.get('State')=='PENDING' and row.get('StartTime')=='Unknown'
         and row.get('Name')=='(null)' and row.get('TRES')=='(null)','NOT_EXACT_PENDING_SENTINEL')
    need(row.get('UserId')==str(uid)=='154459','PENDING_SENTINEL_OWNER')
    need(bool(row.get('NodeList')) and len(step_hosts)==len(set(step_hosts))==40
         and set(step_hosts)==set(parent_hosts),'PENDING_SENTINEL_PARENT_NODESET')
    identity=row.get('SrunHost:Pid','')
    host_label=r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?'
    need(re.fullmatch(host_label+r'(?:\.'+host_label+r')*:[1-9][0-9]*',identity)
         is not None and len(identity.rsplit(':',1)[0])<=253
         and int(identity.rsplit(':',1)[1])<=2147483647,'PENDING_SENTINEL_SRUN_IDENTITY_SHAPE')
    for key in ('AllocTRES','Gres','GRES','TresPerNode','TresPerTask','TresPerStep'):
        need(key not in row or row[key] in ('(null)','N/A','None','0'),'PENDING_SENTINEL_ALLOCATION_FIELD')
    return 'PENDING_NOTIFICATION_PLACEHOLDER_NO_ALLOCATED_TRES'
