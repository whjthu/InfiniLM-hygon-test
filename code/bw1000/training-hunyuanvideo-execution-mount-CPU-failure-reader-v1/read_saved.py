"""Offline only: consume one new failed-prelaunch original and current exact identifiers."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

D=Path(__file__).resolve().parent
G=D.parents[2]
H=lambda raw:hashlib.sha256(raw).hexdigest()


def need(ok,reason):
    if not ok:raise ValueError(reason)


def read(path,expected=None,cap=2*1024**2):
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=cap,'BOUNDED_ORIGINAL')
    raw=path.read_bytes()
    if expected:
        need(H(raw)==expected,'EXACT_ORIGINAL_BYTES_SHA')
    return raw,dict(path=path.relative_to(G).as_posix(),bytes=len(raw),SHA256=H(raw))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    B=json.loads((D/'BINDING.json').read_bytes());evidence={}
    raw,evidence['client']=read(G/B['client_directory']/'receipt.json',B['client_receipt_SHA256']);client=json.loads(raw)
    raw,evidence['intent']=read(G/B['client_directory']/'intent.json');intent=json.loads(raw)
    raw,evidence['attempt']=read(G/B['client_directory']/'attempt.json');attempt=json.loads(raw)
    stderr,evidence['stderr']=read(G/B['client_directory']/'stderr.private',B['client_stderr_SHA256'])
    need(len(stderr)==B['client_stderr_bytes'] and b'ValueError: UNKNOWN_ACTIVE_STEP' in stderr,'ACTUAL_KNOWN_FRONTGATE_EXCEPTION')
    for value in (client,intent,attempt):
        need(value['request_id']==B['request_id'] and value['epoch']==B['epoch'],'ACTUAL_ORIGINAL_REQUEST')
    need(client['publication']==B['publication'] and client['source_manifest_SHA256']==B['source_manifest_SHA256']
         and client['rc']==1 and client['stdout_bytes']==0 and client['stderr_bytes']==847
         and client['exec_request_attempted'] is True and attempt['exec_request_attempted'] is True,'ACTUAL_CLIENT_REMOTE_FAILURE')
    raw,evidence['reconcile']=read(G/B['reconcile']['path'],B['reconcile']['SHA256']);reconcile=json.loads(raw)
    raw,evidence['candidate_identity_read']=read(G/B['candidate_identity_read']['path'],B['candidate_identity_read']['SHA256']);candidate=json.loads(raw)
    need(reconcile['request_id']==B['request_id'] and reconcile['read_only'] is True
         and reconcile['weight_body_reads']==reconcile['new_steps']==0,'ORIGINAL_READONLY_RECONCILE')
    records=reconcile['files'];installed=records['install-intent.json']['value'];ready=records['install-ready.json']['value']
    for value in (installed,ready):
        need(value['request_id']==B['request_id'] and value['publication']==B['publication']
             and value['source_manifest_SHA256']==B['source_manifest_SHA256'] and value['epoch']==B['epoch']
             and value['deadline']==B['epoch']+900,'ORIGINAL_INSTALL_CONTEXT')
    missing=['intent.json','outcome.json','owned-srun.json','worker-intent.json','weight-receipt.json']
    need(all(records[name]['state']=='MISSING' for name in missing),'NEW_CPU_DISPATCH_RECORD_EXISTS')
    need(reconcile['queries']['accounting']['rc']==0 and reconcile['queries']['accounting']['rows']==[], 'EXACT_CPU_LABEL_HAS_ACCOUNTING')
    need(reconcile['queries']['steps']['rc']==0,'CURRENT_STEP_SCHEMA_QUERY_FAILED')
    rows=[dict(word.split('=',1) for word in line.split() if '=' in word) for line in reconcile['queries']['steps']['rows']]
    tdb=[row for row in rows if row.get('StepId')=='1103606.TDB'];need(len(tdb)==1,'EXACT_TDB_ROW')
    tdb=tdb[0]
    need(tdb.get('NodeList') and tdb['State']=='PENDING' and tdb['UserId']=='154459'
         and tdb['StartTime']=='Unknown' and tdb['Name']=='(null)' and tdb['SrunHost:Pid']=='zz-login02:3940770',
         'ACTUAL_TDB_IS_NOT_MISSING_NODELIST')
    external=next(row for row in rows if row.get('StepId')=='1103606.extern')
    need(tdb['NodeList']==external['NodeList'] and tdb['Nodes']=='40','TDB_COVERS_PARENT_NODELIST')
    need(candidate['state']=='OTHER_HOST_IDENTITY_UNKNOWN' and candidate['candidate_PID']==3940770
         and candidate['expected_host']=='zz-login02' and candidate['host']=='zz-login01'
         and candidate['read_only'] is True and candidate['signals_sent']==0,'CROSS_HOST_IDENTITY_BOUNDARY')
    source=G/B['original_cpu_source']
    raw,evidence['published_source_MF']=read(source/'SOURCE_FILES.json',B['source_manifest_SHA256']);mf=json.loads(raw)
    raw,evidence['launcher']=read(source/'launch_once.py',mf['files']['launch_once.py']['SHA256'])
    tree=ast.parse(raw);mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    guards=[n for n in ast.walk(mainnode) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='need'
            and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='UNKNOWN_ACTIVE_STEP']
    launches=[n for n in ast.walk(mainnode) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    need(len(guards)==len(launches)==1 and guards[0].lineno<launches[0].lineno,'PUBLISHED_GUARD_NOT_BEFORE_ONLY_SRUN_POPEN')
    current_mapping=[]
    for relative in B['current_mapping_search_files']:
        raw,pin=read(G/relative);value=json.loads(raw)
        found=[]
        def find(value,path=''):
            if isinstance(value,dict):
                for key,item in value.items():
                    location=path+'/'+str(key)
                    if '3940770' in str(key) or '1103606.TDB' in str(key):found.append(location)
                    find(item,location)
            elif isinstance(value,list):
                for index,item in enumerate(value):find(item,path+'/'+str(index))
            elif isinstance(value,(str,int)) and ('3940770' in str(value) or '1103606.TDB' in str(value)):found.append(path)
        find(value);current_mapping.append(dict(**pin,matches=found))
    result=dict(schema='T09_CPU_ORIGINAL_PRELAUNCH_UNKNOWN_STEP_FAILCLOSED_SAVED_METADATA_V1',
        request_id=B['request_id'],publication=B['publication'],source_manifest_SHA256=B['source_manifest_SHA256'],
        epoch=B['epoch'],deadline=B['epoch']+900,original_state='REMOTE_PYTHON_EXIT1_BEFORE_NEW_CPU_SRUN',
        actual_client_rc=1,actual_client_elapsed_seconds=client['elapsed_seconds'],actual_stdout_bytes=0,
        failure='UNKNOWN_ACTIVE_STEP',guard_correctly_failed_closed=True,NodeList_field_present=True,
        NodeList_vs_Nodes_parser_bug_proven=False,new_CPU_step_submissions=0,weight_body_bytes_read=0,
        new_GPU_calls=0,new_optimizer_updates=0,zero_work_proof='Clientactualrc1+unique published source guard before onlysrunPopen+actualmissing launch/worker records+exactlabelaccountingempty. Not inferred from missingreceipt alone.',
        installed_source_members=ready['completed_source_members'],install_source_bytes=installed['payload_bytes'],
        original_installer_identity=dict(host=installed['host'],PID=installed['PID'],startticks=installed['startticks'],boot_id=installed['boot_id']),
        original_installer_process_live='UNKNOWN_NOT_QUERIED_ON_ORIGINAL_HOST',
        reconcile_observer_host=reconcile['host'],reconcile_observer_differs_from_installer=reconcile['host']!=installed['host'],
        blocking_step=dict(handle='1103606.TDB',state='PENDING',start='Unknown',name_is_null=True,UID=154459,
                           CPU_request=2,nodes=40,NodeList_matches_original_parent=True,reported_srun_host='zz-login02',reported_srun_PID=3940770,
                           provenance_mapping='UNKNOWN_NO_MATCH_IN_LIMITED_CURRENT_FILES' if not any(row['matches'] for row in current_mapping) else 'CURRENT_FILE_REFERENCES_REQUIRE_ROOT_OWNERSHIP_REVIEW'),
        candidate_process_live='UNKNOWN_OTHER_HOST',candidate_observer_host=candidate['host'],candidate_proc_reads=0,signals_sent=0,
        original_GLM188_reservation_retained=True,no_unknown_step_ignored_or_cancelled=True,scope_replay_allowed=False,
        replacement_v2_prepared=False,CPU_GO=False,GPU_GO=False,fullweight_execution_mount_qualified=False,
        missing_original_replay_inputs_unchanged=True,evidence=evidence,current_limited_mapping_search=current_mapping,
        next='Retain originalTDBUNKNOWN. Rootmayconsume onlynewexactoriginalhost/PIDstartticks provenance evidence; no reconnectloop/fallback/blindcancel and no sameCPU retry. Do not repair correct numeric-step guard by ignoringTDB.')
    with Path(args.output).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('original_state','guard_correctly_failed_closed','new_CPU_step_submissions','weight_body_bytes_read','candidate_process_live','scope_replay_allowed')}))


if __name__=='__main__':main()
