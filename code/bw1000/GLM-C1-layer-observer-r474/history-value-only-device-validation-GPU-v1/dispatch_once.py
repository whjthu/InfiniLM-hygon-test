"""Fresh admission, reserve, oncekeeper/GPU, independentpost, then finish."""
from datetime import datetime
from common import *
def idle(budget,pre_already_terminal=False):
    lines=query(budget,'scontrol','show','job',PARENT,'--oneliner');need(len(lines)==1,'PARENT_REQUIRED');j=dict(x.split('=',1) for x in lines[0].split() if '=' in x)
    uid=re.fullmatch(r'(?:[^()\s]+\((\d+)\)|(\d+))',j.get('UserId',''));need(uid and int(uid[1] or uid[2])==os.getuid() and j.get('JobState')=='RUNNING' and j.get('NumNodes')=='40' and 'gres/dcu=320' in j.get('AllocTRES','').split(','),'PARENT_UID_RESOURCE')
    need({GPU_NODE,KEEPER_NODE}<=set(query(budget,'scontrol','show','hostnames',j['NodeList'])) and datetime.fromisoformat(j['EndTime']).timestamp()>=budget.epoch+690,'PARENT_NODES_DEADLINE')
    for line in query(budget,'scontrol','show','step',PARENT,'--oneliner'):
        r=dict(x.split('=',1) for x in line.split() if '=' in x);handle=r.get('StepId','');need(re.fullmatch(PARENT+r'\.(?:[0-9]+|batch|extern)',handle),'STEP_SCHEMA')
        if handle.endswith(('.batch','.extern')):continue
        need(not ({GPU_NODE,KEEPER_NODE}&set(query(budget,'scontrol','show','hostnames',r['NodeList']))),'ACTIVE_STEP_CONFLICT')
    for phase in (('keeper','gpu','post') if pre_already_terminal else ('pre','keeper','gpu','post')):need(account(budget,LABEL+'-'+phase) is None,'EXACT_LABEL_DEDUP')
    return j
def main(publication,claim_sha):
    install=load(W/'install-intent.json');budget=Budget(install['epoch']);q=dict(request_id=install['request_id'],publication=publication,epoch=install['epoch'],deadline=install['epoch']+660,passed=False,state='UNKNOWN',replay_allowed=False,reservation_released=False,GPU_GO=False,Model_GO=False,phases={})
    try:
        claim=load(D/'CLAIM.json');need(claim['ROOT_GO'] and claim['GPU_GO'] and claim['device_validation_GO'] and H((D/'CLAIM.json').read_bytes())==claim_sha,'PUBLISHED_GPU_CLAIM');need(publication==install['publication'],'INSTALL_PUBLICATION')
        qualify_cpu(host=True);budget.left(100,60);parent=idle(budget);save(W/'HOST-preflight.json',dict(parent=parent,epoch=install['epoch'],node=GPU_NODE,keeper=KEEPER_NODE),True)
        pre_start_ns=time.monotonic_ns()
        pre=step(budget,'pre',GPU_NODE+','+KEEPER_NODE,1,1024,0,100,['/usr/local/python3.12/bin/python3','-B',str(D/'host_probe.py'),'pre']);pre_end_ns=time.monotonic_ns();q['phases']['pre']=pre;need(pre['passed'],'PRE_FAILED_NO_GPU')
        payloads={}
        for node in (GPU_NODE,KEEPER_NODE):
            raw=(W/('pre-'+node+'.json')).read_bytes();value=json.loads(raw)
            need(value['node']==node and value['step']==pre['row'][0] and value['kfd_holder_count']==0 and value['same_uid_fd_denied_count']==0 and value['port_bindable'],'PRE_NODE_BINDING')
            payloads[node]=dict(SHA256=H(raw),bytes=len(raw),boot_id=value['boot_id'])
        dispatch_identity=owner(os.getpid())
        returned=dict(request_id=install['request_id'],publication=publication,source_manifest_SHA256=install['source_manifest_SHA256'],claim_SHA256=claim_sha,step=pre['row'][0],observer_host=dispatch_identity['host'],observer_boot_id=dispatch_identity['boot_id'],pre_start_monotonic_ns=pre_start_ns,pre_end_monotonic_ns=pre_end_ns,payloads=payloads)
        save(W/'pre-return.json',returned,True)
        before=load(W/('pre-'+GPU_NODE+'.json'))
        idle(budget,pre_already_terminal=True)
        # Both elapsed values are local to this same dispatch process/host.
        reserve_ns=time.monotonic_ns();need(0<=pre_end_ns-pre_start_ns<=90_000_000_000 and pre_end_ns<=reserve_ns and reserve_ns-pre_start_ns<=90_000_000_000,'PRE_RETURN_LOCAL_MONOTONIC_STALE')
        # Exclusive scopedintent, followed by Slurm's atomicresource allocation.
        # This file doesnotclaim a historicalinventory is currentdeviceownership.
        save(W/'reservation.json',dict(request_id=install['request_id'],publication=publication,source_manifest_SHA256=install['source_manifest_SHA256'],node=GPU_NODE,physical_GPU_cards=1,keeper_node=KEEPER_NODE,epoch=install['epoch'],state='RESERVED',pre_SHA256=H((W/('pre-'+GPU_NODE+'.json')).read_bytes()),pre_return_SHA256=H((W/'pre-return.json').read_bytes()),reserve_dispatch_monotonic_ns=reserve_ns),True)
        control=load(W/'control-state.json');need(control['exec_request_attempted'] is False and all(control[k]==install[k] for k in ('request_id','publication','epoch','source_manifest_SHA256','claim_SHA256')),'BOUND_FALSE_CONTROL');control.update(exec_request_attempted=True,phase='KEEPER_POPEN_IMMINENT');save(W/'control-state.json',control)
        keeper=step(budget,'keeper',KEEPER_NODE,2,1024,0,570,['/usr/local/python3.12/bin/python3','-B',str(D/'keeper.py')]);q['phases']['keeper']=keeper
        need(keeper['terminal'],'KEEPER_UNKNOWN_RETAIN');record=load(W/'keeper-result.json');gpu=record.get('GPU');q['phases']['gpu']=gpu
        need(gpu and gpu['terminal'] and record['keeper_step']==keeper['row'][0],'GPU_UNKNOWN_RETAIN')
        post=step(budget,'post',GPU_NODE+','+KEEPER_NODE,1,1024,0,650,['/usr/local/python3.12/bin/python3','-B',str(D/'host_probe.py'),'post']);q['phases']['post']=post;need(post['passed'],'POST_NOT_INDEPENDENTLY_CLEAN')
        after=load(W/('post-'+GPU_NODE+'.json'));need(after['step']==post['row'][0] and after['pci']==before['pci'] and after['boot_id']==before['boot_id'] and after['monotonic_ns']>=before['monotonic_ns'] and len(after['known_owners'])==2,'POST_IDENTITY')
        keeper_before=load(W/('pre-'+KEEPER_NODE+'.json'));keeper_after=load(W/('post-'+KEEPER_NODE+'.json'));need(keeper_after['step']==post['row'][0] and keeper_after['pci']==keeper_before['pci'] and keeper_after['boot_id']==keeper_before['boot_id'] and len(keeper_after['known_owners'])==1,'KEEPER_POST_IDENTITY')
        reservation=load(W/'reservation.json');reservation.update(state='RELEASED_AFTER_GPU_KEEPER_TERMINAL_AND_POST',post_SHA256=H((W/('post-'+GPU_NODE+'.json')).read_bytes()));save(W/'reservation.json',reservation);q['reservation_released']=True
        result=load(W/'device-result.json');q['device_result_SHA256']=H((W/'device-result.json').read_bytes());q['passed']=bool(keeper['passed'] and gpu['passed'] and record['passed'] and result['passed'] and result['step']==gpu['row'][0]);q['state']='TERMINAL_CLEAN'
    except BaseException as error:q.update(error_type=type(error).__name__,error_message_SHA256=H(str(error).encode()))
    finally:
        control=load(W/'control-state.json');q.update(exec_request_attempted=control['exec_request_attempted'],control_checkpoint_verified=True,control_state_SHA256=H((W/'control-state.json').read_bytes()),current_step_outcome='TERMINAL' if q['state']=='TERMINAL_CLEAN' else 'NOT_SUBMITTED' if control['exec_request_attempted'] is False else 'UNKNOWN');save(W/'outcome.json',q,True)
        print(json.dumps(dict(request_id=q['request_id'],passed=q['passed'],current_step_outcome=q['current_step_outcome'],exec_request_attempted=q['exec_request_attempted'],control_checkpoint_verified=True,control_state_SHA256=q['control_state_SHA256'],outcome_SHA256=H((W/'outcome.json').read_bytes()),reservation_released=q['reservation_released'],GPU_GO=False,Model_GO=False,replay_allowed=False),allow_nan=False),flush=True)
    return 0 if q['passed'] else 1
