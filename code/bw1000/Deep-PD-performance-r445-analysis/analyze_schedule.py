"""Audit scheduling intervals on the single original client clock; no replay."""
import argparse,base64,gzip,hashlib,json,math
from pathlib import Path

H=lambda b:hashlib.sha256(b).hexdigest()
SEAL='7c0baebaf26f41c6aff0a3d5ef5544b025a253a54937d1f2269587fd235d1b6e'

def duration(intervals):
    total=0.;end=-math.inf
    for a,b in sorted(intervals):
        assert math.isfinite(a)and math.isfinite(b)and b>=a
        total+=max(0.,b-max(a,end));end=max(end,b)
    return total

def audit(path):
    raw=Path(path).read_bytes();assert H(raw)==SEAL
    seal=json.loads(raw);files={}
    for f in seal['files']:
        content=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True))
        assert len(content)==f['bytes']and H(content)==f['SHA256']
        name=Path(f['path']).name;assert name not in files;files[name]=content
    c=json.loads(files['Deep382-pd445-20261104-a-client.json'])
    assert c['completed']and c['execution_shutdown_completed']and c['all16_actual_owners_destroyed']
    assert len(c['requests'])==c['planned_external']==22 and c['all_service_cards']==16
    groups=[]
    for group in c['groups']:
        if group['phase']!='capacity':continue
        rows=[r for r in c['requests']if r['request']['group']==group['id']]
        assert len(rows)==4 and all(r['client']['completed']for r in rows)
        start=group['client_group_start_monotonic'];end=group['client_group_end_monotonic'];wall=end-start
        occupied={'P':[],'D':[]};holds=[];details=[]
        for row in rows:
            r=row['request'];commands=row['commands'];events=row['events']
            def command(role,op):
                matches=[x['client_send_monotonic']for x in commands if x['role']==role and x['operation']==op]
                assert len(matches)==1;return matches[0]
            def event(role,kind):
                matches=[x['client_received_monotonic']for x in events if x['role']==role and x['kind']==kind]
                assert len(matches)==1;return matches[0]
            if r['mode']=='co':
                role=r['co_role'];a=command(role,'co');b=event(role,'stage_done')
                assert start<=a<=b<=end;occupied[role].append((a,b))
            else:
                p=command('P','prefill');ready=event('P','prefill_ready');di=command('D','import')
                ir=event('D','import_ready');tx=command('P','transfer');pd=event('P','stage_done');dd=event('D','stage_done')
                assert start<=p<=ready<=di<=ir<=tx<=pd<=end and tx<=dd<=end
                occupied['P'].append((p,pd));occupied['D'].append((di,dd));holds.append((ready,di))
                details.append(dict(index=r['index'],prompt_tokens=r['past'],output_budget=r['output_budget'],P_prefill_command_to_ready_s=ready-p,P_ready_wait_for_D_import_dispatch_s=di-ready,D_import_to_importready_s=ir-di,importready_to_P_transfer_command_s=tx-ir,P_transfer_command_to_source_done_s=pd-tx,D_import_command_to_D_done_s=dd-di))
        both=[]
        for a,b in occupied['P']:
            for x,y in occupied['D']:
                if min(b,y)>max(a,x):both.append((max(a,x),min(b,y)))
        held_by_busy_D=[]
        for a,b in holds:
            for x,y in occupied['D']:
                if min(b,y)>max(a,x):held_by_busy_D.append((max(a,x),min(b,y)))
        ptime=duration(occupied['P']);dtime=duration(occupied['D']);overlap=duration(both);hold=duration(holds);covered=duration(held_by_busy_D)
        assert abs(hold-covered)<.05 and ptime<=wall+.01 and dtime<=wall+.01
        groups.append(dict(id=group['id'],mode=group['mode'],fixture_order=[('L'if r['request']['past']==8489 else'S')for r in rows],group_wall_s=wall,role_command_occupied_s={'P':ptime,'D':dtime},both_roles_command_occupied_s=overlap,neither_role_command_occupied_s=wall-duration(occupied['P']+occupied['D']),P_ready_hold_for_D_s=hold,P_ready_hold_covered_by_existing_D_command_s=covered,P_ready_hold_fraction_of_wall=hold/wall,PD_request_intervals=details))
    assert len(groups)==4 and [g['mode']for g in groups]==['co','PD','PD','co']
    return dict(scope='Original formalC4 client-clock scheduling attribution, no new GPU or performance run',source_seal_SHA256=SEAL,client_SHA256=H(files['Deep382-pd445-20261104-a-client.json']),groups=groups,source_findings=['Scheduler keeps P busy after prefill_ready until D can import and source transfer ACK/reset completes.','The role worker is activeC1 and retains one export snapshot while waiting for D; new P prefill cannot start during that interval.','D import/receive and subsequent decode execute serially on the same role main thread.'],limits=['Command-occupied intervals include CPU/socket/CCL/idle waits; they are not GPU busy.','P ready hold is client-observed queue time, not network transfer time.','A future multi-ticket release-P design requires independent snapshot ownership, capacity and lifecycle proof; this audit does not certify it.','No prediction of ideal-PD throughput, RDMA gains, broad quality or public-SSE SLO.'],decision='Keep current single-ticket activeC1 PD parked; any future PD candidate must change measured ownership/queue bottleneck, rather than repeat this unchanged route.',new_GPU=0,new_performance_round=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seal');p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.seal);Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k!='source_findings'},indent=2))
