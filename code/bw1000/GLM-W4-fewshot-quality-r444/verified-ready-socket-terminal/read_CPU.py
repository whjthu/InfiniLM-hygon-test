"""Read only original CPU883 and repaired cross-node evidence, stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def originals(path):
    raw=path.read_bytes();q=json.loads(raw);rows={}
    for item in q['files']:
        assert not item.get('not_read'),item['path']
        data=item['rawtext'].encode('utf-8')
        assert len(data)==item['bytes']and sha(data)==item['SHA256'],item['path']
        assert item['path']not in rows
        rows[item['path']]=data
    return q,rows,dict(path=str(path),bytes=len(raw),SHA256=sha(raw),files=len(rows),raw_bytes=sum(map(len,rows.values())))


def read(source,oldpath,newpath):
    root=source.resolve();v1=root/'verified-ready-socket-CPU-v1';v2=root/'verified-ready-socket-crossnode-v2'
    m1=json.loads((v1/'SOURCE_MANIFEST.json').read_bytes());m2=json.loads((v2/'SOURCE_MANIFEST.json').read_bytes())
    assert sha((v1/'SOURCE_MANIFEST.json').read_bytes())=='c84006d36adc3d67e33ff347856556d63b0afb911b7e8254551e87ae4369c44a'
    assert sha((v2/'SOURCE_MANIFEST.json').read_bytes())=='3a291d083a7baf3f67b69fadd141f5fcd84ba0187df400ddbe524da390952178'
    for base,m in [(v1,m1),(v2,m2)]:
        for name,h in m['files_SHA256'].items():assert sha((base/name).read_bytes())==h,name
    pm=v1/'provider/SOURCE_MANIFEST.json';assert sha(pm.read_bytes())==m2['provider_manifest_SHA256']==m1['provider_manifest_SHA256']
    proof=json.loads(pm.read_bytes())
    for name,h in proof['files_SHA256'].items():assert sha((v1/'provider'/name).read_bytes())==h,name
    old,oldfiles,oldref=originals(oldpath);new,newfiles,newref=originals(newpath)
    assert old['accounting']==[['910376.883','GLM-r444-verified-ready-socket-v1-CPU','FAILED','1:0','00:00:32','a02r4n[14-15]']]
    suffix='/candidates/GLM444-verified-ready-socket-CPU-v1/'
    def get(rows,tail):
        hits=[raw for path,raw in rows.items()if path.endswith(tail)];assert len(hits)==1,tail
        return hits[0]
    old0=json.loads(get(oldfiles,suffix+'node-0.json'));old1=json.loads(get(oldfiles,suffix+'node-1.json'))
    assert sha(get(oldfiles,suffix+'node-0.json'))==m2['CPU883_receipts_SHA256']['0']
    assert sha(get(oldfiles,suffix+'node-1.json'))==m2['CPU883_receipts_SHA256']['1']
    assert 'multiple values for argument' in old0['errors'][0]['error']
    names=['actual_socket_'+n for n in ['positive','nonce','identity','missing','overlap','loaded','postload_4GiB','service_agreement','expired','truncated','oversized','duplicate_JSON','nonfinite_JSON']]
    names+=['actual_provider_'+n for n in ['positive','STOP','owner_failed','owner_exited']]+['unchanged_keeper_late_READY_latch']
    assert [c['name']for c in old1['cases']]==names and all(c['passed']for c in old1['cases'])and not old1['errors']
    assert old0['actual_whole_source_CLI']and old1['actual_whole_source_CLI']
    for case in old1['cases']:
        if case['name'].startswith('actual_provider_'):
            expected=1 if case['name'].endswith('_positive')else 0
            assert case['result']['publication_attempts']==expected
    for row in [old0,old1]:assert not row['passed']and not row['completed']and row['actual_CPU']=='910376.883'
    account=new['accounting'];assert len(account)==1 and account[0][1:4]==['GLM-r444-ready-socket-crossnode-v2-CPU','COMPLETED','0:0'],account
    step=account[0][0];assert step!='910376.883'
    nextsuffix='/candidates/GLM444-verified-ready-socket-crossnode-v2/'
    nodes=[json.loads(get(newfiles,nextsuffix+'node-%d.json'%i))for i in(0,1)]
    for role,row in enumerate(nodes):
        assert row['completed']and row['passed']and not row['errors']and row['actual_CPU']==step
        assert row['role']==role and row['node'].split('.')[0]==['a02r4n14','a02r4n15'][role]
        assert row['source_manifest_SHA256']==sha((v2/'SOURCE_MANIFEST.json').read_bytes())
        assert row['reused_CPU883_only']['local_cases']==18 and row['reused_CPU883_only']['actual_CLI_roles']==2
        assert row['reused_CPU883_only']['job_itself_FAILED']and not row['reused_CPU883_only']['new_crossnode_qualification']
        assert row['cases']==[] and row['recording_metadata_only'] and not row['GPU']and not row['Model']and not row['Native_dlopen']and not row['SDK']and not row['weights']
        assert not row['forbidden_modules_present']and not row['Native_SDK_maps']
        assert all(row['environment'][k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        expectedip=['12.1.2.74','12.1.2.75'][role]
        for field in ['fresh','post']:
            assert row[field]==[dict(IP=expectedip,port=p,bindable=True)for p in(43843,43844)]
        for name,item in row['actual_imports'].items():
            member='public_glm_native/'+name.split('.')[-1]+'.py'
            assert item['SHA256']==proof['files_SHA256'][member]
            assert '/incoming/GLM444-verified-ready-socket-CPU-v1/provider/'in item['path']
    assert nodes[0]['cross_node_Run_identity']==nodes[1]['cross_node_Run_identity']
    pub=[e for e in nodes[0]['events']if e['event']=='cross.original_hardlink.return']
    assert [e['name']for e in pub]==list(m1['original_ready_SHA256'])and len(pub)==16
    checked=[e for e in nodes[0]['events']if e['event']=='cross.all16_hardlink_and_original_validation.return'];assert len(checked)==1
    sent=[e for e in nodes[0]['events']if e['event']=='cross.same_TCP_receipt_ACK.return'];assert len(sent)==1
    assert max(e['monotonic']for e in pub)<=checked[0]['monotonic']<=sent[0]['monotonic']
    received=[e for e in nodes[1]['events']if e['event']=='cross.verified_after_all16_hardlink_receipt'];assert len(received)==1
    assert sent[0]['records_SHA256']==received[0]['records_SHA256']
    samples=nodes[1]['causally_post_publication_local_samples'];assert [r['name']for r in samples]==[r['name']for r in pub]
    hashes={v['name']:v['SHA256']for v in pub}
    for sample in samples:
        if sample['os_open']['ok']:assert sample['os_open']['result']['SHA256']==hashes[sample['name']]
    # Never subtract absolute clocks from different nodes. This is causal order,
    # not an attributed propagation latency or a replay of old879's wait state.
    return dict(passed=True,CPU_only=True,GPU_GO=False,Model_GO=False,source_manifest_SHA256=sha((v2/'SOURCE_MANIFEST.json').read_bytes()),
        private_provider_manifest_SHA256=m2['provider_manifest_SHA256'],source_members_verified=len(m1['files_SHA256'])+len(m2['files_SHA256']),
        old883=dict(accounting=old['accounting'][0],job_result='FAILED preserved',actual_socket_and_wiring_cases_passed=18,actual_CLI_roles_passed=2,complete_crossnode_receipts=0,originals=oldref),
        repaired_crossnode=dict(accounting=account[0],originals=newref,roles=['node14 sender','node15 receiver'],hardlinks=16,
            exact_verified_records_SHA256=received[0]['records_SHA256'],fresh_and_post_both_ports=[43843,43844],
            causally_post_publication_samples=16,exists_true=sum(s['Path_exists']['ok']and s['Path_exists']['result']for s in samples),
            stat_ok=sum(s['os_stat']['ok']for s in samples),open_ok=sum(s['os_open']['ok']for s in samples),
            sender_local_publish_to_ACK_seconds=sent[0]['monotonic']-pub[0]['monotonic']),
        scope=['Actual original16record validator, real socket frames/nonce/identity/ACK/EOF/FD closure.',
            'Actual Provider.start and Actors.wait_peer_node with Native owner/process/load leaves recording only.',
            'ACK/socket-close then STOP/owner-failed/exited prevents publication; old keeper late-READY latch retained.',
            'Causal after-all16-publication TCP receipt then local exists/stat/open; no cross-node clock subtraction.'],
        limitations=['Not original879 root cause or NFS propagation proof; no Native/device/model/quality qualification.',
            'Nonce provides freshness/correlation within fixed-peer controlled node trust domain, not cryptographic process authentication.',
            'Original keeper first16ready+120 remains final client authority; stage0+120 is a distinct additional guard.',
            'Instantaneous FS STOP checks do not eliminate cross-process races; no automatic GPU or other7 reload.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--observed883',type=Path,required=True);p.add_argument('--observed-v2',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=read(a.source_root,a.observed883,a.observed_v2)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=result['passed'],CPU=result['repaired_crossnode']['accounting'],raw_files=result['old883']['originals']['files']+result['repaired_crossnode']['originals']['files'])))


if __name__=='__main__':main()
