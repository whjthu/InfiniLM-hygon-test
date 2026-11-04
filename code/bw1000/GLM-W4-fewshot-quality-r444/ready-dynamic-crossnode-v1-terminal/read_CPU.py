"""Stdlib replay of existing CPU882 records, no source/module execution."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE='edcce2a3eb9b82704c01ad324d67f55c9b9661a67d63dff1b19ba60b62279270'
PRODUCER='921f0ccd37c741e993fc20c29bece4cff3ed932a70de44065165f7fa6994a0ea'
CONSUMER='3965c4c9536d2f55cb9ebb3da0270be57d924ff9bcacdc8bd3e3f49ad23d270d'
sha=lambda b:hashlib.sha256(b).hexdigest()


def audit(observed,source):
    observation_raw=Path(observed).read_bytes();q=json.loads(observation_raw)
    assert q['accounting']==['910376.882','GLM-r444-ready-dynamic-crossnode-v1-CPU','COMPLETED','0:0','00:00:31','a02r4n[14-15]']
    assert q['GPU']==0 and not q['new_job']and len(q['files'])==21
    files={}
    for n,v in q['files'].items():
        raw=v['rawtext'].encode();assert len(raw)==v['bytes']and sha(raw)==v['SHA256'],n
        files[n]=raw
    assert sha(files['node-0.json'])==PRODUCER and sha(files['node-1.json'])==CONSUMER
    producer=json.loads(files['node-0.json']);consumer=json.loads(files['node-1.json'])
    for r in [producer,consumer]:
        assert r['passed']and r['completed']and not r['errors']and r['actual_CPU']=='910376.882'
        assert r['recording_metadata_only']and r['GPU']==0 and not r['Model']and not r['Native_dlopen']and not r['SDK']
        assert r['source_manifest_SHA256']==SOURCE and not r['Model_READY_created']and not r['historical_cause_or_repair_proven']
    assert(producer['role'],producer['node'])==('producer','a02r4n14')
    assert(consumer['role'],consumer['node'])==('consumer','a02r4n15')
    assert producer['published_recording_ready_files']==consumer['actual_all_ready_returned']==16
    assert producer['recording_Run_identity']==consumer['recording_Run_identity']
    p=producer['events'];c=consumer['events']
    pubs=[e for e in p if e['event']=='producer.original_hardlink_publish.return']
    returns=[e for e in c if e['event']=='consumer.actual_Run_wait.return']
    misses=[e for e in c if e['event']=='consumer.actual_Run_wait.line71_after_false']
    names=['ready-%d-%d.json'%(s,r)for s in range(2)for r in range(8)]
    assert[e['name']for e in pubs]==[e['name']for e in returns]==names
    assert all(e['returned_recording_only']for e in returns)
    handoff=next(e for e in p if e['event']=='producer.observed_consumer_miss')
    assert handoff['consumer_marker']['wait_name']=='ready-0-0.json'
    assert handoff['consumer_marker']['actual_CPU']=='910376.882'
    assert pubs[0]['monotonic']-handoff['monotonic']>=2
    assert all(e['exists_now']for e in pubs)
    assert consumer['actual_wait_poll_counts']=={'ready-0-0.json':296}
    assert all(e['name']=='ready-0-0.json'and e['Path_exists']=={'ok':True,'result':False}
               and not e['os_stat']['ok']and e['os_stat']['errno']==2
               and not e['os_open']['ok']and e['os_open']['errno']==2 for e in misses)
    for name in names:
        file=next(n for n in files if n.endswith('/'+name))
        row=json.loads(files[file]);assert row['recording_metadata_only']and row['not_a_current_loaded_owner']
        assert row['identity']==consumer['recording_Run_identity']
    source=Path(source);mraw=(source/'SOURCE_MANIFEST.json').read_bytes();assert sha(mraw)==SOURCE
    mf=json.loads(mraw)
    for n,h in mf['files_SHA256'].items():assert sha((source/n).read_bytes())==h,n
    for name in names:
        file=next(n for n in files if n.endswith('/'+name))
        assert json.loads(files[file])['original_ready_SHA256']==mf['original_ready_SHA256'][name]
    p0=p[0]['monotonic'];c0=c[0]['monotonic']
    return dict(passed=True,scope='CPU-only original wait with a deliberately delayed publisher; not propagation latency or Model startup repair',
        actual_CPU='910376.882',accounting=q['accounting'],source_manifest_SHA256=SOURCE,
        observation_SHA256=sha(observation_raw),raw_files=21,raw_bytes=sum(map(len,files.values())),
        producer_receipt_SHA256=PRODUCER,consumer_receipt_SHA256=CONSUMER,
        actual_original_Run_all_ready_records=16,producer='node14',consumer='node15',
        producer_local_s=dict(first_publication=pubs[0]['monotonic']-p0,last_publication=pubs[-1]['monotonic']-p0,
                              consumed_actual_first_miss_marker=handoff['monotonic']-p0),
        consumer_local_s=dict(first_ready_return=returns[0]['monotonic']-c0,
            all_ready_return=c[-1]['monotonic']-c0,last_ENOENT_observation=misses[-1]['monotonic']-c0),
        first_wait_name='ready-0-0.json',false_original_predicate_polls=296,
        cross_node_absolute_clock_subtraction=False,
        causal_relation='Producer observed exact consumer firstfalse marker, then delayed2s before anypublication. This one-way marker does not bound producer startup or when it first saw that marker. Producer-local2..4s and consumer-local30s cannot establish publication-to-visibility delay.',
        post_publication_consumer_sample_causally_proven=False,
        producer_startup_or_marker_visibility_delay='UNKNOWN; may contribute to consumer30s wait',
        direct_open_worked_while_exists_false=False,stat_and_open_both_ENOENT_during_misses=True,
        observation_may_refresh_metadata_cache=True,
        original879_wait_name='UNKNOWN',original879_long_stall_reproduced=False,
        exact_filesystem_cache_cause='UNKNOWN;881mountinfo identifiesNFSv3 but exact failure mechanism/config is not proved',
        production_protocol_changed=False,Model_READY_created=False,Native=False,GPU=0,
        decision='STOP: preserve actual30s wait and local publication sequences without attributing30s toNFS propagation. No exists-to-open-only repair proof and no automatic GPU replay.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observed',required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.observed,a.source_root);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
