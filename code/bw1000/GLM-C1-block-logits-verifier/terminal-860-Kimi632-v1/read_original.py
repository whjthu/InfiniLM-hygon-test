"""Offline original-artifact repair; no producer imports, filesystem extraction or jobs."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
ARCHIVE_SHA='8f9eb998ba7a666639ad3b0dd9fbf8d8226c0d5987f1dbd151ed1cd2a16e1192'
RUN='prefix-logit-retention-20261105-a'
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
CONTAINER_ROOT=PurePosixPath('/refactor/results/GLM952-prefix-continuation')/RUN
def main(archive):
    assert hashlib.sha256(Path(archive).read_bytes()).hexdigest()==ARCHIVE_SHA
    tar=tarfile.open(archive);m=json.load(tar.extractfile('MANIFEST.json'));assert len(m['files'])==335
    def read(name):
        raw=tar.extractfile(name).read();pin=m['files'][name]
        assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['SHA256'],name
        return raw
    def j(name):return json.loads(read(name))
    for name in m['files']:read(name)
    result=dict(archive_SHA256=ARCHIVE_SHA,verified_members=335,new_jobs=0,producer_sources_modified=False)
    kw=j('kimi/watch.json');gw=j('glm/watch.json')
    kimi=dict(original_error=kw['error'],traceback=kw['traceback'],Model_READY_observed=False,full_Model_fit_proven=False,normal_destroy_proven=False,planned_questions=256,completed_requests_observed=0,missing=256,nodes=[],post=kw['failure_post'])
    for node in range(4):
        value=j('kimi/node%d.json'%node);progress=[e for e in value['events']if e['event']=='load_progress']
        assert not value.get('Model_READY')and value['fullModel_forward']==value['destroy_calls']==0 and not value.get('error')
        kimi['nodes'].append(dict(node=node,original_report_completed=value['completed'],last_load_progress=progress[-1:]if progress else [],request_files=[name for name in m['files']if name.startswith('kimi/node%d-case-'%node)]))
    for post in kimi['post']:assert not post['kfd_pids']and not post['matching']and not post['expected_PIDs_still_exist']and all(post['ports_bindable'].values())
    result['kimi']=kimi
    def difference(a,b):
        a=(a.astype(np.uint32)<<16).view(np.float32);b=(b.astype(np.uint32)<<16).view(np.float32)
        assert np.isfinite(a).all()and np.isfinite(b).all()
        delta=b.astype(np.float64)-a.astype(np.float64);rmse=float(np.sqrt(np.mean(delta**2)));rms=float(np.sqrt(np.mean(a.astype(np.float64)**2)))
        return dict(exact=bool(np.array_equal(a,b)),same_top1=int(np.argmax(a))==int(np.argmax(b)),maxabs=float(np.abs(delta).max()),RMSE=rmse,reference_RMS=rms,NRMS=rmse/rms if rms else None)
    glm=dict(original_keeper_error=gw['error'],original_keeper_passed=gw['passed'],traceback=gw['traceback'],normal16=True,raw_files=0,raw_bytes=0,rows=[],numeric_qualified=False,performance_qualified=False)
    for stage in range(2):
        stage_done=j('glm/block-stage-terminal-%d.json'%stage);assert stage_done['normal']and stage_done['all16_closed']and stage_done['local_processes_exited']==8
        for rank in range(8):
            value=j('glm/prefix-continuation-%d-%d.json'%(stage,rank));life=j('glm/lifecycle-%d-%d.json'%(stage,rank));closed=j('glm/closed-%d-%d.json'%(stage,rank))
            assert value['normal_shutdown']and life['normal']and life['completed']and life['destroy_attempts']==life['shutdown_attempts']==1
            assert life['destroy_RC']==life['shutdown_RC']==life['resident_reset_RC']==0 and not life['cleanup_errors']and closed['normal']
            assert len(value['cases'])==1;case=value['cases'][0];assert(case['geometry'],case['prefix_tokens'],case['width'],case['accepted'])==('above',2949,8,8)
            if stage==0:assert not case['retained_artifacts'];continue
            words={}
            for artifact in case['retained_artifacts']:
                label=artifact['label'];name='retained-1-%d-%s.bf16'%(rank,label);actual=PurePosixPath(artifact['path'])
                assert actual==CONTAINER_ROOT/name # exact container-root mapping, not basename forgiveness
                member='glm/'+name;assert m['files'][member]['path']==R+str(actual)[len('/refactor'):]
                side=j('glm/'+name[:-5]+'.json');assert side['identity']==value['identity']
                raw=read(member);assert len(raw)==artifact['bytes']and hashlib.sha256(raw).hexdigest()==artifact['SHA256']
                assert artifact['dtype']=='BF16'and artifact['endianness']=='little'
                width=8 if label=='block-proposal'else 1;assert artifact['shape']==[width,154880]
                words[label]=np.frombuffer(raw,'<u2').reshape(width,154880);glm['raw_files']+=1;glm['raw_bytes']+=len(raw)
            assert len(words)==13
            proposal=[difference(words['serial-proposal-%d'%i][0],words['block-proposal'][i])for i in range(8)]
            continuation=[difference(words['serial-continuation-%d'%i][0],words['candidate-continuation-%d'%i][0])for i in range(2)]
            glm['rows'].append(dict(rank=rank,proposal=proposal,continuation=continuation))
    assert glm['raw_files']==104 and glm['raw_bytes']==49561600
    glm['post_idle_records']=[v for v in gw['CPU_idle_observations']if v['phase']!='before']
    assert glm['post_idle_records']and all(v['rc']==0 for v in glm['post_idle_records'])
    glm['diagnosis']='proposal/block computation already diverges'if any(not row['exact']for owner in glm['rows']for row in owner['proposal'])else'proposal rows exact; continuation/cache branch requires investigation'
    glm['dynamic_per_slot_readback']='UNKNOWN; no new getter or inferred raw lengths'
    result.update(glm=glm,offline_artifact_audit_passed=True,old_keeper_failures_preserved=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();result=main(a.archive);Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(passed=True,verified_members=335,raw_files=result['glm']['raw_files'],diagnosis=result['glm']['diagnosis'],rank0=result['glm']['rows'][0])))
