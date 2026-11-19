"""Independent new probe metadata + one fresh sacct query; never reads the1MiB body."""
import argparse,hashlib,json,re,signal,stat,subprocess,time
from common import D,P,RUN,need,load,save,budget,sha
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def main(path):
    context=load(path);epoch=context['epoch'];left=budget(epoch,115);signal.signal(signal.SIGALRM,lambda *_:__import__('os')._exit(124));signal.setitimer(signal.ITIMER_REAL,left())
    result=dict(independent_terminal_metadata_verified=False,range_HTTP_prefix_qualified=False,pins={},errors=[],asset_body_reads=0,old362_queries=0,A100_calls=0,fresh_sacct_calls=0,whole_SHA_verified=False,Model_GO=False,GPU_GO=False,training_GO=False,replay_allowed=False)
    def take(name):
        p=RUN/name;need(p.is_file() and not p.is_symlink() and p.stat().st_size<=262144,'SMALL_REGULAR_NEW_METADATA');a=p.stat();raw=p.read_bytes();b=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'METADATA_STABLE');result['pins'][name]=dict(bytes=len(raw),SHA256=sha(raw));return json.loads(raw)
    try:
        need(load(D/'CLAIM.json')['READ_GO'] is True,'ROOT_READER_ADOPTION')
        need(sha((D/'SOURCE_FILES.json').read_bytes())==context['source_manifest_SHA256'],'PUBLISHED_SOURCE_MF')
        files={name:take(name) for name in ['install-intent.json','control-state.json','step-intent.json','outcome.json','worker/intent.json','worker/outcome.json']}
        for q in files.values():need(all(q[k]==context[k] for k in ['request_id','publication','epoch','deadline','source_manifest_SHA256']),'EXACT_ORIGINAL_CONTEXT')
        producer=files['outcome.json'];node=files['worker/outcome.json'];step=files['step-intent.json'];held=producer['held_srun'];row=producer['accounting']
        need(files['control-state.json']['exec_request_attempted'] is True and 30<=step['Slurm_seconds']<=90 and step['step_end']<=epoch+95 and step['work_deadline']<=epoch+80,'ACTUAL_BOUNDED_STEP')
        need(row and re.fullmatch(P['parent']+r'\.\d+',row[0]) and row[1]==P['label'] and node['step']==row[0] and row[5]==P['node'] and set(row[6].split(','))=={'cpu=4','mem=2G','node=1'},'COMPUTE_RESOURCE_IDENTITY')
        need(node['owner']['UID']==154459 and node['owner']['host']==P['node'] and producer['node']==node,'ACTUAL_COMPUTE_OBSERVER')
        need(held['reaped'] is True and held['returncode'] is not None,'OWNED_SRUN_REAPED')
        need(left()>1,'FRESH_ACCOUNTING_REMAINING_BUDGET')
        result['fresh_sacct_calls']=1
        q=subprocess.run(['sacct','-n','-P','-j',row[0],'-o',COLS],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=min(3,left()))
        need(q.returncode==0 and len(q.stdout)+len(q.stderr)<=1048576,'BOUNDED_FRESH_SACCT')
        rows=[v for line in q.stdout.splitlines() if len(v:=line.split('|'))==7 and v[0]==row[0] and v[1]==P['label']]
        need(len(rows)==1 and rows[0]==row and row[2].split()[0] in TERMINAL,'FRESH_EXACT_NEW_STEP_TERMINAL')
        need(node['whole_SHA_verified'] is False and node['full_weight_file_verified'] is False and 0<=node['explicit_body_consumed']<=1048576 and 0<=node['body_written']<=1048576 and node['body_consumed_before_accepted_headers']==0,'BOUNDED_BODY_NOT_WHOLE_MODEL')
        need(node['HTTP_attempts']==len(node['hops'])<=4 and node['redirects']<=3 and sum(h['explicit_body_consumed'] for h in node['hops'])==node['explicit_body_consumed'],'HTTP_HOPS_DENOMINATOR')
        need(all(h['hostname'] in P['allowed_HTTPS_hosts'] and re.fullmatch('[0-9a-f]{64}',h['URL_SHA256']) for h in node['hops']),'EXACT_HOST_HASH_LOGS')
        need(node['physical_socket_prefetch']=='UNKNOWN_NOT_MEASURED' and node['wire_bytes']=='UNKNOWN_NOT_MEASURED' and node['signed_URLs_saved'] is False,'NO_WIRE_OR_SIGNED_URL_CLAIM')
        if node['passed'] is True:
            need(node['finished_epoch']<=min(epoch+80,step['work_deadline']),'ORIGINAL_WORKER_BUDGET')
            need(node['HTTP_status']==206 and node['Content_Range']=='bytes 0-1048575/4915962496' and node['Content_Length']==1048576 and node['explicit_body_consumed']==node['body_written']==1048576 and re.fullmatch('[0-9a-f]{64}',node['observed_prefix_SHA256']),'ACTUAL_PREFIX_HTTP_IDENTITY')
            need(len(node['CPU_affinity'])==node['worker_CPU']==2 and node['allocated_CPU']==4 and not set(node['CPU_affinity'])&{0,1} and node['AS_limit_bytes']==2147483648,'ALLOC4_WORKER2_MEM2G')
            need(stat.S_ISREG(node['prefix_stat']['st_mode']) and node['prefix_stat']['st_size']==1048576 and node['prefix_stat']['st_uid']==154459,'SAVED_PREFIX_STABLE_FD_METADATA')
            need(all(h['explicit_body_consumed']==0 for h in node['hops'][:-1]),'NO_REDIRECT_BODY_CONSUMED')
        qualified=bool(node['passed'] is True and producer['passed'] is True and held['termination_needed'] is False and held['returncode']==0 and row[2:4]==['COMPLETED','0:0'])
        result.update(independent_terminal_metadata_verified=True,range_HTTP_prefix_qualified=qualified,accounting_columns=COLS,fresh_accounting=row,producer_HTTP_pass=node['passed'],explicit_body_consumed=node['explicit_body_consumed'],prefix_SHA256=node['observed_prefix_SHA256'],proof_boundary='Fresh exact new-step accounting plus frozen worker metadata; no independent body rehash, no full-shard SHA qualification')
    except BaseException as e:result['errors'].append(dict(type=type(e).__name__,code=str(e) if type(e)is ValueError else 'SANITIZED_METADATA_ERROR'))
    save(RUN/'independent-result.json',result);signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({k:result[k] for k in ['independent_terminal_metadata_verified','range_HTTP_prefix_qualified','fresh_sacct_calls','asset_body_reads','errors']}));return 0 if result['independent_terminal_metadata_verified'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);raise SystemExit(main(p.parse_args().context))
