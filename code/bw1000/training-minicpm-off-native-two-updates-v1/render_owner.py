"""Derive the one-shot host entry from the pinned MiniCPM native owner."""
import re

from render_worker import change, check_scope
from scope_contract import INPUT_ROOT, need, sha


def render_owner(original, scope, stage_sha):
    check_scope(scope)
    need(sha(original) == "11e6629bd3c4d1e8e3158ef2a188b80f296178a6bf63abe9169191cc9a875ba3"
         and re.fullmatch(r"[0-9a-f]{64}", stage_sha), "OFF_ORIGINAL_OWNER_OR_STAGE")
    source = original.decode()
    a = source.index("from prepare_minicpm_usgs_pair_target_v3 import")
    b = source.index("def pin(", a)
    source = source[:a] + f'''REPO = Path({str(INPUT_ROOT).split('/incoming/')[0]!r})
RESULTS = REPO/'results'
sys.path.insert(0,str(REPO/'tools'))
from prepare_minicpm_usgs_pair_target_v3 import CORE_ASSETS_SHA, CORE_SHA, need, small, write
from read_training_slurm_membership import allocated_dcu

LABEL, REQUEST = {scope['label']!r}, {scope['request']!r}
PARENT, NODE, PORT = {scope['parent']!r}, {scope['node']!r}, {scope['port']}
RUN = RESULTS/LABEL
HOST_PYTHON = '/usr/local/python3.12/bin/python3'
ACCOUNTING_HELPER = REPO/'code/bw1000/training-qwen-native25-hip-v3'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
PHASES = {{'pre': (0, 2, 2048, 60), 'gpu': (8, 64, 224000, 2100), 'post': (0, 2, 2048, 90)}}
WHOLE, NATIVE, WORKER_CLEANUP = 2400, 1980, 2080
REQUEST_PATTERN = re.compile(PARENT+r'\\.\\d+')
STAGE_SHA = {stage_sha!r}


''' + source[b:]
    a = source.index("def accepted_stage(")
    b = source.index("def worker_config(", a)
    source = source[:a] + '''def accepted_stage(root, expected):
    raw=small(root/'asset-stage-complete.json',expected)
    complete=json.loads(raw)
    need(complete['request']==REQUEST and complete['stage']['path']==str(RUN/'asset-stage-record.json')
         and complete['stage']['SHA256']==STAGE_SHA,'OFF_OWNER_STAGE_COMPLETION')
    stage_raw=small(RUN/'asset-stage-record.json',STAGE_SHA)
    stage=json.loads(stage_raw)
    need(stage['request']==REQUEST and (stage['parent'],stage['node'],stage['port'])==(PARENT,NODE,PORT)
         and stage['scope_SHA256']==complete['scope_SHA256'] and stage['source_rows']==6 and stage['train_media']==4
         and stage['actual_optimizer_updates']==stage['actual_GPU_steps']==0 and stage['state']=='OFF_SIX_TO_800_STAGED','OFF_OWNER_STAGE_SCOPE')
    own=Path(__file__).resolve()
    need(own==RUN/'owner.py' and pin(own,small(own,complete['owner']['SHA256']))==complete['owner'],'OFF_OWNER_BYTES')
    need(set(complete['target'])=={'worker.py','owned_processes.py','snapshot.py'},'OFF_OWNER_STAGE_WORKER_SET')
    for name, record in complete['target'].items():
        p=RUN/'target-worker'/name
        need(pin(p,small(p,record['SHA256']))==record,'OFF_OWNER_STAGE_WORKER_BYTES')
    return dict(asset_stage=pin(RUN/'asset-stage-record.json',stage_raw))


''' + source[b:]
    source = change(source, "MINICPM_USGS_TARGET_WORKER_CONFIG_V1", "MINICPM_OFF_TARGET_WORKER_CONFIG_V1")
    source = change(source, "'-S', '2026-11-19T00:00:00', '-o', COLUMNS]", "'-o', COLUMNS]")
    source = change(source, "text=True, timeout=8)", "text=True, timeout=8, env=dict(os.environ,TZ='UTC'))")
    source = source.replace("len(row) == 7", "len(row) == 9")
    source = change(source, "allocated_dcu(row[6]) == PHASES[phase][0])",
        """allocated_dcu(row[6]) == PHASES[phase][0]
             and exact_tres(row[6],phase) and row[4].isdigit() and int(row[4]) <= 60*int(argv[argv.index('-t')+1])
             and row[7] <= row[8] and row[7] not in ('','Unknown') and row[8] not in ('','Unknown'))""")
    source = source.replace("('continuous', 'resumed')", "('continuous',)")
    source = change(source, "(RESULTS / 'training-domain-usgs-table-reading-v1/data/train', None, 'ro')",
                    f"(Path({INPUT_ROOT!r}), None, 'ro')")
    a = source.index("def original_membership(")
    b = source.index("def controller(", a)
    source = source[:a] + '''def exact_tres(raw,phase):
    entries=raw.split(',')
    tres=dict(x.split('=',1) for x in entries)
    devices,cpus,mem,_=PHASES[phase]
    mems={str(mem)+'M'}
    if mem%1024==0:mems.add(str(mem//1024)+'G')
    expected={'cpu','mem','node'}|({'gres/dcu'} if devices else set())
    return len(entries)==len(tres) and set(tres)==expected and tres['cpu']==str(cpus) and tres['node']=='1' and tres['mem'] in mems and (not devices or tres['gres/dcu']==str(devices))


def original_membership():
    def query(*args):
        return subprocess.check_output(args,text=True,timeout=8)
    def fields(text):
        return dict(x.split('=',1) for x in text.split() if '=' in x)
    def names(value):
        return query('scontrol','show','hostnames',value).splitlines()
    first=query('scontrol','show','job',PARENT,'-o').splitlines()
    need(len(first)==1,'OFF_PARENT_CONTROLLER_AMBIGUOUS')
    job=fields(first[0])
    alternate=query('squeue','-h','-j',PARENT,'-o','%i|%T|%N').splitlines()
    need(len(alternate)==1,'OFF_PARENT_QUEUE_AMBIGUOUS')
    qid,qstate,qnodes=alternate[0].split('|')
    need(job['JobId']==qid==PARENT and job['JobState']==qstate=='RUNNING'
         and job['UserId'].endswith(f'({os.getuid()})') and NODE in names(job['NodeList']) and NODE in names(qnodes),
         'OFF_PARENT_DUAL')
    for line in query('scontrol','show','step',PARENT,'-o').splitlines():
        row=fields(line); step=row['StepId']
        if step in (PARENT+'.batch',PARENT+'.extern'):continue
        need(REQUEST_PATTERN.fullmatch(step) and NODE not in names(row['NodeList']),'OFF_NODE_CONTROLLER_OCCUPIED')
    for line in query('squeue','--steps','-h','-j',PARENT,'-o','%i|%N').splitlines():
        step,nodes=line.split('|')
        if step in (PARENT+'.batch',PARENT+'.extern'):continue
        need(REQUEST_PATTERN.fullmatch(step) and NODE not in names(nodes),'OFF_NODE_QUEUE_OCCUPIED')


def accept_authority(path, expected, stage_sha, source_sha, publication):
    raw=small(path,expected)
    grant=json.loads(raw)
    now=time.time()
    need(grant['request']==REQUEST and (grant['parent'],grant['node'],grant['port'])==(PARENT,NODE,PORT)
         and grant['publication']==publication and grant['asset_stage_complete_SHA256']==stage_sha
         and grant['source_manifest_SHA256']==source_sha and grant['root_GO'] is True and grant['GPU_GO'] is True
         and grant['training_GO'] is True and grant['ordinary_excluded'] is True and grant['no_prior_numbered_holder'] is True
         and grant['observed_epoch']<=now<grant['expires_epoch']<=grant['observed_epoch']+65,'OFF_FRESH_OWNER_AUTHORITY')
    scope=json.loads(small(RUN/'off-scope.json',grant['scope_SHA256']))
    need(scope['request']==REQUEST and scope['label']==LABEL and scope['publication']==publication
         and scope['source_manifest_SHA256']==source_sha and scope['GPU_GO'] is False and scope['training_GO'] is False,
         'OFF_OWNER_STAGED_CLAIM')
    return grant


''' + source[b:]
    source = change(source, "def controller(stage_sha, source_sha, publication):",
                    "def controller(stage_sha, source_sha, publication, authority, authority_sha):")
    source = change(source, "    worker = stage_worker(source_sha)",
                    "    grant=accept_authority(authority,authority_sha,stage_sha,source_sha,publication)")
    source = change(source, "asset_stage_SHA256=stage_sha, source_manifest_SHA256=source_sha, node=NODE",
                    "asset_stage_SHA256=stage_sha, source_manifest_SHA256=source_sha, authority_SHA256=authority_sha, planned_native_forwards=64, planned_updates=2, node=NODE")
    a = source.index("        folder = RUN / 'target-worker'")
    b = source.index("        snapshot =", a)
    source = source[:a] + source[b:]
    source = change(source, "            phases['gpu'] = execute_step('gpu', gpu_command(cfg, epoch), elapsed, epoch)",
                    "            original_membership()\n            need(time.time()<grant['expires_epoch'],'OFF_OWNER_AUTHORITY_EXPIRED_AFTER_PRE')\n            phases['gpu'] = execute_step('gpu', gpu_command(cfg, epoch), elapsed, epoch)")
    source = change(source, "recovery = json.loads(small(RUN / 'paired-recovery.json'))",
                    "recovery = json.loads(small(RUN / 'off-first-two.json'))")
    source = change(source, "recovery['exact_paired_recovery'] is True",
                    "recovery['same_update_on_all_ranks'] is True and recovery['native_forwards']==64 and recovery['actual_optimizer_updates']==2")
    source = change(source, "result = controller(args.stage_sha, args.source_sha, args.publication)",
                    "result = controller(args.stage_sha, args.source_sha, args.publication, Path(args.authority), args.authority_sha)")
    source = change(source, "('stage-sha', 'source-sha', 'publication')", "('stage-sha', 'source-sha', 'publication','authority','authority-sha')")
    compile(source, "owner.py", "exec")
    return source.encode()
