"""Guarded single-node BF16 continuation for the first three exact ordinary incumbents."""

import argparse
import ast
from contextlib import closing
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import time
from types import FunctionType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r25-early-ordinary-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r25_early_ordinary_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
BASE = ('tools/run_training80_prebridge_five_guarded_refill_once.py', '7b88d30a4af7eff4f25fbcc663de674af328a03de74e364c517752bd599d306c')
BASE_CLAIM = ('coordination/tasks/training80-prebridge-five-guarded-bf16-v1.json', 'c4c18f644e6ebc2a99ea0e19d2261a9cbbe47efe743e79dd5ae8f79fea4ce974')
ORDINARY = ('tools/run_training80_ordinary58_1140_refill_once.py', '9cdf04cf946058c837ff9b5101f9eff56e969c595ae3b2397fe064834f2c4ba2')
R25 = ('local-only/read-training80-r25-next-incumbents-once.py', '0462a5247bb78a0b4f61095d3be8bf1f66c06d73339043654dbc135f68f4a789')
ADOPTION = '41c04c4692c8d57232ec9f0b4bcdd83a5754254917742d745a5b47010868e994'
OBSERVERS = {'1103606':'1103606.3088', '1194900':'1194900.1182'}
HOUR = '0e1a294fabfc4854854c09ed6b4d6f83'
GPU_TRES = 'cpu=8,gres/dcu=8,mem=16G,node=1'
SPEC = {
    'f13r2n08': ('ordinary2', '1103606.3218', '1103606.3098', 'cef01e3f963c5e6a8641aeae36a5a67b', '4e7b5ccb787e4660867907a6694f56a1'),
    'f14r3n04': ('ordinary1', '1194900.1256', '1194900.1186', '8ad9a60638735b67888ed9581f789548', '0752937265d54324b57daae89e88c7b6'),
    'f13r3n04': ('ordinary1', '1194900.1257', '1194900.1185', '8ad9a60638735b67888ed9581f789548', '1f9e9e29f570481ab99c29214b0825ee'),
}
UTC = "import os\nos.environ.update(TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')\n"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, sha=None, limit=262144):
    require(path.resolve(strict=True) == path and path.is_relative_to(ROOT), 'exact configured file differs')
    with path.open('rb') as handle:
        raw = handle.read(limit+1)
    require(0 < len(raw) <= limit and (sha is None or digest(raw) == sha), 'pinned receipt or source bytes differ')
    return raw


def toolbox(spec):
    name, expected = spec
    path = ROOT / name
    pin(path,expected)
    return runpy.run_path(str(path))


def unique(text, old, new):
    require(text.count(old) == 1, 'frozen guard insertion point differs')
    return text.replace(old,new,1)


def load(draft=False):
    base = toolbox(BASE)
    config = base['prepared'](json.loads(pin(ROOT / BASE_CLAIM[0], BASE_CLAIM[1])))
    ordinary = toolbox(ORDINARY)
    old_material = ordinary['plan']()
    read = toolbox(R25)
    raw = pin(DRAFT if draft and not CLAIM.exists() else CLAIM)
    claim = json.loads(raw)
    require((claim['scope'], claim['state'], claim['launcher'], claim['scope_mode'], claim['original_adoption_SHA256'],
             claim['max_invocations_per_node'], claim['GPU_per_target'], claim['ledger_mutations'], claim['replay_allowed'])
            == (STEM,'PREPARED_ONLY_ACTUAL_TERMINAL_REQUIRED',WRAPPER.relative_to(ROOT).as_posix(),
                'ONE_NODE_EXACT_ORDINARY_BRIDGE_STANDBY',ADOPTION,1,8,0,False)
            and set(claim['targets']) == set(SPEC), 'fixed independent singleton claim differs')
    rows = {}
    for node,(selector,step,ancestor,original_claim,request) in SPEC.items():
        [row] = [item for item in read['rows_for'](selector) if item['node']==node]
        source = read['G1' if selector=='ordinary1' else 'G2']
        progress = json.loads(pin(ROOT / source[0],source[1]))['response']['nodes'][node]
        i = int(selector[-1])-1
        require(row['previous']==step and row['cgroup_SHA256'] is None and progress['growth_DCU']==8
                and progress['submit_SHA256'] and re.fullmatch(r'[a-f0-9]{64}',progress['submit_SHA256'])
                and old_material[4]['request_ids'][i]==original_claim
                and any(old['node']==node and old['previous']==ancestor for old in old_material[-1][i]),
                'original ordinary group or worker identity differs')
        prepare_epoch = datetime.fromisoformat(row['prep_Beijing']).timestamp()
        require(abs(prepare_epoch + 900 - row['planned_end_epoch']) < 0.000002, 'ordinary preparation epoch differs')
        row.update(prepare_epoch=prepare_epoch, submit_SHA256=progress['submit_SHA256'], ledger_old=ancestor, ledger_request=original_claim,
                   new_request=request, published_group=i+1)
        require(claim['targets'][node] == dict(parent=row['parent'], previous_step=step, prior_standby_request=row['request'],
                    original_group_request=original_claim, original_group=i+1, prior_device_mode='ORDINARY_BRIDGE_BF16',
                    prior_submit_SHA256=row['submit_SHA256'], request=request), 'claim changed the original ordinary scope')
        rows[node] = row
    return base,config,ordinary,old_material,read,rows,digest(raw)


def exact_no_other_claims(ordinary, material, row):
    ledger, identity, _ = ordinary['custody'](material)
    ordinary['check_authority'](material,ledger,identity,ADOPTION)
    previous = material[0]
    previous['local_census']()
    node,parent = row['node'],row['parent']
    with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True,timeout=5)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        rows = db.execute('SELECT t.parent,t.predecessor,t.request,c.wave,c.group_id FROM targets t JOIN claims c ON c.request=t.request WHERE t.node=?',(node,)).fetchall()
        request = row['ledger_request']
        events = db.execute('SELECT state,detail FROM events WHERE request=? ORDER BY sequence',(request,)).fetchall()
        active_exact = (parent,row['ledger_old'],request,ordinary['WAVE'],row['published_group'])
        require(active_exact in rows and [state for state,_ in events] == ['CLAIMED','BRIDGE_INTENT','BRIDGE_ACKNOWLEDGED']
                and all(json.loads(raw)['bridge_request']==row['request'] for _,raw in events[1:]),
                'original ordinary exact authenticated bridge acknowledgement differs')
        require(all(item[0]==parent and item[1]!=row['previous'] and (item==active_exact or item[3]==previous['WAVE']) for item in rows)
                and all(req != row['new_request'] for _,_,req,_,_ in rows), 'target has a new or unresolved ordinary booking')
    return rows


def slot_paths(row):
    prefix = LOCAL / f"{STEM}-{row['node']}-{row['new_request']}"
    return tuple(prefix.with_name(f'{prefix.name}-{kind}.json') for kind in ('intent','send','reply'))


def read_path(row, nonce):
    require(isinstance(nonce,str) and re.fullmatch(r'[a-f0-9]{32}',nonce) and int(nonce,16)>0 and nonce!=row['new_request'], 'new exact read identifier required')
    return LOCAL / f"{STEM}-{row['node']}-read-{nonce}.json"


def read_program(context, row, nonce):
    read = context[4]
    source = UTC + read['render']([row],nonce)
    # This admission reader checks disk-owner ID; it never treats a shared admin file as the original worker.
    source = unique(source,'    assert path.resolve(strict=True)==path\n',
                    '    assert path.resolve(strict=True)==path and path.stat().st_uid==os.getuid()==154459\n')
    compile(source,'<ordinary singleton predecessor read>','exec')
    return source


def observe_predecessor(context,row,nonce):
    source = read_program(context,row,nonce).encode()
    require(time.time() >= row['prepare_epoch'], 'ordinary preparation time has not opened')
    config = context[1]
    support = config[0]
    helper = runpy.run_path(str(support['HELPER']))
    ssh = shlex.join(['/usr/local/python3.12/bin/python3','-I','-S','-B','-'])
    path = read_path(row,nonce)
    with path.open('x',encoding='utf8') as saved:
        t0 = time.monotonic()
        anchor = time.time()-t0
        try:
            reply = subprocess.run([*helper['ssh'](ROOT.parent),ssh],input=source,capture_output=True,timeout=45)
            rc,stdout,stderr=reply.returncode,reply.stdout,reply.stderr
        except subprocess.TimeoutExpired as error:
            rc,stdout,stderr=None,error.stdout or b'',error.stderr or b''
        except (OSError, ValueError, AssertionError, StopIteration, subprocess.SubprocessError) as error:
            rc,stdout,stderr=None,b'',str(error).encode()
        t1=time.monotonic()
        observed = None
        if rc == 0 and 0 < len(stdout) <= 8192 and t1-t0<=45:
            try:
                body=json.loads(stdout)
                [item]=body['targets']
                if body['read_id']==nonce and body['Slurm_mutations']==body['physical_admissions']==0 and (item['node'],item['step'],item['request'])==(row['node'],row['previous'],row['request']):
                    observed=body
            except (ValueError, KeyError, TypeError):
                pass
        json.dump(dict(rc=rc, observed=observed, source_SHA256=digest(source), received_monotonic=t1, anchor=anchor,
                       RTT=t1-t0,stdout_SHA256=digest(stdout),stderr_SHA256=digest(stderr),replay_allowed=False),saved,sort_keys=True)
    return dict(path=path.name,SHA256=digest(pin(path)),predecessor=row['previous'],old_terminal_proven=(observed or {}).get('targets',[{}])[0].get('original_terminal_and_targets_empty',False), replay_allowed=False)


def terminal(context,row,nonce,proof,*,now,wall):
    require(proof['rc']==0 and proof['replay_allowed'] is False and 0 <= now-proof['received_monotonic'] <=45
            and abs(wall-now-proof['anchor'])<2 and 0<=proof['RTT']<=45
            and proof['source_SHA256']==digest(read_program(context,row,nonce).encode()), 'fresh exact predecessor receipt required')
    return terminal_identity(row,nonce,proof)


def terminal_identity(row,nonce,proof):
    observed=proof['observed']
    [item]=observed['targets']
    label=f"training80-dcu-standby-{row['request'][:12]}-{row['node']}"
    acct=item['accounting']
    require(observed['read_id']==nonce and observed['Slurm_mutations']==observed['physical_admissions']==0
            and (item['step'],item['node'],item['request'])==(row['previous'],row['node'],row['request'])
            and item['original_terminal_and_targets_empty'] is True and item['scontrol_target_steps']==item['squeue_target_steps']==[]
            and item['worker_SHA256']==row['worker_SHA256'] and item['expected_SHA256']==row['expected_SHA256']
            and re.fullmatch(r'[a-f0-9]{64}',item['terminal_SHA256'])
            and len(acct)==9 and acct[:4]==[row['previous'],label,'COMPLETED','0:0'] and acct[5:7]==[row['node'],GPU_TRES]
            and int(acct[4])>=row['worker_seconds'] and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',acct[8]),
            'same ordinary natural return, exact C0/End and two empty scheduler views required')
    return acct,item['terminal_SHA256']


def render(context,row,account,terminal_pin,publication):
    base,config,*_=context
    node,parent=row['node'],row['parent']
    # The base function's positional historical scout inputs are discarded below. The original ordinary path had a bridge, not a direct scout step.
    config[3][node]=(account,[None]*9,row['pci'])
    function=base['render']
    renderer=FunctionType(function.__code__,dict(function.__globals__,STEM=STEM,PARENT=parent))
    output=renderer(node,{'targets':{node:{'request':row['new_request']}}},config,publication)
    label=f'training80-long-busy-{HOUR}-{parent}'
    for key in ('launch','observe'):
        source=output[key]
        source=unique(source,f"OBSERVER_STEP={config[2]['observer']['step']!r}",f'OBSERVER_STEP={OBSERVERS[parent]!r}')
        source=unique(source,f"OBSERVER_LABEL={config[2]['observer']['label']!r}",f'OBSERVER_LABEL={label!r}')
        label_suffix = row['new_request'][:12]
        source=unique(source,f"LABEL={'training80-g4-fresh-'+label_suffix!r}",f"LABEL={'training80-r25-fresh-'+label_suffix!r}")
        output[key]=UTC+source
    source=output['launch']
    locked='            check_previous(account(PREVIOUS))'
    require(source.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)') < source.index(locked) < source.index('outcome["zero_GRES_call_attempted"] = True'),
            'original shared lock and independent fresh gate order changed')
    meta = {name: row[key] for name,key in (('expected.json','expected_SHA256'),('worker.json','worker_SHA256'),('submission-receipt.json','submit_SHA256'))}
    meta['terminal.json']=terminal_pin
    revalidation=(
        '            require(os.getuid() == 154459, "wrong original Slurm owner")\n'
        f"            original_dir = RESULTS / {'training80-dcu-standby-'+row['request']!r}\n"
        f'            original_records = {{}}\n            for member, expected_sha in {meta!r}.items():\n'
        "                proof_path = (original_dir if member == 'submission-receipt.json' else original_dir / NODE) / member\n"
        '                require(proof_path.resolve(strict=True) == proof_path and proof_path.stat().st_uid == os.getuid(), "old ordinary evidence owner or path differs")\n'
        "                with proof_path.open('rb') as old_handle:\n"
        '                    old_record_bytes = old_handle.read(16385)\n'
        '                require(0 < len(old_record_bytes) <= 16384 and sha256(old_record_bytes).hexdigest() == expected_sha, "old ordinary evidence changed")\n'
        '                original_records[member] = json.loads(old_record_bytes)\n'
        "            old_submit, old_worker, old_expected, old_done = (original_records[key] for key in ('submission-receipt.json','worker.json','expected.json','terminal.json'))\n"
        "            [assignment] = [record for record in old_submit['submit_records'] if record['node'] == NODE]\n"
        f"            require(old_submit['request'] == old_expected['request'] == {row['request']!r} and old_submit['synthetic'] is old_worker['synthetic'] is True\n"
        "                    and assignment['parent'] == old_expected['parent'] == PARENT and assignment['label'] == old_expected['label'] == PREVIOUS_LABEL\n"
        "                    and old_worker['step'] == old_done['step'] == PREVIOUS and old_worker['node'] == old_expected['node'] == old_done['node'] == NODE\n"
        f"                    and assignment['worker_seconds'] == old_expected['seconds'] == {row['worker_seconds']!r}\n"
        f"                    and old_worker['epoch'] == {row['worker_start_epoch']!r} and old_worker['worker_pid'] == {row['previous_worker_pid']!r}\n"
        "                    and sorted(old_worker['pci']) == sorted(old_expected['pci']) == PCI and old_done['state'] == 'WORKER_RETURNED'\n"
        "                    and old_done['epoch'] >= old_worker['epoch'] + old_expected['seconds'], 'original ordinary worker did not return naturally')\n"
    )
    source=unique(source,locked,revalidation+locked)
    # Ordinary ancestry is proved from the actual bridge receipt. No direct prior CPU clearance is fabricated.
    source=unique(source,'            check_first_clearance(account(CLEAR_STEP))\n','')
    source=unique(source,'original_clearance=CLEAR_STEP, new_clearance=step',
                  f"original_bridge_submit_SHA256={row['submit_SHA256']!r}, new_clearance=step")
    source=unique(source,'({OBSERVER_STEP, own} if own else {OBSERVER_STEP})','({own} if own else set())')
    source=unique(source,'            root.mkdir(mode=0o700)',
                  "            require(isinstance(raw, dict) and {k: sha256(v).hexdigest() for k,v in raw.items()} == PINS, 'worker source map differs')\n"
                  '            root.mkdir(mode=0o700)')
    output['launch']=source
    for key in ('launch','observe'):
        compile(output[key],f'<ordinary singleton {node} {key}>','exec')
    return output


def publication(context,revision):
    require(Path(__file__).resolve()==WRAPPER and re.fullmatch(r'[a-f0-9]{40}',revision),'exact installed singleton publication required')
    base,config,ordinary,old_material,*_=context
    base['verify_publication'](revision,config)
    _,publisher=old_material[-2]['modules']()
    ordinary['publication'](revision,old_material,publisher)
    for path in (WRAPPER,CLAIM):
        require(publisher['git']('cat-file','blob',revision+':'+path.relative_to(ROOT).as_posix())==pin(path),'published singleton source or claim differs')


def dispatch(context,row,revision,terminal_arg):
    base,config,ordinary,old_material,*_=context
    path,token,reply=slot_paths(row)
    require(not any(os.path.lexists(item) for item in (path,token,reply)),'fixed singleton original requires reconciliation')
    publication(context,revision)
    exact_no_other_claims(ordinary,old_material,row)
    require(row['node'] not in base['local_holds'](),'READY, reserved or unresolved priority has the node')
    nonce,sha=terminal_arg
    proof=json.loads(pin(read_path(row,nonce),sha))
    account,terminal_pin=terminal(context,row,nonce,proof,now=time.monotonic(),wall=time.time())
    rendered=render(context,row,account,terminal_pin,revision)
    helper=runpy.run_path(str(config[0]['HELPER']))
    sender=runpy.run_path(str(config[0]['TRANSPORT']))
    argv=helper['ssh'](ROOT.parent)
    require(0<=time.monotonic()-proof['received_monotonic']<=45,'old terminal read aged before single send reservation')
    data=dict(node=row['node'],request=row['new_request'],publication=revision,claim_SHA256=context[-1], terminal_read_id=nonce,
              terminal_read_SHA256=sha,program_SHA256=digest(rendered['launch'].encode()),GPU_requested=8,replay_allowed=False)
    config[1]['save'](path,data)
    sent=sender['send'](argv,request=row['new_request'],program=rendered['launch'],data=b'',send_token=token,authority_seconds=18,outcome_seconds=82)
    config[1]['save'](reply,dict(data,transport=sent))
    return dict(receipt=reply.name,SHA256=digest(pin(reply)),state=(sent.get('response') or {}).get('state'),replay_allowed=False)


def observe_original(context,row):
    base,config,*_=context
    intent_path,token,_=slot_paths(row)
    intent=json.loads(pin(intent_path))
    require(intent['request']==row['new_request'] and intent['node']==row['node'] and intent['replay_allowed'] is False
            and token.is_file(), 'only an exact previously sent singleton can be observed')
    nonce=intent['terminal_read_id']
    proof=json.loads(pin(read_path(row,nonce),intent['terminal_read_SHA256']))
    prior_account,terminal_pin=terminal_identity(row,nonce,proof)
    program=render(context,row,prior_account,terminal_pin,intent['publication'])
    require(digest(program['launch'].encode())==intent['program_SHA256'], 'original singleton program differs')
    observer=base['observe']
    bound=FunctionType(observer.__code__,dict(observer.__globals__,STEM=STEM,paths=lambda node,request:
                        slot_paths(row) if (node,request)==(row['node'],row['new_request']) else ()))
    return bound(row['node'],program,config)


def host_fixture(rendered,row,account,standby):
    module=ast.parse(rendered)
    guard=next(obj for obj in module.body if isinstance(obj,ast.FunctionDef) and obj.name=='run_guarded')
    locked=next(obj for obj in ast.walk(guard) if isinstance(obj,ast.With)
                and any(isinstance(item.optional_vars,ast.Name) and item.optional_vars.id=='lock' for item in obj.items))
    loop=next(obj for obj in locked.body if isinstance(obj,ast.For) and ast.unparse(obj.target)=='(member, expected_sha)')
    end=next(i for i,obj in enumerate(locked.body) if ast.unparse(obj)=='check_previous(account(PREVIOUS))')
    start=next(i for i,obj in enumerate(locked.body) if 'wrong original Slurm owner' in ast.unparse(obj))
    stage=next(i for i,obj in enumerate(locked.body) if ast.unparse(obj).startswith('root.mkdir('))
    assignments=locked.body[start:end+1]+locked.body[stage-1:stage+3]
    require('stream.write(raw[\'worker.py\'])' in ast.unparse(assignments[-1]),'ordinary actual staging block differs')
    write=next(obj for obj in ast.parse(standby['controller.py']).body if isinstance(obj,ast.FunctionDef) and obj.name=='write')
    writer=dict(os=os,json=json)
    exec(compile(ast.Module(body=[write],type_ignores=[]),'<frozen controller writer>','exec'),writer)
    node,step,parent=row['node'],row['previous'],row['parent']
    label=account[1]
    old_request=row['request']
    pins={name:digest(raw) for name,raw in standby.items()}
    owner_uid=154459
    original_stat=Path.stat

    def test_stat(path,*args,**kwargs):
        values=list(original_stat(path,*args,**kwargs))
        values[4]=owner_uid
        return os.stat_result(values)

    for mode in ('exact','early','owner'):
        with TemporaryDirectory() as tmp:
            results=Path(tmp).resolve()
            previous_root=results/('training80-dcu-standby-'+old_request)
            target=previous_root/node
            target.mkdir(parents=True)
            new=results/'new-synthetic-worker'
            assignment=dict(node=node,parent=parent,label=label,worker_seconds=row['worker_seconds'])
            old_expected=dict(node=node,parent=parent,label=label,request=old_request,seconds=row['worker_seconds'],pci=row['pci'])
            old_worker=dict(step=step,node=node,synthetic=True,worker_pid=row['previous_worker_pid'],epoch=row['worker_start_epoch'],pci=row['pci'])
            done=dict(step=step,node=node,state='WORKER_RETURNED',epoch=row['planned_end_epoch']+(-2 if mode=='early' else 1))
            files={'submission-receipt.json':dict(request=old_request,synthetic=True,submit_records=[assignment]),
                   'expected.json':old_expected,'worker.json':old_worker,'terminal.json':done}
            pinned={}
            for name,value in files.items():
                path=(previous_root if name=='submission-receipt.json' else target)/name
                data=json.dumps(value).encode()
                path.write_bytes(data)
                pinned[name]=digest(data)
            loop.iter.func.value=ast.parse(repr(pinned),mode='eval').body
            full_account=account[:4]+account[5:7]+account[8:]
            host=dict(RESULTS=results,os=SimpleNamespace(getuid=lambda: owner_uid if mode!='owner' else 0),PARENT=parent,NODE=node,
                      PREVIOUS=step,PREVIOUS_LABEL=label,sha256=sha256,json=json,require=require,PCI=row['pci'],PINS=pins,raw=dict(standby),
                      check_previous=lambda r: require(r==full_account,'old exact C0 changed'),account=lambda s: full_account if s==step else None,
                      root=new,controller={'write':writer['write']},time=time,REQUEST=row['new_request'],PUBLICATION=None,
                      step=parent+'.99999',outcome={'probe_stdout_SHA256':'0'*64})
            with patch.object(Path,'stat',test_stat):
                try:
                    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[])), '<ordinary locked receipts + staging>', 'exec'),host)
                except ValueError as error:
                    expected={'early':'original ordinary worker did not return naturally','owner':'wrong original Slurm owner'}
                    require(mode in expected and str(error)==expected[mode] and not new.exists(),'wrong original ordinary release or UID verdict')
                else:
                    require(mode=='exact' and (new/'worker.py').read_bytes()==standby['worker.py'],'locked ordinary stage bytes differ')


def fixture(context,row):
    nonce='b'*32
    label=f"training80-dcu-standby-{row['request'][:12]}-{row['node']}"
    account=[row['previous'],label,'COMPLETED','0:0',str(row['worker_seconds']),row['node'],GPU_TRES,'2026-11-25T04:00:00','2026-11-25T07:00:03']
    point=dict(node=row['node'],request=row['request'],step=row['previous'],accounting=account,original_terminal_and_targets_empty=True,
               worker_SHA256=row['worker_SHA256'],expected_SHA256=row['expected_SHA256'],terminal_SHA256='a'*64,scontrol_target_steps=[],squeue_target_steps=[])
    envelope=dict(rc=0,replay_allowed=False,anchor=10,received_monotonic=89,RTT=3,source_SHA256=digest(read_program(context,row,nonce).encode()),
                  observed=dict(read_id=nonce,Slurm_mutations=0,physical_admissions=0,targets=[point]))
    def bind(doc):
        return terminal(context,row,nonce,doc,now=90,wall=100)
    acct,pin_value=bind(envelope)
    rendered=render(context,row,acct,pin_value,None)
    require('check_first_clearance(account(CLEAR_STEP))' not in rendered['launch'] and 'original_bridge_submit_SHA256=' in rendered['launch']
            and 'old_record_bytes = old_handle.read' in rendered['launch'] and 'stream.write(raw["worker.py"])' in rendered['launch'],
            'compiled ordinary post-gate stream differs')
    host_fixture(rendered['launch'],row,account,context[1][6])
    bads=(dict(envelope,received_monotonic=30),dict(envelope,source_SHA256='f'*64),
          dict(envelope,observed=dict(envelope['observed'],targets=[dict(point,squeue_target_steps=['new'])])),
          dict(envelope,observed=dict(envelope['observed'],targets=[dict(point,accounting=account[:2]+['RUNNING']+account[3:])])),
          dict(envelope,observed=dict(envelope['observed'],targets=[dict(point,worker_SHA256='1'*64)])))
    for denied in bads:
        try:
            bind(denied)
        except ValueError:
            continue
        raise ValueError('unsafe ordinary preparation passed')
    return dict(node=row['node'],predecessor=row['previous'],request=row['new_request'],worker_seconds=rendered['worker_seconds'],
                positive=2,negative=len(bads)+2,compile_fixture_SHA256=digest(rendered['launch'].encode()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--observe-original',action='store_true')
    mode.add_argument('--publication')
    parser.add_argument('--node',choices=SPEC)
    parser.add_argument('--terminal-receipt',nargs=2,metavar=('READ_ID','SHA256'))
    args=parser.parse_args()
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one exact node and fresh terminal required for live send')
    context=load(draft=args.check)
    rows=context[-2]
    if args.check:
        result=dict(claim_SHA256=context[-1],checks=[fixture(context,row) for row in rows.values()],SCNet=0,Git=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=observe_predecessor(context,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe_original(context,rows[args.node])
    else:
        result=dispatch(context,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
