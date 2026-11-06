"""Finite, evidence-bound tuning queue; does not launch arbitrary experiment code.

Root dispatches reserved packets through the model's frozen launch_once adapter.
No expired reservation, connection error or local process exit frees a resource.
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import time

ACTIVE = ('RESERVED', 'SUBMITTED', 'UNKNOWN')
STAGES = ('source', 'CPU', 'component', 'model', 'quality', 'service')

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def expand(campaign):
    """Deterministic bounded Cartesian space; proposals confer no run permission."""
    out = []
    for family in campaign['families']:
        axes = family['space']
        require(all(isinstance(v, list) and v for v in axes.values()), 'empty parameter axis')
        require(math.prod(map(len, axes.values())) <= family['max_candidates'], 'space exceeds declared budget')
        require(set(family['baseline']) == set(axes), 'baseline axes differ')
        require(all(v in axes[k] for k, v in family['baseline'].items()), 'baseline outside search space')
        for values in itertools.product(*(axes[k] for k in sorted(axes))):
            params = dict(zip(sorted(axes), values))
            identity = dict(model=family['model'], method=family['method'],
                            source=family['source'], workload=family['workload'], parameters=params)
            row = dict(identity, id=digest(identity), family=family['id'],
                       baseline=params == family['baseline'],
                       initial_stage=family['initial_stage'],
                       prerequisites=family['prerequisites'],
                       hypothesis=family['hypothesis'],
                       zone=family['zone'], nodes_required=family['nodes_required'],
                       memory_floor_bytes=family['memory_floor_bytes'],
                       stage_budgets_s=family['stage_budgets_s'])
            out.append(row)
    require(len({r['id'] for r in out}) == len(out), 'duplicate semantic candidates')
    return sorted(out, key=lambda r: r['id'])


def receive_snapshot(fleet, request_started_monotonic, request_id, raw_response_SHA256):
    """Stamp the authenticated transport receipt once, never when rereading a file.

    Cluster wall timestamps are descriptive only. The request roundtrip bounds
    the oldest possible fresh remote observation; a local monotonic clock measures
    subsequent age. A boot/clock-step anchor prevents reuse across local reboots.
    """
    now=time.monotonic()
    require(0<=now-request_started_monotonic<=60, 'snapshot transport exceeded60s')
    require(bool(request_id)and bool(raw_response_SHA256), 'original transport identity required')
    require('local_receipt'not in fleet, 'do not restamp an old snapshot')
    result=dict(fleet)
    result['local_receipt']=dict(received_monotonic=now,roundtrip_s=now-request_started_monotonic,
        boot_epoch_anchor=time.time()-now,request_id=request_id,raw_response_SHA256=raw_response_SHA256,
        payload_SHA256=digest(fleet),clock='local monotonic; remote wall time never subtracted')
    return result

class Queue:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=15, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA foreign_keys=ON;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, spec TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(id),
          stage TEXT NOT NULL, state TEXT NOT NULL, binding TEXT NOT NULL,
          nodes TEXT NOT NULL, handle TEXT, result TEXT, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, at REAL NOT NULL,
          run TEXT, kind TEXT NOT NULL, data TEXT NOT NULL);
        ''')

    def event(self, ident, kind, data):
        self.db.execute('INSERT INTO events(at,run,kind,data) VALUES(?,?,?,?)',
                        (time.time(), ident, kind, canonical(data).decode()))

    def seed(self, rows):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for row in rows:
                old=self.db.execute('SELECT spec FROM candidates WHERE id=?',(row['id'],)).fetchone()
                require(old is None or old[0]==canonical(row).decode(), 'candidate policy changed; version campaign explicitly')
                self.db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?)', (row['id'], canonical(row).decode()))
            self.event(None, 'seed', {'candidates': len(rows)})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def prepare(self, binding, repo):
        """Verify published exact bytes, workload, floor, stage and explicit root signoff.

        A prepared run is a different object from an expanded parameter proposal.
        This never calls the published launcher or claims an idle device.
        """
        require(binding['root_reviewed'] is True, 'root review missing')
        row = self.db.execute('SELECT spec FROM candidates WHERE id=?', (binding['candidate'],)).fetchone()
        require(row is not None, 'unknown candidate')
        spec = json.loads(row[0])
        require(binding['stage'] in ('component','model','quality','service'), 'GPU queue accepts only device stages; source/CPU use CPU-only adapters')
        require(type(binding['rung']) is int and 0<=binding['rung']<=2, 'predeclared rung0..2 required')
        require(binding['workload_SHA256'] == digest(spec['workload']), 'workload drift')
        require(binding['memory_floor_bytes'] == spec['memory_floor_bytes'], 'floor drift')
        require(binding['deadline_s'] > 0 and binding['deadline_s'] <= spec['stage_budgets_s'][binding['stage']], 'deadline exceeds campaign')
        require(binding['nodes_required'] == spec['nodes_required'], 'resource shape drift')
        require(binding['zone'] == spec['zone'], 'zone drift')
        require(binding['prerequisites_closed'] == spec['prerequisites'], 'prerequisites not explicitly closed')
        require(bool(binding['prerequisite_evidence']), 'prerequisite evidence absent')
        require(len(binding['artifacts']) >= 3, 'source/deps/claim artifacts required')
        require({'source', 'deps', 'claim', 'launcher'} <= {a['role'] for a in binding['artifacts']}, 'incomplete run bundle')
        commit = binding['publication_commit']
        require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'exact publication commit required')
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.PIPE)
        git('merge-base', '--is-ancestor', commit, 'refs/remotes/origin/bw1000-experiments')
        for item in binding['artifacts']:
            name = item['path']
            require(name.startswith('code/bw1000/') and '..' not in Path(name).parts and '\\' not in name, 'artifact outside own source')
            require(hashlib.sha256(git('show', commit + ':' + name)).hexdigest() == item['SHA256'], 'published artifact drift')
            require(sha(Path(repo)/name) == item['SHA256'], 'local artifact drift')
        require(binding['claim_semantics_checked'] is True, 'model adapter must validate exact claim semantics')
        require(bool(binding['inputoutput_binding_SHA256']) and bool(binding['dependency_binding_SHA256']), 'missing exact input/dependency binding')
        ident = digest(binding)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            attempts=self.db.execute('SELECT binding,state,result FROM runs WHERE candidate=? AND stage=?',
                                     (binding['candidate'],binding['stage'])).fetchall()
            require(not any(json.loads(v['binding'])['rung']==binding['rung'] for v in attempts),
                    'same candidate/stage/rung already attempted; reconcile original run')
            if binding['rung']>0:
                prior=[v for v in attempts if json.loads(v['binding'])['rung']==binding['rung']-1]
                require(len(prior)==1 and prior[0]['state']=='TERMINAL' and
                        json.loads(prior[0]['result']).get('promote') is True,
                        'next rung requires previous actual promoted terminal')
            self.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
                (ident, binding['candidate'], binding['stage'], 'READY', canonical(binding).decode(), '[]', None, None, time.time()))
            self.event(ident, 'prepared', {'publication_commit': commit})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        return ident

    def plan(self, fleet, now=None):
        now = time.monotonic() if now is None else now
        receipt=fleet['local_receipt']
        require(digest({k:v for k,v in fleet.items()if k!='local_receipt'})==receipt['payload_SHA256'], 'snapshot payload drift')
        require(abs((time.time()-time.monotonic())-receipt['boot_epoch_anchor'])<2, 'local reboot/clock change requires fresh receipt')
        age=now-receipt['received_monotonic']
        require(0 <= age and 0<=receipt['roundtrip_s'] and age+receipt['roundtrip_s']<=60, 'fresh idle snapshot required, max60s from original request')
        require(fleet['complete'] is True and fleet['total_cards'] == 136, 'incomplete cluster inventory')
        seen, free = set(), {}
        for name, node in fleet['nodes'].items():
            require(len(node['PCIs']) == 8 and len(set(node['PCIs'])) == 8, 'need eight unique PCI/node')
            require(not (seen & {(name, p) for p in node['PCIs']}), 'duplicate identity')
            seen.update((name,p) for p in node['PCIs'])
            if all(node.get(k) is True for k in ('KFD_clean','PID_clean','ports_clean','membership_valid')) and node['active_steps'] == []:
                free.setdefault(node['zone'], []).append(name)
        require(len(seen) == 136, '17node PCI inventory incomplete')
        for row in self.db.execute('SELECT nodes FROM runs WHERE state IN (?,?,?)', ACTIVE):
            for name in json.loads(row[0]):
                for zone in free.values():
                    if name in zone:
                        zone.remove(name)
        ready = []
        for r in self.db.execute('SELECT * FROM runs WHERE state="READY" ORDER BY created,id'):
            b = json.loads(r['binding'])
            ready.append((r, b))
        selected = []
        # FIFO across approved families; no stale guessed performance priority.
        for r, b in ready:
            available = sorted(free.get(b['zone'], []))
            allowed = set(b['allowed_nodes'])
            chosen = [n for n in available if n in allowed][:b['nodes_required']]
            if len(chosen) != b['nodes_required']:
                continue
            selected.append(dict(run=r['id'], candidate=r['candidate'], stage=r['stage'], nodes=chosen,
                                 binding_SHA256=digest(b), publication_commit=b['publication_commit']))
            for n in chosen:
                free[b['zone']].remove(n)
        result = dict(snapshot_SHA256=digest(fleet), observed_epoch=fleet['observed_epoch'], jobs=selected,
                      selected_cards=sum(8*len(j['nodes']) for j in selected),
                      selected_cards_scope='exclusive reserved node capacity, not actual requested/submitted/busy cards',
                      actual_submitted_cards=0,
                      busy_percent=fleet.get('measured_busy_percent'), busy_is_allocation=False,
                      ready=len(ready), proposed_is_executable=False)
        result['plan_SHA256'] = digest(result)
        return result

    def reserve(self, plan, fleet):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            actual = self.plan(fleet)
            require(actual == plan, 'plan changed, replan before dispatch')
            for job in plan['jobs']:
                self.db.execute('UPDATE runs SET state="RESERVED",nodes=? WHERE id=?',
                                (canonical(job['nodes']).decode(),job['run']))
                self.event(job['run'], 'reserved', {'snapshot_SHA256': plan['snapshot_SHA256'], 'nodes': job['nodes']})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def retire_ready(self, ident, reason, evidence_SHA256):
        """Retire a proven prelaunch source defect; never release reserved/unknown work."""
        require(isinstance(reason,str) and reason.strip(), 'prelaunch reason required')
        require(len(evidence_SHA256)==64 and all(c in '0123456789abcdef' for c in evidence_SHA256), 'exact prelaunch evidence SHA required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r=self.db.execute('SELECT state,nodes,handle FROM runs WHERE id=?',(ident,)).fetchone()
            require(r is not None and r['state']=='READY' and json.loads(r['nodes'])==[] and r['handle'] is None,
                    'only unreserved READY may retire; reconcile original active handle')
            result=dict(retired_before_reservation=True,reason=reason,evidence_SHA256=evidence_SHA256,
                        GPU_submission=False,measurement_terminal=False)
            self.db.execute('UPDATE runs SET state="RETIRED_PRELAUNCH",result=? WHERE id=?',(canonical(result).decode(),ident))
            self.event(ident,'retired_prelaunch',result)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def submission(self, ident, handle, unknown=False):
        require(bool(handle.get('request_id')) and bool(handle.get('intent')), 'original request and intent required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r = self.db.execute('SELECT state FROM runs WHERE id=?', (ident,)).fetchone()
            require(r is not None and r[0] == 'RESERVED', 'not exclusively reserved; no replay')
            self.db.execute('UPDATE runs SET state=?,handle=? WHERE id=?',
                            ('UNKNOWN' if unknown else 'SUBMITTED', canonical(handle).decode(), ident))
            self.event(ident, 'submission_unknown' if unknown else 'submitted', handle)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish(self, ident, result):
        """Only actual terminal+post proof releases nodes. A failure remains a failure."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r = self.db.execute('SELECT * FROM runs WHERE id=?', (ident,)).fetchone()
            require(r is not None and r['state'] in ('SUBMITTED','UNKNOWN'), 'original run is not submitted')
            original = json.loads(r['handle'])
            require(result['request_id'] == original['request_id'] and result['intent'] == original['intent'], 'different execution identity')
            require(result['terminal'] is True and result['post_clean'] is True, 'resource release requires actual terminal and post')
            require(bool(result['evidence_SHA256']) and result['planned'] > 0, 'original evidence and planned denominator required')
            counts = [result[k] for k in ('completed','failed','rejected','canceled','missing')]
            require(all(type(v) is int and v >= 0 for v in counts) and sum(counts) == result['planned'], 'denominator lost')
            require(result['qualification'] in STAGES, 'explicit qualification scope required')
            require(result['keeper_passed'] in (True, False, None), 'keeper outcome absent')
            if result.get('promote'):
                require(result['passed'] is True and result['keeper_passed'] is True and result['missing'] == 0, 'failure/unknown cannot promote')
                require(result['qualification'] == r['stage'], 'component result cannot promote model/service')
            self.db.execute('UPDATE runs SET state="TERMINAL",result=? WHERE id=?', (canonical(result).decode(),ident))
            self.event(ident, 'terminal', result)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def summary(self):
        return dict(candidates=self.db.execute('SELECT count(*) FROM candidates').fetchone()[0],
                    runs={r[0]:r[1] for r in self.db.execute('SELECT state,count(*) FROM runs GROUP BY state')},
                    launches_performed_by_this_tool=0)

def select_next_rung(rows, keep_fraction=1/3, min_pairs=3):
    """Within one frozen comparison, keep measured winners; never fill from failures.

    Deterministic paired bootstrap of warm per-request ratios (candidate/base).
    Source/CPU/component cohorts cannot be mixed with model/service cohorts.
    No decision on formal quality NI is inferred from this screening rank.
    """
    import random
    import statistics
    require(0 < keep_fraction <= 1, 'invalid retention')
    for r in rows:
        require({'model','hardware','precision','scope','workload_SHA256','weights_SHA256','metric'} <= set(r['comparison']), 'incomplete comparison identity')
        require(r['comparison']['scope'] in ('component','model','service'), 'build cannot rank as performance')
    keys={digest(r['comparison']) for r in rows}
    require(len(keys)<=1, 'different hardware/workload/precision/scope comparisons')
    accepted, rejected=[],[]
    for r in rows:
        bad=[]
        counts=[r[k]for k in ('completed','failed','rejected','canceled','missing')]
        require(type(r['planned'])is int and r['planned']>0 and all(type(v)is int and v>=0 for v in counts)and sum(counts)==r['planned'], 'ranking denominator lost')
        if r['completed']<r['baseline_completed'] or r['rejected']>r['baseline_rejected'] or r['canceled']>r['baseline_canceled']:bad.append('completion/reject/cancel tradeoff requires explicit service objective, no automatic speedup promotion')
        if r['quality_correct']<r['baseline_quality_correct']:bad.append('quality loss requires prespecified statistical review')
        if not r['numeric_passed'] or not r['post_clean'] or r['keeper_passed'] is not True:bad.append('numeric/lifecycle/keeper')
        if r['missing'] or r['failed']:bad.append('incomplete/failed denominator')
        if not r['quality_screen_passed']:bad.append('quality screen')
        pairs=r['warm_per_user_ratios']
        if len(pairs)<min_pairs or any(not math.isfinite(x) or x<=0 for x in pairs):bad.append('insufficient warm paired measurements')
        if bad:
            rejected.append(dict(candidate=r['candidate'],reasons=bad));continue
        rng=random.Random(digest([r['candidate'],pairs]))
        samples=sorted(statistics.median(rng.choices(pairs,k=len(pairs))) for _ in range(2000))
        lo,hi=samples[49],samples[1949]
        # Throughput ratios >1 mean faster. A confidence interval spanning1 does
        # not become a win merely because this is the only surviving candidate.
        if lo<=1:
            rejected.append(dict(candidate=r['candidate'],reasons=['speedup not resolved above1'],ratio95=[lo,hi]));continue
        accepted.append(dict(candidate=r['candidate'],median_ratio=statistics.median(pairs),ratio95=[lo,hi]))
    accepted.sort(key=lambda r:(-r['ratio95'][0],r['candidate']))
    keep=math.ceil(len(rows)*keep_fraction)
    return dict(promoted=accepted[:keep],not_promoted=rejected+accepted[keep:],
                formal_NI_proven=False,goal_qualified=False,retention=keep_fraction)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--db', default='local-only/tuning/queue.sqlite3')
    sub=a.add_subparsers(dest='command',required=True)
    for name, fields in [('expand',['campaign']),('rank',['results']),('seed',['campaign']),('prepare',['binding']),
                         ('plan',['snapshot']),('reserve',['plan','snapshot']),
                         ('submitted',['run','handle']),('unknown',['run','handle']),('finish',['run','result']),('retire-ready',['run','proof']),('summary',[])]:
        p=sub.add_parser(name)
        for f in fields:p.add_argument(f)
    args=a.parse_args();read=lambda p:json.loads(Path(p).read_bytes())
    if args.command=='rank':out=select_next_rung(read(args.results))
    elif args.command=='expand':out=expand(read(args.campaign))
    else:
        q=Queue(args.db)
        if args.command=='seed':q.seed(expand(read(args.campaign)));out=q.summary()
        elif args.command=='prepare':out={'run':q.prepare(read(args.binding),Path(__file__).resolve().parents[1])}
        elif args.command=='plan':out=q.plan(read(args.snapshot))
        elif args.command=='reserve':q.reserve(read(args.plan),read(args.snapshot));out=q.summary()
        elif args.command in ('submitted','unknown'):q.submission(args.run,read(args.handle),args.command=='unknown');out=q.summary()
        elif args.command=='finish':q.finish(args.run,read(args.result));out=q.summary()
        elif args.command=='retire-ready':
            proof=read(args.proof);q.retire_ready(args.run,proof['reason'],proof['evidence_SHA256']);out=q.summary()
        else:out=q.summary()
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
