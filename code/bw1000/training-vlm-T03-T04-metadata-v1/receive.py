"""Hygon endpoint: strict36-member metadata-only landing, never a model root."""
import sys
import pwd
import subprocess


def parent_reservations(plan):
    def query(argv):
        tick()
        budget = min(12, DEADLINE-time.time()-5, MONO_END-time.monotonic()-5)
        require(budget > 0, 'no parent query budget')
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=budget)
        require(result.returncode == 0 and len(result.stdout) <= 65536 and len(result.stderr) <= 65536,
                'bounded parent query failed')
        return result.stdout.decode('utf-8').strip()
    parents = plan['parent_jobs']
    raw = query(['squeue', '-h', '-j', ','.join(parents), '-o', '%i|%T|%D|%N|%u'])
    rows = [line.split('|') for line in raw.splitlines()]
    require(len(rows) == 2 and len({x[0] for x in rows}) == 2, 'parent row denominator differs')
    queue = {x[0]:x for x in rows}
    require(set(queue) == set(parents), 'foreign or missing parent')
    owner = pwd.getpwuid(os.getuid()).pw_name
    output, all_nodes = [], []
    for parent in parents:
        row = queue[parent]
        require(len(row) == 5 and row[1] == 'RUNNING' and row[2] == '40' and row[4] == owner,
                'parent no longer owned RUNNING40')
        fields = dict(x.split('=', 1) for x in query(['scontrol', 'show', 'job', parent, '-o']).split() if '=' in x)
        require(fields['JobId'] == parent and fields['JobState'] == 'RUNNING' and fields['NumNodes'] == '40'
                and fields['NodeList'] == row[3] and fields['UserId'].split('(', 1)[0] == owner,
                'squeue/scontrol parent identity differs')
        tres = dict(x.split('=', 1) for x in fields['AllocTRES'].split(','))
        require(tres.get('gres/dcu') == '320' and tres.get('node') == '40',
                'parent actual reserved DCU320 differs')
        nodes = query(['scontrol', 'show', 'hostnames', row[3]]).splitlines()
        require(len(nodes) == len(set(nodes)) == 40, 'parent40 hostnames differ')
        all_nodes.extend(nodes)
        output.append(dict(job_id=parent, state='RUNNING', nodes=nodes, node_count=40, owner=owner,
                           AllocTRES=fields['AllocTRES'], reserved_DCU=320, EndTime=fields.get('EndTime')))
    require(len(all_nodes) == len(set(all_nodes)) == 80, 'parent80 membership overlaps')
    return dict(observed_epoch=time.time(), parents=output, reserved_nodes=80, reserved_cards=640,
                cards_per_node=8, cards_are_reservation_not_busy=True, GPU_busy='NOT_MEASURED',
                GPU_physical_inventory='NOT_APPLICABLE_TO_HOST_METADATA_SCOPE')


def verify_members(files, plan, context):
    require(set(files) == set(plan['members']) | {'initial-bindings.json'}, 'member set differs')
    for name, pin in plan['members'].items():
        raw = files[name]
        if pin['kind'] == 'metadata':
            require(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'original metadata differs')
        else:
            require(len(raw) == 8 + pin['header_bytes'] and struct.unpack('<Q', raw[:8])[0] == pin['header_bytes']
                    and sha(raw[8:]) == pin['header_SHA256'], 'original header differs')
    raw = files['initial-bindings.json']
    require(len(raw) <= plan['max_initial_binding_bytes'], 'initial binding too large')
    initial = json.loads(raw)
    require(canonical(initial) == raw and initial['context'] == context and
            initial['schema'] == 'T03_T04_initial_dependencies_v1' and
            initial['initial_tensor_bytes_read'] == 0 and
            initial['cross_platform_RNG_or_dropout_qualified'] is False,
            'derived binding contract differs')
    require(set(initial['cases']) == set(plan['cases']), 'initial case set differs')
    for key, case in plan['cases'].items():
        value = initial['cases'][key]
        rows = value['rows']
        require(value['rank_denominator'] == 8 and [x['rank'] for x in rows] == list(range(8)), 'rank set differs')
        for rank, row in enumerate(rows):
            prefix = case['run_root']
            for field, rel in [('source_native', 'train/nvidia-baseline/rank-%02d.json' % rank),
                               ('source_capture_receipt', 'capture/rank-%02d/receipt.json' % rank)]:
                require(row[field] == dict(path=prefix+'/'+rel, **case['records'][rel]), 'initial metadata source differs')
            require(row['initial_path'] == prefix+'/capture/rank-%02d/initial.pt' % rank and
                    row['native_global_step'] == 25 and row['captured_optimizer_updates'] == 2 and
                    row['initial_tensor_read'] is False and type(row['initial_file_bytes']) is int and
                    0 < row['initial_file_bytes'] < 2*1024**3 and
                    re.fullmatch('[0-9a-f]{64}', row['initial_file_SHA256']) is not None and
                    re.fullmatch('[0-9a-f]{64}', row['named_initial_digest']) is not None, 'initial row differs')
        require(value['named_digest_all8_agree'] == (len({x['named_initial_digest'] for x in rows}) == 1)
                and value['file_SHA_all8_agree'] == (len({x['initial_file_SHA256'] for x in rows}) == 1),
                'initial agreement differs')
    return initial


def run_receiver(plan, context):
    cpus = arm(context['epoch'], context['deadline'])
    reservation = parent_reservations(plan)  # same epoch; no namespace or metadata write before this gate
    root = Path(plan['target_namespace'])
    identity = own(root, dict(context, endpoint='receiver', CPU_affinity=cpus, nice=10,
                              RLIMIT_AS_bytes=512*1024**2, actual_GPU_cards=0,
                              parent_reservations=reservation))
    completed = []
    try:
        compressed = sys.stdin.buffer.read(plan['max_compressed_bytes'] + 1)
        files = unpack(compressed, plan, context)
        initial = verify_members(files, plan, context)
        for name in sorted(files):
            tick()
            relative(name)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(files[name])
                stream.flush()
                os.fsync(stream.fileno())
            require(sha(path.read_bytes()) == sha(files[name]), 'written metadata differs')
            completed.append(dict(name=name, bytes=len(files[name]), SHA256=sha(files[name])))
        tick()
        value = dict(identity, passed=True, metadata_only=True, full_weight_identity=False,
                     actual_GPU_cards=0, completed_members=36, member_denominator=36,
                     files=completed, bundle_bytes=len(compressed), bundle_SHA256=sha(compressed),
                     initial_rank_denominator=16, initial_binding_SHA256=sha(files['initial-bindings.json']),
                     named_digest_agreement={k:v['named_digest_all8_agree'] for k,v in initial['cases'].items()},
                     weight_payload_bytes_read=0, initial_tensor_bytes_read=0, GPU_GO=False, Model_GO=False)
    except BaseException as error:
        value = dict(identity, passed=False, error_type=type(error).__name__, completed_members=len(completed),
                     member_denominator=36, replay_allowed=False, GPU_GO=False, Model_GO=False)
    save(root/'outcome.json', value)
    summary = {k:v for k,v in value.items() if k not in ('files', 'parent_reservations')}
    summary['parent_reservations_SHA256'] = sha(canonical(reservation))
    summary['reserved_cards'] = 640
    summary['reserved_not_busy'] = True
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if value['passed'] else 1


raise SystemExit(run_receiver(PLAN, CONTEXT))
