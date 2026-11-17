"""A100-27 endpoint: exact metadata/header reads; no tensor payload reads."""
import sys


def run_source(plan, context):
    cpus = arm(context['epoch'], context['deadline'])
    root = Path(plan['source_namespace'])
    identity = own(root, dict(context, endpoint='source', CPU_affinity=cpus, nice=10,
                              RLIMIT_AS_bytes=512*1024**2, actual_GPU_cards=0))
    reads, files, bindings, stats = 0, {}, {}, {}
    try:
        for name, pin in plan['members'].items():
            if pin['kind'] == 'metadata':
                data, before = read_pinned(pin['source'], pin, plan['max_file_bytes'])
                reads += len(data)
            else:
                tick()
                path = no_links(pin['source'])
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, 'rb') as stream:
                    before_st = os.fstat(stream.fileno())
                    require(stat.S_ISREG(before_st.st_mode) and before_st.st_size == pin['shard_bytes'],
                            'shard stat differs')
                    prefix = stream.read(8)
                    require(len(prefix) == 8 and struct.unpack('<Q', prefix)[0] == pin['header_bytes'],
                            'shard header length differs')
                    header = stream.read(pin['header_bytes'])
                    require(len(header) == pin['header_bytes'] and sha(header) == pin['header_SHA256'],
                            'shard header SHA differs')
                    require(len(set(json.loads(header)) - {'__metadata__'}) == pin['tensors'],
                            'header tensor count differs')
                    require(stable(before_st) == stable(os.fstat(stream.fileno())) == stable(path.stat()),
                            'shard stat changed during header read')
                    data, before = prefix + header, stable(before_st)
                reads += len(data)
            files[name], stats[name] = data, before
        for key, case in plan['cases'].items():
            rows = []
            for rank in range(8):
                np = case['records']['train/nvidia-baseline/rank-%02d.json' % rank]
                rp = case['records']['capture/rank-%02d/receipt.json' % rank]
                native, _ = read_pinned(case['run_root'] + '/train/nvidia-baseline/rank-%02d.json' % rank, np, plan['max_file_bytes'])
                receipt, _ = read_pinned(case['run_root'] + '/capture/rank-%02d/receipt.json' % rank, rp, plan['max_file_bytes'])
                reads += len(native) + len(receipt)
                rows.append(initial_row(case, rank, json.loads(native), json.loads(receipt), np, rp))
            bindings[key] = dict(rows=rows, rank_denominator=8,
                                 named_digest_all8_agree=len({x['named_initial_digest'] for x in rows}) == 1,
                                 file_SHA_all8_agree=len({x['initial_file_SHA256'] for x in rows}) == 1)
        require(reads == plan['exact_source_read_bytes'], 'source read byte accounting differs')
        derived = canonical(dict(schema='T03_T04_initial_dependencies_v1', context=context, cases=bindings,
                                 initial_tensor_bytes_read=0, cross_platform_RNG_or_dropout_qualified=False))
        require(len(derived) <= plan['max_initial_binding_bytes'], 'derived initial binding budget exceeded')
        files['initial-bindings.json'] = derived
        require(len(files) == 36, 'metadata member denominator differs')
        compressed = pack(files, plan, context)
        tick()
        outcome = dict(identity, passed=True, metadata_only=True, full_weight_identity=False,
                       member_count=36, source_files_read=67, source_bytes_read=reads,
                       weight_payload_bytes_read=0, initial_tensor_bytes_read=0,
                       bundle_bytes=len(compressed), bundle_SHA256=sha(compressed), source_file_stats=stats)
        sys.stdout.buffer.write(compressed)
        sys.stdout.buffer.flush()
        save(root/'outcome.json', outcome)
        return 0
    except BaseException as error:
        save(root/'outcome.json', dict(identity, passed=False, error_type=type(error).__name__,
                                      source_bytes_read=reads, completed_members=len(files), replay_allowed=False))
        return 1


raise SystemExit(run_source(PLAN, CONTEXT))
