"""One read-only terminal seal and stdlib audit of the existing 569/840 run.

Raw seals contain existing fixtures and stay on R/L, never Git. No submit,
cancel, GPU API, tokenizer import, model import, or retry is implemented.
"""
import argparse, ast, base64, gzip, hashlib, json, math, subprocess
from pathlib import Path

RUN = 'ordinary-20261103-e'
PREFIX = 'Kimi-ordinary-r410-' + RUN
NODES = ['a14r2n08', 'a14r2n09', 'a14r2n10', 'a14r2n12']
RUNTIME = '490c138d7c0f3708be312a98f4989c1f8c0fd00009eeb88b521ebe240c896794'
CONTROLLER = 'd00ef72dd1a655a045983680cbba5ed757fa8fe41bcef81f2f30979616301420'
NATIVE = '6eb2bd61b5bfc0c5df3fe809bf37f065de2832be07765554db86e512cb6cd1ba'
CPU825 = 'cb3f1db0e0cd9578c77581a379d35f6683d91d6cfdb7556365a97b28ae16ed80'
CPU826 = 'd89eee95fce8a5322781087202abec482e1576e79908349c22bc9e861d1eb9b9'
PROMPTS = ['66090df28f4a0b8024a03f769dd9d1f76c91a0348ea7141a31c48ae96f373934',
           'ffcad0cc15164ae3c697434fc93772ffb86df5a7d1c4d608dfc0894dad024054']
sha = lambda b: hashlib.sha256(b).hexdigest()
ids_sha = lambda v: sha(json.dumps(v, separators=(',', ':')).encode())


def capture(root):
    root = Path(root)
    files, hashes = {}, {}
    def add(name, path):
        path = Path(path); data = path.read_bytes()
        files[name] = dict(path=str(path), SHA256=sha(data), base64=base64.b64encode(data).decode())
        return json.loads(data) if path.suffix == '.json' else data
    watch = add('watch', root/'results'/(PREFIX+'-watch.json'))
    assert watch['completed'] is True, 'Original run is not terminal; do not poll/reload'
    add('actualCLI', watch['actual_final_CLI_preflight']['path'])
    add('plan', root/'results'/(PREFIX+'-execution-plan.json'))
    add('claim', root/'control'/(PREFIX+'-claim.json'))
    for version, digest, label in [('runtime-v3', RUNTIME, 'runtime'), ('controller-v5', CONTROLLER, 'controller')]:
        directory = root/'incoming/Kimi-TP8PP4-ordinary-r410'/version
        manifest = add(label+'/SOURCE_MANIFEST.json', directory/'SOURCE_MANIFEST.json')
        assert files[label+'/SOURCE_MANIFEST.json']['SHA256'] == digest
        for name, expected in manifest['files_SHA256'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            add(label+'/'+name, directory/name); assert files[label+'/'+name]['SHA256'] == expected
    build = add('CPU825', root/'candidates/Kimi-TP8PP4-ordinary-r410-CPU-v4/CPU-result.json')
    add('CPU826', root/'candidates/Kimi-TP8PP4-ordinary-r410-runtime-CPU-v1/CPU-result.json')
    fp = json.loads(base64.b64decode(files['runtime/FIXTURE_PINS.json']['base64']))
    pins = json.loads(base64.b64decode(files['runtime/bindings.json']['base64']))
    def host(p):
        p = str(p)
        if p.startswith('/refactor/'): return root/p.removeprefix('/refactor/')
        if p.startswith('/shared/'): return root.parent/'shared'/p.removeprefix('/shared/')
        return Path(p)
    add('fixtures', host(fp['frozen_inputs_path']))
    for filename in ['config.json', 'model.safetensors.index.json']:
        path = pins['model_view']['path']+'/'+filename
        hashes[path] = sha(host(path).read_bytes())
    for node in range(4):
        path = root/'results'/(PREFIX+'-node'+str(node)+'.json')
        if path.exists(): add('node/'+str(node), path)
    for path, expected in build['all_linked_object_SHA256'].items():
        hashes[path] = sha(host(path).read_bytes()); assert hashes[path] == expected
    hashes[build['native']] = sha(host(build['native']).read_bytes())
    for item in build['actual_dependency_library_SHA256'].values():
        hashes[item['path']] = sha(host(item['path']).read_bytes())
    steps = ['910380.569', '910407.840']
    for row in watch.get('post', watch.get('post_after_failure', [])):
        if row['actual_step'] not in steps: steps.append(row['actual_step'])
    argv = ['sacct', '-j', ','.join(steps), '-P', '-n', '--format=JobIDRaw,JobName%160,State,ExitCode,ElapsedRaw,NodeList%120']
    accounting = subprocess.check_output(argv, text=True, timeout=30)
    return dict(schema=1, files=files, current_artifact_hashes=hashes,
                accounting_argv=argv, accounting=accounting, GPU_or_Slurm_submissions=0)


def metrics(request):
    events = request['events']; ids = request['generated_ids']
    assert [x['token'] for x in events] == ids
    t = [x['at_s'] for x in events]; assert all(math.isfinite(x) and x >= 0 for x in t)
    assert all(b >= a for a, b in zip(t, t[1:]))
    visible = [e['at_s'] for e in events if e['token'] not in [163585, 163586]]
    gaps = sorted(b-a for a, b in zip(visible, visible[1:]))
    def pct(p, values=gaps):
        if not values: return None
        k = (len(values)-1)*p; i = int(k); j = min(i+1, len(values)-1)
        return values[i]+(values[j]-values[i])*(k-i)
    span = visible[-1]-visible[0] if visible else None
    rate = (len(visible)-1)/span if len(visible) > 1 and span > 0 else None
    allspan = t[-1]-t[0] if t else None; allgaps = sorted(b-a for a,b in zip(t,t[1:]))
    return dict(name=request['name'], generated_including_EOS=len(ids), visible_non_EOS=len(visible),
                EOS_count=len(ids)-len(visible), TTFT_s=visible[0] if visible else None,
                first_last_visible_s=span, visible_tok_s=rate, visible_tok_per_32card_s=rate/32 if rate else None,
                ITL_P95_s=pct(.95), ITL_P99_s=pct(.99), ITL_max_s=max(gaps) if gaps else None,
                including_EOS_first_last_s=allspan,
                including_EOS_decode_tok_s=(len(t)-1)/allspan if len(t)>1 and allspan>0 else None,
                including_EOS_ITL_P95_s=pct(.95,allgaps), including_EOS_ITL_P99_s=pct(.99,allgaps),
                generation_through_EOS_s=request['full_request_s'])


def audit(seal):
    files = seal['files']; errors = []; result = dict(completed=True, passed=False, checks=[], errors=errors,
        actual_GPU='910380.569', actual_CPUkeeper='910407.840', HTTP=False, SSE=False,
        client_SLO=False, fresh_TP16_causal_comparison=False, topology_package_not_network_only=True,
        independent_tokenizer_decode=False, parser_scope='Recheck producer pinned XTML parser result and independent integer gold; no tokenizer re-decode in stdlib reader',
        clock_scope='Node0 ordinary engine-consumer request-relative monotonic after before-reset and after four-leader token gather; not HTTP/SSE/client. Never subtract clocks across nodes or sum overlapping stages.')
    def data(name):
        item = files[name]; raw = base64.b64decode(item['base64']); assert sha(raw) == item['SHA256'], name
        return raw
    obj = lambda name: json.loads(data(name))
    def check(label, fn):
        try: fn(); result['checks'].append(dict(name=label, passed=True))
        except Exception as e: errors.append(dict(check=label, error=repr(e))); result['checks'].append(dict(name=label, passed=False))
    def source():
        for label, digest in [('runtime', RUNTIME), ('controller', CONTROLLER)]:
            assert sha(data(label+'/SOURCE_MANIFEST.json')) == digest
            for name, expected in obj(label+'/SOURCE_MANIFEST.json')['files_SHA256'].items(): assert sha(data(label+'/'+name)) == expected
        assert sha(data('CPU825')) == CPU825 and sha(data('CPU826')) == CPU826
        build = obj('CPU825'); assert build['completed'] and build['passed'] and build['native_SHA256'] == NATIVE
        assert len(build['all_linked_object_SHA256']) == 33
        for path, digest in build['all_linked_object_SHA256'].items(): assert seal['current_artifact_hashes'][path] == digest
        assert seal['current_artifact_hashes'][build['native']] == NATIVE
        assert sha(data('fixtures')) == obj('runtime/FIXTURE_PINS.json')['frozen_inputs_SHA256']
        cli = obj('actualCLI'); watch = obj('watch')
        assert sha(data('actualCLI')) == watch['actual_final_CLI_preflight']['SHA256']
        assert cli['passed'] and not cli['Native_dlopen'] and cli['GPU'] == 0 and cli['Slurm_commands'] == 0
        assert cli['runtime_source_manifest_SHA256'] == RUNTIME and {v['node'] for v in cli['nodes']} == set(range(4))
        for role in cli['nodes']:
            assert role['passed'] and role['typed_native_s'] == 5100
            assert role['required_environment_observed'] == obj('runtime/bindings.json')['required_environment']
        model = obj('runtime/bindings.json')['model_view']
        assert seal['current_artifact_hashes'][model['path']+'/config.json'] == model['config_SHA256']
        assert seal['current_artifact_hashes'][model['path']+'/model.safetensors.index.json'] == model['index_SHA256']
        tree = ast.parse(data('runtime/bootstrap.py')); main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        assert main.args.defaults[1].value is None and tree.body[-1].body[0].value.func.id == 'main'
    check('frozen_sources_Native33objects_825_826_fixtures_actual_default_entry', source)
    watch = obj('watch'); plan = obj('plan'); rows = [line.split('|') for line in seal['accounting'].splitlines() if line]
    def lifecycle():
        assert watch['completed'] and watch['passed'] and watch['actual_GPU'] == '910380.569'
        assert watch['last_observation']['srun_exit_code'] == 0
        for step in ['910380.569', '910407.840']:
            row = [r for r in rows if r[0] == step]; assert len(row) == 1 and row[0][2] == 'COMPLETED' and row[0][3] == '0:0'
        assert plan['run'] == RUN and plan['runtime_source_manifest_SHA256'] == RUNTIME and plan['controller_source_manifest_SHA256'] == CONTROLLER
        assert plan['Native_SHA256'] == NATIVE and plan['target_cards'] == 32 and plan['draft_cards'] == 0
        epoch = watch['original_GPU_start_unix']; assert watch['Native_deadline_unix'] == epoch+5100 and watch['GPU_deadline_unix'] == epoch+5400 and watch['shell_deadline_unix'] == epoch+5280
        assert watch['keeper_deadline_unix'] == watch['original_keeper_start_unix']+6000
        post = watch['post']; assert len(post) == 4 and {p['node'] for p in post} == set(NODES)
        assert len({p['actual_step'] for p in post}) == 1
        for p in post:
            assert p['passed'] and p['KFD_empty'] and not p['matching_bootstrap_PIDs'] and p['no_LISTEN'] and all(p['ports_bindable'].values())
            assert all(e['state'] == '06' for e in p['port_entries'])
        postrow = [r for r in rows if r[0] == post[0]['actual_step']]; assert len(postrow) == 1 and postrow[0][2:4] == ['COMPLETED', '0:0']
        result.update(GPU_elapsed_s=int(next(r[4] for r in rows if r[0] == '910380.569')), post_step=post[0]['actual_step'], original_GPU_epoch=epoch)
        result['GPU_residency_card_s'] = 32*result['GPU_elapsed_s']
    check('Slurm_srun_original_deadlines_independent_four_node_post', lifecycle)
    specs = obj('runtime/CAPI_PROVIDERS.json')['symbols']; expected_libs = obj('runtime/bindings.json')['actual_dependency_library_SHA256']
    fixtures = obj('fixtures')['fixtures']; signatures = []; loads = []; engine_s = []; request_walls = []
    result['observed_workers'] = []
    for node in range(4):
        if 'node/'+str(node) in files:
            q = obj('node/'+str(node))
            result['observed_workers'].append(dict(node=node, phase=q.get('phase'), error=q.get('error'),
                loaded_keys=q.get('loaded_keys'), CAPI_completed=len(q.get('ordinary_calls', [])),
                normal8destroy=q.get('all8_owned_workers_shutdown_completed', False),
                sampler_closed=q.get('sampler_CPU_cleanup_completed', False),
                control_closed=q.get('control_CPU_cleanup_completed', False), cleanup_errors=q.get('cleanup_errors')))
    for node in range(4):
        def worker(node=node):
            q = obj('node/'+str(node)); assert q['actual_step'] == '910380.569' and q['node'] == node and q['source_manifest_SHA256'] == RUNTIME and q['Native_SHA256'] == NATIVE
            assert q['completed'] and q['all8_owned_workers_shutdown_completed'] and q['sampler_CPU_cleanup_completed'] and q['control_CPU_cleanup_completed'] and not q['cleanup_errors']
            assert q['layout']['TP'] == 8 and q['layout']['PP'] == 4 and q['layout']['PP_stage'] == node and q['layout']['sampling_node'] == 3
            assert q['layout']['local_workers'] == 8 and q['layout']['TP_node_rank'] == 0 and not q['layout']['public_graph_enabled'] and not q['layout']['owned_metadata_enabled']
            assert q['all93_stage_keys']['all93_layer_ids_exactly_once_across_PPstages'] and q['cache_geometry']['max_total_tokens'] == 32768
            keyinfo = q['actual_engine_weight_keys']; ranges = [[0,24],[24,47],[47,71],[71,93]]
            assert keyinfo['actual_layer_ids'] == list(range(*ranges[node])) and keyinfo['actual_range'] == ranges[node]
            assert keyinfo['embedding_present'] == (node == 0) and keyinfo['lm_head_present'] == (node == 3)
            assert q['loaded_keys'] == keyinfo['key_count'] == sum(p['keys'] for p in q['native_load_flush_profiles'])
            if node == 3:
                head = q['head_after_requests']; assert head['expected_CPU_source_registered'] and len(head['owners']) == 8
                for owner in head['owners']:
                    assert owner['postload_rows_verified'] and owner['physical_head_shape'] == [20480,7168]
                    assert owner['head_parameter_bytes'] == 293601280 and not owner['full_GPU_head_retained']
            assert set(q['actual_CAPI_providers']) == set(specs) and len(specs) == 20
            for name, spec in specs.items():
                assert q['actual_CAPI_providers'][name] == spec['path'] and seal['current_artifact_hashes'][spec['path']] == spec['SHA256']
            for name, expected in expected_libs.items(): assert q['actual_loaded_libraries'][name] == expected
            def masks(v):
                assert v['verified'] and v['width'] == 4; owners = v['actual_owner_report']['owners']; assert len(owners) == 8 and {o['device'] for o in owners} == set(range(8))
                allcpus = []; out = []
                for o in owners:
                    cpus = [int(x) for x in o['verified_mask'].split(',')]; assert o['changes'] == 1 and len(set(cpus)) == len(cpus) == 4
                    allcpus += cpus; out.append((o['device'], o['changes'], o['verified_mask']))
                assert len(set(allcpus)) == 32; return sorted(out)
            assert masks(q['owner_affinity_after_first_forward']) == masks(q['owner_affinity'])
            memory = q['last_memory_guard']; assert memory['minimum_free_bytes'] > 4<<30 and len(memory['owners']) == 4
            assert all(not p['snapshot']['sampler_error'] and p['snapshot']['sample_count'] >= 1 for p in memory['owners'])
            values = [m[0] for p in memory['owners'] for m in p['snapshot']['memory']]; assert len(values) == 32 and min(values) > 4<<30
            pair = q['ordinary_report']; assert pair['completed'] and len(pair['requests']) == 2
            calls = q['ordinary_calls']; expected_calls = []; sig = []
            for i, (r, f) in enumerate(zip(pair['requests'], fixtures)):
                assert r['name'] == f['name'] and r['input_tokens'] == [8635, 32638][i] and r['prompt_ID_SHA256'] == PROMPTS[i] == ids_sha(f['prompt_tokens'])
                ids = r['generated_ids']; assert 1 <= len(ids) <= 128 and ids[-1] in [163585, 163586] and not any(t in [163585, 163586] for t in ids[:-1])
                assert r['output_budget'] == 128 and r['completed'] and r['normal_reset'] and r['finish'] == 'eos' and r['output_ID_SHA256'] == ids_sha(ids)
                assert len(f['prompt_tokens'])+128 <= 32768 and r['last_past'] == len(f['prompt_tokens'])+len(ids)-1
                quality = r['quality']; assert quality['passed'] and quality['natural_EOS'] and quality['tool'] == 'calculate' and quality['arguments'] == f['expected_calculate_arguments'] == {'a':16195, 'b':39747}
                assert all(type(v) is int for v in quality['arguments'].values())
                for past in range(0, len(f['prompt_tokens']), 512):
                    chunk = f['prompt_tokens'][past:past+512]; expected_calls.append((0, past, len(chunk), ids_sha(chunk)))
                for j in range(len(ids)-1): expected_calls.append((1, len(f['prompt_tokens'])+j, 1, ids_sha([ids[j]])))
                sig.append((r['name'], r['prompt_ID_SHA256'], r['output_ID_SHA256'], len(ids)))
            assert len(calls) == len(expected_calls) == q['output_metrics']['engine_calls']
            for serial, (c, expected) in enumerate(zip(calls, expected_calls)):
                assert c['serial'] == serial and (c['kind'], c['past'], c['tokens'], c['input_ID_SHA256']) == expected
                assert c['CAPI_return'] == 0 and c['validated'] and c['published_flags'] == (3 if node == 3 else 0)
                assert math.isfinite(c['local_CAPI_wall_s']) and c['local_CAPI_wall_s'] >= 0
                if node == 3: assert c['tail_logits_finite'] is True
            signatures.append(sig); loads.append(q['loading_seconds']); engine_s.append(q['output_metrics']['engine_wall_seconds']); request_walls.append([r['full_request_s'] for r in pair['requests']])
            if node == 0: result['node0_engine_consumer_timeline'] = [metrics(r) for r in pair['requests']]
            if node == 0: result['context_capacity'] = [dict(prompt_tokens=r['input_tokens'], output_budget=128, context=32768, conservative_budget_headroom=32768-r['input_tokens']-128, actual_last_cache_past=r['last_past'], actual_unused_cache_positions=32768-r['last_past']) for r in pair['requests']]
        check('actual_worker_'+str(node), worker)
    if len(signatures) == 4:
        check('four_leader_same_actual_tokens', lambda: require(all(s == signatures[0] for s in signatures)))
        def call_enclosures():
            reports = [obj('node/'+str(i)) for i in range(4)]
            maxima = [max(q['ordinary_calls'][i]['local_CAPI_wall_s'] for q in reports) for i in range(len(reports[0]['ordinary_calls']))]
            for q in reports:
                assert all(abs(c['CAPI_max4leader_s']-value) <= 1e-9 for c,value in zip(q['ordinary_calls'], maxima))
            result['sum_aligned_call_max4leader_CAPI_enclosure_s'] = sum(maxima)
            result.update(external_engine_requests=2, matched_logical_forward_rounds=len(maxima),
                          CAPI_calls_per_leader=[len(q['ordinary_calls']) for q in reports],
                          leader_CAPI_calls_total=sum(len(q['ordinary_calls']) for q in reports))
            result['CAPI_enclosure_scope'] = 'Max of four matched-call durations, summed serial calls; includes CAPI input/engine/collective waits/output readback. Not pure GPU busy/TP time or complete outer service wall.'
        check('matched_call_max4leader_duration_enclosures', call_enclosures)
        result.update(load_s_per_leader=loads, max_load_s=max(loads), engine_enclosure_s_per_leader=engine_s,
                      complete_request_generation_through_EOS_s_per_leader=request_walls,
                      request_wall_max4=[max(w[i] for w in request_walls) for i in range(2)], normal_workers=32)
    result['passed'] = not errors
    result['preserved_prior_failures'] = '835 missingenv;836 hiddenGPU CDLL abort;563 uniformprovider;566 prematureaffinity;label-d zerojob ELFcrossSHA: separately retained, not overwritten or counted as new quality/performance negatives'
    return result


def require(value):
    assert value


def main():
    p = argparse.ArgumentParser(); p.add_argument('--collect-root', type=Path); p.add_argument('--seal', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.collect_root:
        seal = capture(a.collect_root)
        with a.seal.open('xb') as f: f.write(gzip.compress(json.dumps(seal, separators=(',', ':')).encode(), mtime=0))
    else: seal = json.loads(gzip.decompress(a.seal.read_bytes()))
    result = audit(seal); result['seal_SHA256'] = sha(a.seal.read_bytes())
    with a.output.open('x', encoding='utf8') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k:result.get(k) for k in ['completed','passed','actual_GPU','actual_CPUkeeper','errors','node0_engine_consumer_timeline','seal_SHA256']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
