"""CPU-only audit of a failed original run's completed prefix; never qualifies the run."""
import argparse, importlib.util, json, math
from pathlib import Path
import frozen_terminal as f


def completed_prefix(requests):
    assert isinstance(requests, list) and len(requests) <= 33
    n = 0
    while n < len(requests) and requests[n].get('completed'):
        assert requests[n]['reset_before'] and requests[n]['reset_after']
        n += 1
    assert len(requests) - n <= 1, 'noncontiguous completion or multiple partial requests'
    if n < len(requests):
        assert requests[n]['reset_before'] and not requests[n]['reset_after']
    return n


def cross_prefix(ranks, workers):
    assert len(ranks) == 16 and [r['rank'] for r in ranks] == list(range(16))
    assert all(r['identity_passed'] for r in ranks)
    assert len({(r['node'], r['PCI']) for r in ranks}) == 16
    n = min(len(r['validated']) for r in ranks)
    for i in range(n):
        rows = [r['validated'][i] for r in ranks]
        for k in ('index', 'ordinal', 'suite', 'diagnostic', 'screen_index', 'prompt_tokens',
                  'prefill_fragments', 'generated_including_EOS', 'output_IDs_SHA256',
                  'natural_EOS', 'cap_hit'):
            assert all(r[k] == rows[0][k] for r in rows), 'rank disagreement: ' + k
    assert len(workers) == 2 and all(workers)
    phases = ['created', 'loaded'] + ['case' + str(i) for i in range(n)]
    for i, w in enumerate(workers):
        assert w['actual_GPU'] == f.GPU and w['gpu_epoch'] == f.EPOCH
        assert w['node'] == f.NODES[i]
        observed = [r['phase'] for r in w['rounds']]
        assert observed[:len(phases)] == phases
    assert [r['digest'] for r in workers[0]['rounds'][:len(phases)]] == [
        r['digest'] for r in workers[1]['rounds'][:len(phases)]]
    return n


def main(original, source, accounting, plan_path, watch):
    source, original = Path(source), Path(original)
    rows = f.accounting(json.loads(Path(accounting).read_bytes()))
    assert all(rows[k]['normalized_state'] in f.TERMINAL for k in (f.GPU, f.KEEPER))
    assert not f.successful(rows[f.GPU]) or not f.successful(rows[f.KEEPER]), 'failed-run reader only'
    assert f.H((Path(__file__).parent/'frozen_terminal.py').read_bytes()) == 'a9431a8c8071335931552cac815ed2954d8237513c57cc0881fb372d168ada61'
    sm = (source/'SOURCE_MANIFEST.json').read_bytes()
    assert f.H(sm) == f.MF and f.H((source/'GPU_MANIFEST.json').read_bytes()) == f.GMF
    assert all(f.H((source/n).read_bytes()) == h for n,h in json.loads(sm)['files_SHA256'].items())
    inp = json.loads((source/'INPUT_BINDING.json').read_bytes())
    raw = Path(plan_path).read_bytes()
    assert f.H(raw) == inp['private_plan']['SHA256'] and len(raw) == inp['private_plan']['bytes']
    plan = json.loads(raw)
    assert plan['planned'] == len(plan['requests']) == 32
    assert [p['ordinal'] for p in plan['requests']] == f.ORDINALS
    assert len({p['case_id'] for p in plan['requests']}) == 32
    assert sorted(p['suite'] for p in plan['requests']) == ['gsm8k']*16+['mmlu']*16
    for p, frozen in zip(plan['requests'], inp['private_plan']['requests']):
        assert all(p[k] == frozen[k] for k in ('ordinal','case_id','suite','prompt_tokens',
                   'prompt_IDs_SHA256','original_private_SHA256','body_SHA256','max_new_tokens','prefill_window'))
    originals = []
    def read(path):
        path = Path(path); raw = path.read_bytes()
        originals.append(dict(name=path.name, bytes=len(raw), SHA256=f.H(raw)))
        return json.loads(raw)
    w = read(watch)
    assert w['completed'] and not w['passed'] and w['epoch'] == f.EPOCH and w['CPU_keeper'] == f.KEEPER
    f.collector_state(w)
    owners = [read(original/('owner-%02d.json'%i)) for i in range(16)]
    workers = [read(original/('worker-%d.json'%i)) for i in range(2)]
    spec = importlib.util.spec_from_file_location('frozen_scoring', source/'scoring.py')
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    pins = json.loads((source/'CPU_BINDING.json').read_bytes())['actual_project_DSO_SHA256']
    reports = []; keyhashes = set(); uids = set()
    for rank, q in enumerate(owners):
        r = dict(rank=rank, node=q.get('node'), PCI=q.get('physical_PCI'), identity_passed=False,
                 validated=[], frames=None, errors=[], producer_completed=completed_prefix(q['requests']),
                 producer_saved=len(q['requests']), normal_close=bool(q.get('normal_close')),
                 final_driver_sampler_present='driver_memory' in q)
        try:
            assert q['global_rank'] == rank and q['node'] == f.NODES[rank//8] and q['device'] == rank%8
            assert q['actual_GPU'] == f.GPU and q['gpu_epoch'] == f.EPOCH
            assert q['owner_TID'] == q['PID'] and q['UID'] == 154459
            assert q['source_manifest_SHA256'] == f.MF and q['GPU_manifest_SHA256'] == f.GMF
            assert q['Native_SHA256'] == f.NATIVE and q['private_input_SHA256'] == inp['private_plan']['SHA256']
            assert q['all78layers_loaded'] and q['loaded_parameters'] == q['checkpoint_key_count'] == 117060
            assert q['actual_layers'] == list(range(78))
            geo=q['constructor_geometry']; post=q['postload_geometry']
            assert post['loaded_parameters']==117060 and post['Model_READY']
            assert geo['TP']==geo['world_size']==16 and geo['PP']==1
            assert geo['world_rank']==geo['tp_rank']==rank and geo['rows']==1 and geo['capacity_per_row']==32768
            assert geo['layer_begin']==0 and geo['layer_end']==78 and geo['local_attention_heads']==4 and geo['local_index_heads']==2
            assert geo['embedding_vocab_begin']==rank*9680 and geo['embedding_vocab_end']==(rank+1)*9680
            assert geo['replicated_embedding_BF16_bytes']==0 and geo['embedding_BF16_local_bytes']==118947840
            assert geo['cache_logical_bytes']['sum']==8355053568
            assert set(q['CAPI_bindings'])=={'id_size','id','create','error','keys','load_batch','finish','forward','control','reset','geometry','destroy'}
            assert set(q['observer_CAPI_bindings'])=={'prepare','policy','next_decode','read'}
            assert all(v['SHA256']==f.NATIVE for v in [*q['CAPI_bindings'].values(), *q['observer_CAPI_bindings'].values()])
            assert set(q['actual_provider_maps'])==set(pins) and all(q['actual_provider_maps'][p]['SHA256']==h for p,h in pins.items())
            assert q['profile_getter_binding']['SHA256']==f.NATIVE and q['profile_getter_binding']['count']==6
            assert q['profile_getter_binding']['symbol']=='modern_glm_pipeline_last_profile'
            assert q['observer_pool_events']==512 and q['observer_source_event_bound']==250
            assert q['weights'] and not q['Model_qualified'] and not q['HTTP'] and q['observer_disabled_for_screen']
            assert not q.get('error')
            assert q['resets']==sum(bool(x.get('reset_before'))+bool(x.get('reset_after')) for x in q['requests'])
            assert all(x['floor']==f.FLOOR and x['free']>=f.FLOOR and x['total']==f.TOTAL for x in q['memory_boundaries'])
            boundaries={x['phase'] for x in q['memory_boundaries']}
            assert {'before_observer_prepare','after_observer_prepare'} <= boundaries
            common=q['original_common_CPUmask']; assert len(common)>=64 and len(set(common))==len(common)
            assert q['initial_owner_CPUmask']==common[(rank%8)*8:(rank%8)*8+4]
            keyhashes.add(q['checkpoint_keys_SHA256']);uids.add(tuple(q['UID_SHA256']))
            r['identity_passed']=True
            forward=0;controls=0
            for i in range(r['producer_completed']):
                a=f.schedule()[i];ci=a['case_index'];x=q['requests'][i]
                row,frames,fc,cc=f.request(x,plan['requests'][ci],inp['private_plan']['requests'][ci],a,i,rank,common,inp,forward)
                entry=next(e for e in q['events'] if e['phase']=='request.enter' and e['ordinal']==i)
                assert x['request_enter_ns']/1e9<=entry['monotonic']+1e-6
                if i==0:
                    assert q['observer_diagnostic']==x['observer_report'];r['frames']=frames
                else:
                    assert x['observer_mark_calls_before']==x['observer_mark_calls_after']==q['requests'][0]['observer_mark_calls_after']
                if rank==0:
                    score=scorer.grade(plan['requests'][ci]['case'],x['output_text'],natural_EOS=x['natural_EOS'],cap_hit=x['cap_hit'],unexpected_tool=False)
                    assert score==x['score'];row.update(correct=score['correct'],parse_failure=score['parse_failure'])
                forward+=fc;controls+=cc;r['validated'].append(row)
            assert q['Native_forward_attempts']>=q['Native_forward_completed']>=forward and q['control_calls']>=controls
            r.update(prefix_forward=forward,prefix_control=controls,observed_forward_completed=q['Native_forward_completed'],
                     observed_forward_attempts=q['Native_forward_attempts'],observed_controls=q['control_calls'],observed_resets=q['resets'])
        except Exception as e:
            r['errors'].append(type(e).__name__+': '+str(e)[:200])
        reports.append(r)
    cross_errors=[];n=0
    try:
        assert len(keyhashes)==len(uids)==1
        n=cross_prefix(reports,workers)
    except Exception as e:cross_errors.append(type(e).__name__+': '+str(e)[:200])
    post_clean=False;post_step=None;post_error=None
    try:post_step=f.audit_post(w,rows,owners,workers,strict=False);post_clean=True
    except Exception as e:post_error=type(e).__name__+': '+str(e)[:200]
    accepted=reports[0]['validated'][:n]
    screen=f.screen_summary([r for r in accepted if not r['diagnostic']])
    return dict(schema='GLM906-failed-prefix-v2',engineering_full_entry_passed=False,promote=False,
                GPU=f.GPU,keeper=f.KEEPER,actual_accounting=rows,planned_requests=33,screen32=screen,
                independently_validated_prefix=n,root_requests=accepted,rank_reports=reports,
                cross_errors=cross_errors,post_step=post_step,post_clean=post_clean,post_error=post_error,
                normal_destroy_owners=sum(bool(q.get('normal_close')) for q in owners),
                original_artifacts=originals,private_plan_SHA256=f.H(raw),source_manifest_SHA256=f.MF,
                reader_SHA256=f.H(Path(__file__).read_bytes()),
                limits=['Failed original GPU/keeper remain failed; complete33 and normaldestroy are unqualified.',
                        'Partial/missing requests remain in33/32 denominators; completed prefix is not cohort acceptance.',
                        'Frozen scorer reused; no new Native, model load, weights, SDK replay, Slurm or GPU execution.',
                        'Missing final driver sampler prevents sampler-floor qualification; API boundaries only.',
                        'Host/event windows overlap and include enqueue idle; not exclusive GPU busy/FLOPs.',
                        'No paired performance baseline, wide quality, formal NI or HTTP/client SLO qualification.'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('original','source','accounting','plan','watch','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();o=Path(a.output);assert not o.exists()
    v=main(a.original,a.source,a.accounting,a.plan,a.watch)
    with o.open('x',encoding='utf8') as out:json.dump(v,out,indent=2)
    print(json.dumps({k:v[k] for k in ('independently_validated_prefix','screen32','post_clean','cross_errors')}))
