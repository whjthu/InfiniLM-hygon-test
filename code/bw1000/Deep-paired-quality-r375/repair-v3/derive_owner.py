"""Create a new immutable owner from the qualified366 bytes; never edit366."""
import ast
import hashlib
import json
import pathlib
import shutil

D = pathlib.Path(__file__).resolve().parent
G = D.parents[2]
BASE = G / 'code/bw1000/Deep-HC-T1-Model-r366/owner-v2'
OUT = D / 'owner'
assert not OUT.exists()
assert hashlib.sha256((BASE / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest() == '573b0e979a5da240a55169101c42755427eb1d67301c0e50be0c38a283e5610f'
mf = json.loads((BASE / 'SOURCE_MANIFEST.json').read_bytes())
for name, expected in mf['files_SHA256'].items():
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected, name
shutil.copytree(BASE, OUT)
for name in ['quality_plan.py', 'client_quality_pair.py']:
    shutil.copyfile(D / name, OUT / name)
card = json.loads((G / 'results/bw1000/2026-11-02/GLM373-preparation-card-v2.json').read_bytes())
refs = {'source_card_SHA256': '3412cb5e8c328f988954e8076185901a76c14cd63f7b308e11888e7b11562730',
        'cases': card['cases']['GSM8K'] + card['cases']['MMLU'],
        'shards': card['plans']['Deep']['shards'], 'sources': []}
remote = ['incoming/Kimi-heldout-resident-data-r9235/gsm8k-disjoint256.json',
          'incoming/Deep-semantic68-owner-r320/benchmark/mmlu-fixed32.json',
          'incoming/GLM-real-concurrency-r9132/quality-bench-mmlu-r9132.json']
for source, path in zip(card['sources'], remote):
    refs['sources'].append({'repo_path': source['path'], 'remote_relative': path, 'SHA256': source['SHA256']})
warm_path = 'code/bw1000/public-native-bridge/GLM-latest-full-owner-r290/benchmark/gsm8k-fixed32.json'
warm_bytes = (G / warm_path).read_bytes()
warm = json.loads(warm_bytes)['cases'][0]
refs['sources'].append({'repo_path': warm_path, 'remote_relative': 'incoming/Deep-semantic68-owner-r320/benchmark/gsm8k-fixed32.json',
                        'SHA256': hashlib.sha256(warm_bytes).hexdigest()})
canonical = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
refs['warm'] = {'source_file': warm_path, 'source_row': 0,
                'row_canonical_SHA256': hashlib.sha256(canonical(warm)).hexdigest(),
                'selection': 'First case of original frozen32, both arms; excluded from evaluation; no answer/length selection'}
(OUT / 'CASE_REFS.json').write_text(json.dumps(refs, indent=2, ensure_ascii=False) + '\n', encoding='utf8', newline='\n')


def change(name, replacements):
    p = OUT / name
    text = p.read_text(encoding='utf-8-sig')
    for old, new, expected in replacements:
        assert text.count(old) == expected, (name, old, text.count(old), expected)
        text = text.replace(old, new)
    p.write_text(text, encoding='utf8', newline='\n')


for name in ['bootstrap_r206b.py', 'preweight_r206b.py', 'native_decode_r206b.py']:
    text = (OUT / name).read_text()
    change(name, [('2580', '7080', text.count('2580')),
                  ('Deep-HC-T1-Model-r366-', 'Deep-paired-quality-r375-', text.count('Deep-HC-T1-Model-r366-'))])
change('client_owner_r281.py', [('Deep-HC-T1-Model-r366-', 'Deep-paired-quality-r375-', 1)])
text = (OUT / 'worker_r206b.py').read_text()
replacements = [
    ('Deep-HC-T1-Model-r366-', 'Deep-paired-quality-r375-', 2),
    ("'default_output_budget':128", "'default_output_budget':4096", 1),
    ('2580', '7080', 2),
    ("str(D/'client_chunk_pair.py')", "str(D/'client_quality_pair.py')", 1),
    ("from chunk_contract import descriptor_for_index", "from quality_plan import descriptor_for_index", 1),
    ("from chunk_contract import MEMORY_FLOOR_BYTES", "from quality_plan import MEMORY_FLOOR_BYTES", 1),
    ("assert len(report['requests'])<6", "assert len(report['requests'])<66", 1),
    ("assert actual_HTTP==6 and len(report['requests'])==6", "assert actual_HTTP==66 and len(report['requests'])==66", 1),
]
old = "report['prepared_entry_identity']=prepared['identity'];report['source_manifest_SHA256']=os.environ['DEEP_SOURCE_MANIFEST_SHA'];report['plan']={'HTTP':6,'Native':6,'phase_Native_end':{'prefill-pair':6}};report['Model_qualification_scope']='Frozen8489toolprompt, fixedchunk1024/output128,2warmOFFON thenHCmodeABBA; T1BF16HC hook only; no widequality/SLO/Modelgain inheritance';report['prefill_chunks_supported']=[1024,2048];report['prefill_schedule']=[1024]*6;report['HC_mode_schedule']=[0,1,0,1,1,0]"
new = "from quality_plan import encoding,plan,selected_shard\n enc=encoding();frozen_plan=plan();assert prepared['identity']==enc['prepared_entry_identity']\n report.update(prepared_entry_identity=prepared['identity'],comparison_contract=enc['comparison_contract'],replica=selected_shard(),source_manifest_SHA256=os.environ['DEEP_SOURCE_MANIFEST_SHA'],plan={'HTTP':66,'Native':66,'phase_Native_end':{'prefill-pair':66}},Model_qualification_scope='Custom0shot64GSM+64MMLU whole questions distributed4replicas, percaseOFFON/ONOFF balanced; not official benchmark/no regression/SLO proof',prefill_chunks_supported=[1024,2048],prefill_schedule=[1024]*66,HC_mode_schedule=[a['arm'] for a in frozen_plan],planned_Native=66,attempted_Native=0,completed_Native=0)"
replacements.append((old, new, 1))
old = "selected_binding=descriptor_for_index(b,len(report['requests']));chunk_size=selected_binding['prefill_chunk']"
new = "selected_binding=descriptor_for_index(b,len(report['requests']));chunk_size=selected_binding['prefill_chunk']\n   planned=selected_binding['planned_case'];encoded=enc['cases'][planned['case_id']];assert req.max_new_tokens==4096 and len(req.prompt_token_ids)==encoded['prompt_tokens'] and admission['history_render_audit']['Native_prompt_token_ids_SHA256']==encoded['prompt_IDs_SHA256']\n   assert hashlib.sha256(json.dumps(list(req.prompt_token_ids),separators=(',',':')).encode()).hexdigest()==encoded['prompt_IDs_SHA256']\n   assert admission['diagnostic_user']=='Deep375-r%d-i%d-mode%d'%(selected_shard(),len(report['requests']),planned['arm']);report['attempted_Native']+=1"
replacements.append((old, new, 1))
replacements.append(("if failure is None:req.finish(reason)", "if failure is None:report['completed_Native']+=1;req.finish(reason)", 1))
change('worker_r206b.py', replacements)
text = (OUT / 'GPU_inside_r277.sh').read_text()
change('GPU_inside_r277.sh', [('Deep-HC-T1-Model-r366/owner-v2', 'Deep-paired-quality-r375/owner', text.count('Deep-HC-T1-Model-r366/owner-v2')),
                              ('2670', '7140', 1)])
for p in OUT.rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'))
for p in OUT.rglob('*.sh'):
    assert b'\r' not in p.read_bytes(), p
manifest = {'schema': 1, 'parent_source_manifest_SHA256': hashlib.sha256((BASE / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
            'scope': 'Immutable finite66 per replica quality consumer; same976f/bootstrap/loader/forward/HC; full4096/32768',
            'files_SHA256': {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'SOURCE_MANIFEST.json'}}
(OUT / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8', newline='\n')
print(json.dumps({'completed': True, 'owner_manifest_SHA256': hashlib.sha256((OUT / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest(), 'files': len(manifest['files_SHA256'])}))
