import datetime
import hashlib
import json
import pathlib

r = pathlib.Path.home() / 'hygon-inference-refactor-20261014'
w = json.loads((r / 'Kimi-Model-events-r8669zq-watch.json').read_text())
assert w['terminal']['JobID'] == '910375.654'
out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'CPU_only': True, 'GPU': False, 'actual_handle': '910375.654',
       'scope': 'Original completed reports memory/cache numeric metadata only; no weights or activations, no new GPU admission.',
       'nodes': []}
for node in range(4):
    p = r / 'results' / f"kimi-Model-events-r8669zq-{w['run']}-node{node}.json"
    raw = p.read_bytes()
    q = json.loads(raw)
    out['nodes'].append({'node': node, 'report_SHA256': hashlib.sha256(raw).hexdigest(),
                         'cache_geometry': q['cache_geometry'],
                         'memory_fields': {k: v for k, v in q.items() if 'memory' in k.lower()},
                         'compact_workspace_constructor_counts': q.get('compact_workspace_constructor_counts')})
print(json.dumps(out))
