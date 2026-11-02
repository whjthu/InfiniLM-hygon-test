"""Read-only, bounded dataset/cache inventory; no model imports or jobs."""
import hashlib
import json
import os
import pathlib

R = pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
OLD = pathlib.Path('/public/home/aclsylqidf/hygon-inference-20261013')
roots = [R / 'datasets', R / 'data', R / 'benchmarks', R / 'incoming',
         OLD / 'datasets', OLD / 'data', pathlib.Path.home() / '.cache/huggingface']
out = {'read_only': True, 'GPU': 0, 'imports_model': False, 'roots': [], 'matches': []}
for root in roots:
    out['roots'].append({'path': str(root), 'exists': root.exists()})
    if not root.exists():
        continue
    for base, dirs, files in os.walk(root):
        rel = pathlib.Path(base).relative_to(root)
        depth = len(rel.parts)
        dirs[:] = [x for x in dirs if x not in ('.git', 'node_modules')]
        if depth >= 4:
            dirs[:] = []
        for name in files:
            p = pathlib.Path(base) / name
            if not any(s in str(p).lower() for s in ('mmlu', 'gsm8k', 'quality-gsm', 'quality-bench')):
                continue
            st = p.stat()
            item = {'path': str(p), 'bytes': st.st_size}
            if st.st_size <= 20_000_000 and p.suffix.lower() in ('.json', '.jsonl', '.txt', '.csv', '.parquet', '.arrow'):
                item['SHA256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            out['matches'].append(item)
            if len(out['matches']) >= 500:
                out['truncated'] = True
                break
        if out.get('truncated'):
            break
    if out.get('truncated'):
        break
print(json.dumps(out))
