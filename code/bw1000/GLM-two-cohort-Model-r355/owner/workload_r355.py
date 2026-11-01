"""Two unchanged private requests/official IDs; no payloads are published."""
from pathlib import Path
import hashlib,json
PLAN=(False,True,False,True,True,False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def load(folder):
    from workload_r349 import body_and_gold
    card=json.loads((Path(folder)/'FIXTURE_CARD.json').read_bytes())
    assert [x['id']for x in card['fixtures']]==['short-read','anchor-calculate']
    rows=[]
    for row,f in enumerate(card['fixtures']):
        raw,body,gold=body_and_gold(f)
        p=Path(f['current_prepared_prompt_IDs']);encoded=p.read_bytes()
        assert sha(encoded)==f['current_prepared_private_IDs_file_SHA256']
        q=json.loads(encoded);ids=q['prompt_token_ids']
        assert len(ids)==f['current_prepared_prompt_tokens'] and all(type(x)is int and 0<=x<154880 for x in ids)
        assert sha(json.dumps(ids,separators=(',',':')).encode())==f['current_prepared_prompt_IDs_SHA256']
        assert len(ids)+4096<=32768
        rows.append(dict(row=row,fixture=f,body=body,raw=raw,gold=gold,ids=ids))
    assert [len(x['ids'])for x in rows]==[479,8900]
    return rows


def marker_identity(run,index,manifest):
    return dict(run=run,pair_index=index,source_manifest_SHA256=manifest,
                requests=2,ordinary_budget=4096,rows=2,context=32768)
