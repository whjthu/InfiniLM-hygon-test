"""Exact retained teacher decode fixture, shared by admission and GPU consumer."""
import pathlib,hashlib
def load_fixture(root,fixture):
 root=pathlib.Path(root);rows=[]
 for item in fixture['files']:
  rel=pathlib.PurePosixPath(item['path']).relative_to('/refactor');p=root/rel;raw=p.read_bytes()
  assert len(raw)==item['bytes'],('fixture_bytes',str(p),len(raw));h=hashlib.sha256(raw).hexdigest();assert h==item['SHA256'],('fixture_SHA256',str(p),h);rows.append((p,raw,h))
 assert len(rows)==2 and rows[0][1]==rows[1][1],'frozen_rank0_rank6_input_disagree'
 assert fixture['shape']==[16,7168] and fixture['dtype']=='BF16' and len(rows[0][1])==16*7168*2
 return rows[0][1],{'files':[{'file':str(p),'bytes':len(raw),'SHA256':h}for p,raw,h in rows],'selected_shape':fixture['shape'],'selected_SHA256':rows[0][2],'scope':fixture['scope']}
