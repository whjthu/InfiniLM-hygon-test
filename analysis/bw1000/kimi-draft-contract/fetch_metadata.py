from pathlib import Path
import json,hashlib,urllib.request,datetime,concurrent.futures
r=Path(__file__).parent
m=json.loads((r/'source-manifest.json').read_text())
def fetch(kv):
 n,v=kv
 try:
  with urllib.request.urlopen(v['url'],timeout=20)as f:b=f.read(200001)
  assert len(b)<200001
  (r/n).write_bytes(b)
  return n,dict(url=v['url'],bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),ref='main mutable; snapshot hash; exact training commit UNKNOWN')
 except Exception as e:return n,dict(url=v['url'],error=repr(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=7)as ex:m['sources']=dict(ex.map(fetch,m['sources'].items()))
m['checked_at_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(r/'source-manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
print(json.dumps(m,indent=2))
