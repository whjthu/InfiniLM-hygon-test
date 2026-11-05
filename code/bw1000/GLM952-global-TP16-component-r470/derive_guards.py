"""Only exact pinned source derivation; never execute/import generated Model source."""
from pathlib import Path
import argparse,hashlib,json

def derive(mapping,output):
 spec=json.loads((Path(__file__).parent/'GUARD_DERIVATION.json').read_bytes());output=Path(output);output.mkdir(exist_ok=False);result={}
 for r in spec['records']:
  p=Path(mapping[r['source']]);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==r['SHA256'];text=b.decode();assert text.count(r['old'])==1
  new=('#include "geometry.hpp"\n'+text.replace(r['old'],r['new'])).encode();target=output/(r['name']+'.cpp');target.write_bytes(new);result[r['name']]=dict(parent_SHA256=r['SHA256'],derived_SHA256=hashlib.sha256(new).hexdigest())
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source_map');p.add_argument('output');a=p.parse_args();print(json.dumps(derive(json.loads(Path(a.source_map).read_bytes()),a.output)))
