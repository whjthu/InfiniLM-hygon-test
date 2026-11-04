"""Rebuild the exact private bundle from14 immutable gzip parts, no execution."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json
H=lambda b:hashlib.sha256(b).hexdigest()
def materialize(parts_dir,metadata,output):
    source=Path(parts_dir);m=json.loads(Path(metadata).read_bytes());base=Path(output);base.mkdir(exist_ok=False);objects=base/'objects';objects.mkdir();files={}
    for pin in m['parts']:
        packed=(source/(pin['phase']+'.json.gz')).read_bytes();assert len(packed)==pin['bytes']and H(packed)==pin['SHA256'];plain=gzip.decompress(packed);assert H(plain)==pin['uncompressed_SHA256'];part=json.loads(plain);assert part['phase']==pin['phase']and len(part['files'])==pin['files']
        for name,entry in part['files'].items():
            raw=base64.b64decode(entry['base64'],validate=True);assert len(raw)==entry['bytes']and H(raw)==entry['SHA256'];path=objects/entry['SHA256']
            if path.exists():assert path.read_bytes()==raw
            else:
                with path.open('xb')as f:f.write(raw)
            value=dict(path=path.relative_to(base).as_posix(),bytes=len(raw),SHA256=H(raw),original_path=entry['original_path'])
            if name in files:assert files[name]==value
            else:files[name]=value
    q=dict(schema_version=1,provider_source_SHA256='822f91e71088cf6c63828f442b956ab6d59744becdbf23433b8ddeef2769f4bb',deployment_manifest_SHA256='c383fe62814f899071927dd5b1eda906166a55a3a88f81618f46d323fad2f2e7',supplement_manifest_SHA256='c110e98e982a2e47c6a45b19f7cc6668bce39052afacaa17eb4f6c6beb579458',diagnostic_recording_only=False,attempt_lineage_explicit=True,files=files,original_parts=m['parts'],new_HTTP=0,new_jobs=0)
    raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode();assert H(raw)==m['bundle_SHA256'];(base/'bundle.json').write_bytes(raw)
    return dict(bundle_SHA256=H(raw),raw_files=len(files),jobs=0,HTTP=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parts-dir',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(materialize(a.parts_dir,a.metadata,a.output_dir)))
