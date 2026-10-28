"""CPU-only derivative of already-published136b; no launch or parent mutation.

Root reviews/publishes the generated pins and writes new runtime GO afterward.
The exact original 12-case GSM32 matrix is retained, only private run labels move.
"""
from pathlib import Path
import argparse,hashlib,json
R=Path('/refactor');D=R/'incoming/Kimi-real-MoE-phase-r151';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--CPU-build-sha',required=True);a=p.parse_args();bp=R/'candidates/Kimi-real-MoE-phase-r151/CPU-build-recovered-r151a.json';assert sha(bp)==a.CPU_build_sha;b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen']
assert b['source_manifest_SHA256']==sha(D/'source-manifest-r151.json')
OLD=R/'incoming/Kimi-sparse-profile-Model-r136b';NEW=R/'incoming/Kimi-sparse-profile-Model-r151';assert not NEW.exists();producer=json.loads((OLD/'producer-source-r136b.json').read_bytes());assert sha(OLD/'producer-source-r136b.json')=='8d56b5d38e99f7229fd56a4682a17da07cc562b62392a8cc1bee7fa834f863cb'
NEW.mkdir()
for rel,h in producer['files_SHA256'].items():
 src=OLD/rel;assert sha(src)==h;out=NEW/rel.replace('r136b','r151');out.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes()
 if src.suffix in ['.py','.json','.sh']:raw=raw.replace(b'r136b',b'r151')
 out.write_bytes(raw)
receipt=NEW/'runtime/model-native-receipt-r9282.json';receipt.write_bytes(bp.read_bytes())
binding=NEW/'runtime/native_binding_r9143.py';text=binding.read_text();text=text.replace("names=['modern_kimi_model_events_configure_r8669zq'","names=['modern_kimi_moe_phase_mark_r151','modern_kimi_model_events_configure_r8669zq'");binding.write_bytes(text.encode())
summary=NEW/'runtime/summarize_decoder_events_r135.py';text=summary.read_text();anchor="    v=float(p['stream_span_ms']);";assert anchor in text;text=text.replace(anchor,"    if p['is_MoE'] and p.get('from_layer')==p['layer'] and (p['from']=='FFN.begin' or p['from'].startswith('MoE.')) and (p['name']=='FFN.end' or p['name'].startswith('MoE.')):name='MoE_routed_shared_TPwait'\n"+anchor);summary.write_bytes(text.encode())
mp=NEW/'runtime/model-manifest-r9183.json';m=json.loads(mp.read_bytes());m['native_DSO_SHA256']=b['native_SHA256'];m['loaded_library_SHA256'].pop('libKimi_W2_pair_r9280.so',None);m['loaded_library_SHA256'][Path(b['native']).name]=b['native_SHA256'];prefix='/refactor/incoming/Kimi-sparse-profile-Model-r151/'
for path in m['source_SHA256']:
 if path.startswith(prefix):m['source_SHA256'][path]=sha(NEW/path[len(prefix):])
mp.write_bytes((json.dumps(m,indent=2)+'\n').encode())
go=NEW/'launcher/GPU-GO-schema.json';v=json.loads(go.read_bytes());v['manifest_SHA256']=sha(mp);go.write_bytes((json.dumps(v,indent=2)+'\n').encode())
pinpath=NEW/'launcher/model-source-pins.json';pins=json.loads(pinpath.read_bytes());pins['manifest_SHA256']=sha(mp);pins['native_DSO_SHA256']=b['native_SHA256']
for path in pins['files']:
 if path.startswith(prefix):pins['files'][path]=sha(NEW/path[len(prefix):])
pinpath.write_bytes((json.dumps(pins,indent=2)+'\n').encode())

q={'source_only':True,'GPU':False,'GPU_submitted':False,'CPU_build_SHA256':sha(bp),'native_SHA256':b['native_SHA256'],'manifest_SHA256':sha(mp),'files_SHA256':{f.relative_to(NEW).as_posix():sha(f)for f in sorted(NEW.rglob('*'))if f.is_file()},'histogram':'DEFERRED','same_original12_question_matrix':True,'parent_model_manifest_SHA256':'99d845cc08d0b13569e7ccd0ea253c27945391709f9175ceae4e845c44b15a14','root_publication_GO_required':True}
out=NEW/'producer-source-r151.json';out.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'generated':True,'GPU':False,'runtime':str(NEW),'producer_SHA256':sha(out),'manifest_SHA256':sha(mp)}))
