"""Materialize an immutable ordinary-entry overlay from exact frozen r454."""
from pathlib import Path
import argparse,hashlib,json
sha=lambda b:hashlib.sha256(b).hexdigest()
def build(parent,output,expected):
 parent,output=Path(parent),Path(output);raw=(parent/'SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==expected;m=json.loads(raw);output.mkdir(parents=True,exist_ok=False)
 for name,h in m['files_SHA256'].items():
  p=parent/name;assert sha(p.read_bytes())==h
  if name.startswith('public_deep_model/')or name in ['config.template.json','MODEL_PIN.json']:
   p2=output/name;p2.parent.mkdir(parents=True,exist_ok=True);p2.write_bytes(p.read_bytes())
 def change(name,old,new):
  p=output/'public_deep_model'/name;s=p.read_text();assert s.count(old)==1,(name,old);p.write_bytes(s.replace(old,new).encode())
 change('prepared_factory.py','class SelectedDeepBridge(TextDeliveryMixin, bridge_class):','from .token_clock import AdmissionMixin\n    class SelectedDeepBridge(AdmissionMixin, TextDeliveryMixin, bridge_class):')
 change('cli.py',"provider=DeepProvider(config,prepared)","provider=DeepProvider(config,prepared)\n from .token_clock import ClockServerMixin\n class ObservedServer(ClockServerMixin,Server):pass\n Server=ObservedServer")
 change('cli.py',"bind_public_sampling(server,prepared['engine']);return server,provider,prepared", "bind_public_sampling(server,prepared['engine'])\n server.token_clock_enabled=config.get('token_clock_enabled',False) is True\n def sink(q):\n  p=Path(config['runtime_directory'])/'token-clock-private.jsonl'\n  with p.open('a')as f:f.write(json.dumps(q)+'\\n')\n server.token_clock_sink=sink\n return server,provider,prepared")
 # Exact decode return hook before existing rc/finite/sample; no Native/math edit.
 change('native_decode.py','def sample(value):',"def observe_return(kind,past,count):\n  observer=getattr(req,'_token_clock',None)\n  if observer is not None:observer.returned(kind,past,count)\n def sample(value):")
 p=output/'public_deep_model/native_decode.py';s=p.read_text();assert s.count(';check_rc(code);finite_check(logits)')==2;s=s.replace('[0,len(part)]);check_rc(code)',"[0,len(part)]);observe_return('prefill',past,len(part));check_rc(code)");s=s.replace('[0,1]);check_rc(code)',"[0,1]);observe_return('decode',past,1);check_rc(code)");assert s.count("observe_return('prefill'")==1 and s.count("observe_return('decode'")==1;p.write_bytes(s.encode())
 change('native_decode.py','  return token',"  observer=getattr(req,'_token_clock',None)\n  if observer is not None:observer.sampled()\n  return token")
 change('incremental_text.py','        self.published = stable','        self.raw_last = text\n        self.published = stable')
 change('incremental_text.py','                # Decoding work',"                observer=getattr(req,'_token_clock',None)\n                if observer is not None:observer.text_progress(len(ids),not bool(delta),prefix.raw_last.endswith('\\ufffd'),final)\n                # Decoding work")
 (output/'public_deep_model/token_clock.py').write_bytes((Path(__file__).parent/'token_clock.py').read_bytes())
 m.update(schema='public-token-clock-overlay-r480',CPU_GO=False,GPU_GO=False,Model_GO=False,token_clock_default=False,parent_source_manifest_SHA256=expected)
 m['files_SHA256']={str(p.relative_to(output)).replace('\\','/'):sha(p.read_bytes())for p in output.rglob('*')if p.is_file()};(output/'SOURCE_MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('output');p.add_argument('--parent-sha',required=True);a=p.parse_args();build(a.parent,a.output,a.parent_sha)
