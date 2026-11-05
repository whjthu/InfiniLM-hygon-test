"""Run old and repaired production collectors on the same sealed actual1476 files."""
import argparse,base64,gzip,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path,PurePosixPath
D=Path(__file__).resolve().parent
ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
SEAL='4233bec10c6906d0190fc0e519c0fbd38940282fde635e93422d89d68a2bc118'
OLD='83784ca11b057cb3bde2fce475f4868fd3eeba57a1452e5ef2184b34c6930952'
NEW='7803687eab94cdca74392f1c0e4be34cb3e070c59f1218509a0b07a234cc0743'
ENTRY='458748fc82a48a0472677f8b8d93e9fec0713a9dab12339548ea574a6135d447'
WRONG='2d73637f73ec91cd1b1e20c9a188be4b4171ced7e3b7f03eb28671d9f11aaafa'
ROOT_AUDIT='6fb1cb047db6504c6264fd7d6b2b6bc39b655a82bdd5c091047ebdaa08908fbe'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def run(archive,root_audit):
 raw=Path(archive).read_bytes();assert sha(raw)==SEAL;s=json.loads(gzip.decompress(raw));files={}
 for f in s['files']:
  b=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True));assert sha(b)==f['SHA256']and len(b)==f['bytes'];assert f['path']not in files;files[f['path']]=b
 assert len(files)==69;rb=Path(root_audit).read_bytes();assert sha(rb)==ROOT_AUDIT;root=json.loads(rb);assert root['passed']and root['original_outcome']['keeper']=='910407.1035 FAILED1'
 def one(suffix):
  matches=[(p,b)for p,b in files.items()if p.endswith(suffix)];assert len(matches)==1;return matches[0]
 old_path,old=one('/hc-pre-deployment-v2/collect.py');support_path,support=one('/hc-pre-deployment-v2/support.py');consume_path,consume=one('/hc-pre-entry-v1/hc_consume.py');ep,eb=one('/entry-report.json');entry=json.loads(eb);closed=json.loads(one('-GPU_PROCESS_CLOSED.json')[1]);watch=json.loads(one('-watch.json')[1]);new=(D/'collect.py').read_bytes();assert sha(old)==OLD and sha(new)==NEW and sha(eb)==ENTRY and closed['entry_report_SHA256']==ENTRY
 assert sha(support)=='cdf078222743f78e089b758038d08edd7c60e0e53314be75ebbdb3eaa53be1a2'
 replacements=[(b"output=Path(output);p=output/'entry-report.json';q=json.loads(p.read_bytes())",b"output=Path(output);entry_report_path=output/'entry-report.json';q=json.loads(entry_report_path.read_bytes())"),(b"hc=w['HC_pre_split'];p=actual(hc['packet']['path']);b=p.read_bytes()",b"hc=w['HC_pre_split'];hc_packet_path=actual(hc['packet']['path']);b=hc_packet_path.read_bytes()"),(b"scalars.append(dict(path=str(p),bytes=len(b),SHA256=sha(p)))",b"scalars.append(dict(path=str(hc_packet_path),bytes=len(b),SHA256=sha(hc_packet_path)))"),(b"return dict(entry_report_SHA256=sha(p)",b"return dict(entry_report_SHA256=sha(entry_report_path)")]
 expected=old
 for a,b in replacements:assert expected.count(a)==1;expected=expected.replace(a,b)
 assert expected==new
 # Relocate immutable files only. The actual support.actual function retains its
 # original implementation; its filesystem root R is redirected to a private
 # extraction because original POSIX mount paths are unavailable on Windows.
 # Neither collector, JSON, nor hc_consume is mocked or substituted.
 with tempfile.TemporaryDirectory(prefix='Deep467-collector-original1476-')as tmp:
  tmp=Path(tmp);targets={p for p in files if p==ep or '/scalar-events/'in p}|{old_path,support_path,consume_path}
  for p in targets:
   assert p.startswith(ROOT+'/');q=tmp/PurePosixPath(p).relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(files[p])
  parent=tmp/PurePosixPath(old_path).relative_to(ROOT).parent;supportmodule=load('support',parent/'support.py');saved={n:sys.modules.get(n)for n in['support','hc_consume']};savedpath=list(sys.path)
  try:
   supportmodule.R=tmp;sys.modules['support']=supportmodule;sys.modules.pop('hc_consume',None)
   oldmodule=load('old_actual1476_collector',parent/'collect.py');newmodule=load('repaired_actual1476_collector',D/'collect.py');output=tmp/PurePosixPath(ep).relative_to(ROOT).parent
   oldresult=oldmodule.collect(output,entry['identity']['actual_step'],entry['identity']['gpu_start_epoch']);fixed=newmodule.collect(output,entry['identity']['actual_step'],entry['identity']['gpu_start_epoch'])
   assert oldresult['entry_report_SHA256']==WRONG and fixed['entry_report_SHA256']==ENTRY
   assert {k:v for k,v in oldresult.items()if k!='entry_report_SHA256'}=={k:v for k,v in fixed.items()if k!='entry_report_SHA256'}
   assert oldresult['entry_report_SHA256']!=closed['entry_report_SHA256']and fixed['entry_report_SHA256']==closed['entry_report_SHA256']
   # Preserve complete returned data with extraction path names normalized back
   # to original /refactor paths; values and packet hashes remain actual.
   def normalized(v):
    if isinstance(v,dict):return{k:normalized(x)for k,x in v.items()}
    if isinstance(v,list):return[normalized(x)for x in v]
    if isinstance(v,str):
     try:return '/refactor/'+Path(v).relative_to(tmp).as_posix()
     except ValueError:return v
    return v
   oldresult,fixed=normalized(oldresult),normalized(fixed)
  finally:
   sys.path[:]=savedpath
   for n,v in saved.items():
    if v is None:sys.modules.pop(n,None)
    else:sys.modules[n]=v
 return dict(schema='Deep467-production-collector-variable-repair-v3-regression',passed=True,actual_old_production_collect_executed=True,actual_new_production_collect_executed=True,actual_hc_consume_executed=True,recording_or_fabricated_JSON=False,archive_SHA256=SEAL,verified_files=69,original_collector_SHA256=OLD,repaired_collector_SHA256=NEW,root_independent_audit_SHA256=ROOT_AUDIT,old_footer_equal=False,repaired_footer_equal=True,only_output_difference='entry_report_SHA256',all_other_terminal_output_data_equal=True,old_output=oldresult,repaired_output=fixed,original_keeper_failed=True,original_watch_passed=watch['passed'],original_forced_cleanup=watch['forced_cleanup'],original_reports_unchanged=True,path_relocation_only='Original support module with R redirected to temporary extraction; no collector/consumer/body/token/time fields mocked.',Native_dlopen=False,SDK_Runtime=False,Model=False,GPU=False,new_jobs=0,new_HTTP=0)
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--root-audit',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=run(a.archive,a.root_audit);Path(a.output).write_text(json.dumps(q,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps({k:q[k]for k in['passed','actual_old_production_collect_executed','actual_new_production_collect_executed','all_other_terminal_output_data_equal','old_footer_equal','repaired_footer_equal','new_jobs','GPU']}))
if __name__=='__main__':main()
