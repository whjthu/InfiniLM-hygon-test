from pathlib import Path
import base64,gzip,hashlib,importlib.util,json
import argparse
parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--old-bundle',type=Path,required=True);parser.add_argument('--private-output',type=Path);args=parser.parse_args()
A=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reader457',A/'analyze_archive.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
H=lambda b:hashlib.sha256(b).hexdigest();newfile=args.archive;archive=json.loads(gzip.decompress(newfile.read_bytes()));bundlefile=args.old_bundle;bundlefolder=bundlefile.parent;bundle=json.loads(bundlefile.read_bytes())
def newraw(suffix):
 matches=[row for name,row in archive['files'].items()if name.endswith(suffix)];assert len(matches)==1;row=matches[0];raw=base64.b64decode(row['base64']);assert H(raw)==row['SHA256']and len(raw)==row['bytes'];return raw
def oldraw(name):
 row=bundle['files'][name];raw=(bundlefolder/row['path']).read_bytes();assert H(raw)==row['SHA256']and len(raw)==row['bytes'];return raw
owner=json.loads(newraw('/owner.json'));cases=json.loads(newraw('/fixtures-r1/cases.json'))['cases'];previous={};oldclient=json.loads(oldraw('j0/client.json'));oldowner=json.loads(oldraw('j0/owner.json'))
assert owner['Native_SHA256']==oldowner['Native_SHA256']==module.NATIVE and owner['HC_mode']==oldowner['HC_mode']==0
oldwatch=json.loads(oldraw('j0/watch.json'));newwatch=json.loads(newraw('-watch.json'));assert oldwatch['actual_GPU']=='910374.1396'and newwatch['actual_GPU']=='910374.1413'and oldwatch['fresh_before'][0]['node']==newwatch['fresh_before'][0]['node']=='a02r4n04'
out=[];private=[]
for i,row in enumerate(owner['requests']):
 caseid=['gsm8k-r444-739','mmlu-r444-4335'][i];case=next(c for c in cases if c['id']==caseid);nr=newraw('/request-%d.request.json'%i);nw=newraw('/response-%d.wire'%i);nwire=module.wire(nw);ns=module.independent_grade(case,nwire['content'],True,nwire['tools']);ids=row['generated_token_ids']
 record=dict(case_id=caseid,new_index=i,new_mode=row['r454_mode'],new_tokens=len(ids),new_generated_IDs_SHA256=H(module.compact(ids)),new_score=ns,same_Native_a314=True,same_node='a02r4n04',cross_load=True,previous=[])
 for old in oldowner['requests']:
  if old['r450_case_id']!=caseid:continue
  logical=old['r450_logical_index'];oldbody=oldraw('j0/%04d.request.json'%logical);oldwire=module.wire(oldraw('j0/%04d.wire'%logical));oldids=old['generated_token_ids'];score=module.independent_grade(case,oldwire['content'],True,oldwire['tools'])
  assert nr==oldbody and row['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']==old['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']
  first=next((k for k,(x,y)in enumerate(zip(ids,oldids))if x!=y),None)
  if first is None and len(ids)!=len(oldids):first=min(len(ids),len(oldids))
  item=dict(mode=old['r450_mode'],old_owner_request_ordinal=oldowner['requests'].index(old),old_logical_index=logical,old_score=score,old_tokens=len(oldids),old_generated_IDs_SHA256=H(module.compact(oldids)),same_body_bytes=True,same_prompt_ID_hash=True,generated_IDs_equal=ids==oldids,first_generated_ID_difference=first,old_and_new_natural_EOS=True,old_and_new_no_cap=True,old_and_new_mode_matches=old['r450_mode']==row['r454_mode'],old_actual_owner_counters_present=len(old['r450_actual_owner_counts'])==8)
  record['previous'].append(item);private.append(dict(case_id=caseid,old_mode=old['r450_mode'],new_answer_text=nwire['content'],old_answer_text=oldwire['content'],new_IDs=ids,old_IDs=oldids,gold_numeric=case.get('gold_numeric'),gold_choice=case.get('gold_choice')))
 assert len(record['previous'])==2;out.append(record)
q=dict(passed=True,scope='Only original read-only samecase450OFF/ON versus457fixedON; not controlled performance or crossload stability',new_archive_SHA256=H(newfile.read_bytes()),old_bundle_SHA256=H(bundlefile.read_bytes()),rows=out,rootcause='UNKNOWN; source/profile startup and request history differ across loads. Bit/token differences alone do not identify numerical, state, parser or dispatch cause.',new_jobs=0,new_requests=0)
if args.private_output:args.private_output.write_bytes((json.dumps(private,ensure_ascii=False)+chr(10)).encode())
print(json.dumps(q,ensure_ascii=False,indent=2))
