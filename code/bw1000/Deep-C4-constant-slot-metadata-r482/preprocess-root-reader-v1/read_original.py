"""Independent original1112 M-text/receipt/current snapshot reconciliation.
No compiler, Native, SDK, Slurm, network, or Model execution. Three original
class-header byte streams are rehashed; other header hashes remain producer
snapshots rather than an independent live1755-header rehash.
"""
from pathlib import Path
import argparse,hashlib,json,shlex
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
HEADER='b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6'
RECEIPT='0664c51a0c9f0a7dabfe75de7edda607e44d55134f57672b1d00fe4d5a79ac5f'

def parse_original(text,target):
 text=text.replace('\\\r\n',' ').replace('\\\n',' ');names=set();lines=[s for s in text.splitlines()if s.strip()]
 assert lines
 for line in lines:
  left,colon,right=line.partition(':');assert colon and left.strip()==target
  names.update(shlex.split(right,posix=True))
 assert names and all(p.startswith('/')for p in names)
 return names

def read(private,receipt,accounting,repository):
 private=Path(private);assert H(receipt)==RECEIPT;q=json.loads(Path(receipt).read_bytes());assert q['actual_CPU']=='910407.1112'and q['completed']and q['passed']and q['completed_TUs']==q['planned_TUs']==50 and q['all50_fresh_M_complete']and q['known_historical_pins_match'];assert q['compile_commands']==q['link_commands']==q['SDK_explicit_calls']==0 and not q['Native_dlopen']and not q['Runtime_created']and not q['weights_opened']and not q['failures']and not q['changed_historical_dependencies']and not q['source_mismatches']and not q['class_header_variants']
 a=json.loads(Path(accounting).read_bytes());assert a['rc']==0;a=a['out'];a=json.loads(a)if isinstance(a,str)else a;rows=[x.split('|')for x in a['accounting']];assert len(rows)==1 and rows[0][0]=='910407.1112'and rows[0][2:4]==['COMPLETED','0:0']
 source=repository/'code/bw1000/Deep-C4-constant-slot-metadata-r482/preprocess-CPU-v1';assert H(source/'SOURCE_MANIFEST.json')==q['source_manifest_SHA256']=='b57bf40b4b5533f876d52d1ca690a008b86861b46bee755642cb9e392970402f';assert H(source/'CPU_RESOURCE_CLAIM.json')==q['claim_SHA256']=='8bede52f8621b7ca7dbb32cedb9787ba95446df3770df674b37d37000f45d56e'
 mf=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert all(H(source/n)==h for n,h in mf['files_SHA256'].items());recipe=json.loads((source/'ORIGINAL_TU_RECIPES.json').read_bytes());hist=json.loads((source/'HISTORICAL_DEPENDENCIES.json').read_bytes());records=recipe['objects'];assert len(records)==50
 index=json.loads((private/'original-index.json').read_bytes());assert len(index)==53 and len({r['name']for r in index})==53
 for row in index:assert H(private/row['name'])==row['SHA256']and(private/row['name']).stat().st_size==row['bytes']
 classfiles={r['path']:r['SHA256']for r in index if r['name'].endswith('.hpp')};assert len(classfiles)==3 and set(classfiles.values())=={HEADER}
 known=dict(hist['entries']);known.update({v['source']:v['source_SHA256']for v in records.values()});known_values=set(known.values());unmatched_occurrences=[];expected_consumers=[];missing_historical=[];units={};allpaths=set()
 for key,t in sorted(q['units'].items()):
  assert t['target']==key and t['rc']==0 and t['linked_object']in records;r=records[t['linked_object']]
  assert t['source']==r['source']and t['source_SHA256']==r['source_SHA256']and t['original_object_SHA256']==r['object_SHA256']and t['quote_include_origin']==r['quote_include_origin']
  md=private/(key+'.M');assert H(md)==t['MD_SHA256'];paths=parse_original(md.read_text(),key);assert paths==set(t['dependency_paths'])==set(t['fresh_dependencies_SHA256'])and t['source']in paths
  assert t['argv']==r['preprocess_prefix']+['-M','-MF',t['MD'],'-MT',key,t['source']]
  expected=dict(hist['entries'][i]for i in hist['entry_indices_by_linked_object'][t['linked_object']]);had_map=bool(expected)
  if not had_map:missing_historical.append(key)
  expected[t['source']]=t['source_SHA256'];headers=[]
  for path in sorted(paths):
   value=t['fresh_dependencies_SHA256'][path];assert value==q['snapshot_SHA256'][path]
   if path in expected:assert expected[path]==value
   else:unmatched_occurrences.append(dict(TU=key,linked_object=t['linked_object'],path=path,SHA256=value,expected=None))
   if value==HEADER:
    assert classfiles.get(path)==HEADER;headers.append(dict(path=path,SHA256=value))
  if headers:expected_consumers.append(dict(TU=key,linked_object=t['linked_object'],original_object_SHA256=t['original_object_SHA256'],source=t['source'],quote_include_origin=t['quote_include_origin'],class_headers=headers))
  allpaths.update(paths);units[key]=dict(linked_object=t['linked_object'],original_object_SHA256=t['original_object_SHA256'],source=t['source'],source_SHA256=t['source_SHA256'],quote_include_origin=t['quote_include_origin'],compiler_prefix=r['preprocess_prefix'],MD_SHA256=t['MD_SHA256'],dependency_paths=sorted(paths),old_full_dependency_map_present=had_map)
 assert allpaths==set(q['snapshot_SHA256'])and len(unmatched_occurrences)==len(q['unmatched_dependencies'])==5141
 assert {json.dumps(v,sort_keys=True)for v in unmatched_occurrences}=={json.dumps(v,sort_keys=True)for v in q['unmatched_dependencies']}
 assert expected_consumers==q['class_header_consumers']and len(expected_consumers)==5 and len(missing_historical)==14
 unknown={v['path']:v['SHA256']for v in unmatched_occurrences};same={p:h for p,h in unknown.items()if known.get(p)==h};other={p:h for p,h in unknown.items()if p not in known and h in known_values};new={p:h for p,h in unknown.items()if p not in known and h not in known_values};assert len(unknown)==len(same)+len(other)+len(new)
 affected={u['TU']for u in expected_consumers};compile_keys=sorted(affected|set(missing_historical));assert len(compile_keys)==19;fresh=[units[k]for k in compile_keys];host=sum(v['compiler_prefix'][0]=='g++'for v in fresh);assert host==7
 summary=dict(schema=1,actual_CPU='910407.1112',completed=True,passed_original50_preprocess=True,accounting=rows[0],elapsed_inside_s=q['elapsed_s'],receipt_SHA256=H(receipt),original53_index_SHA256=H(private/'original-index.json'),source_manifest_SHA256=q['source_manifest_SHA256'],claim_SHA256=q['claim_SHA256'],all50_M_text_sets_equal_receipt_and_snapshot=True,current_unique_dependency_paths=len(allpaths),original_class_header_copies_rehashed=3,class_header_SHA256=HEADER,class_header_consumers=expected_consumers,old_full_dependency_maps_present=36,old_full_dependency_maps_absent=14,unmatched_occurrences=5141,unmatched_unique_paths=len(unknown),same_path_matches_other_original_TU_pin=len(same),different_path_same_recorded_bytes=len(other),newly_observed_no_historical_pin=len(new),changed_known_historical_pins=0,no_historical_map_TUs=missing_historical,proposed_compile_TUs=compile_keys,proposed_compile_count=19,proposed_HOST_compile=7,proposed_GPU_source_compile=12,proposed_reuse_objects=31,actual_compile_commands=0,actual_link_commands=0,Native_CPU_GO=False,GPU_GO=False,Model_GO=False,notes=['Only5 consumers of the changed class, including newly found original5803 textModel TU; all original3 header copies match actual raw bytes.', 'The14 units lacking old full dep maps are not automatically qualified for reuse. Next Native build should fresh-compile those14 plus5 class consumers, rather than assume5/45.', '981 exact path pins from other originalTUs and331 equal-byte different paths are separate observations, not canonical inode proof or directory allowlists.', '106 newly observed byte/path pins remain a new current snapshot, not a reconstruction of old missing MDs. Record all exactly for next build.', 'Fresh source/deps/toolchain and live31object/DSO SHA and mirrored new-header closure must be checked in one new Native CPU claim before compilation.', 'Raw originalM text and three class headers independently consumed; current1755header bytes, binary ABI, SDK/device/numerics/performance not independently replayed.'])
 return summary,dict(schema=1,actual_CPU='910407.1112',receipt_SHA256=RECEIPT,source_manifest_SHA256=q['source_manifest_SHA256'],path_SHA256=q['snapshot_SHA256'],units=units,class_header_consumers=expected_consumers,new_current_unhistorical_paths=new,proposed_compile_TUs=compile_keys)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('private',type=Path);p.add_argument('receipt',type=Path);p.add_argument('accounting',type=Path);p.add_argument('repository',type=Path);p.add_argument('summary',type=Path);p.add_argument('snapshot',type=Path);a=p.parse_args();r,s=read(a.private,a.receipt,a.accounting,a.repository);r['reader_SHA256']=H(__file__);a.summary.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');a.snapshot.write_text(json.dumps(s,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps({k:r[k]for k in ['actual_CPU','passed_original50_preprocess','current_unique_dependency_paths','proposed_compile_count','proposed_HOST_compile','proposed_GPU_source_compile','proposed_reuse_objects']}))
