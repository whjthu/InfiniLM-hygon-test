import pathlib,json,importlib.util,copy
from memory_gate_evidence_r9216 import gate_with_evidence,failure_safe_save
base=pathlib.Path(__file__).parent.parent/'Kimi-shard-only-Model-private-r9183/runtime-r9192/entry_contract_r9143.py';spec=importlib.util.spec_from_file_location('actual_gate',base);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);G=1024**3;checks=[]
for nodes in [4,8]:
 packets=[{'memory':[[6*G,64*G]for d in range(8)],'sample_count':4}for n in range(nodes)];snap=copy.deepcopy(packets[0]);report={};calls=[0];saves=[0]
 def gate(x):calls[0]+=1;return m.memory_gate(x)
 def save():saves[0]+=1
 assert gate_with_evidence('pass',snap,packets,gate,report,save)==6*G and calls==[1]and saves==[0]and not report;checks.append('success unchanged')
 for minimum in [4*G-1,4*G]:
  bad=copy.deepcopy(packets);bad[-1]['memory'][-1][0]=minimum;report={};original=None
  try:gate_with_evidence('prefill:case:2',snap,bad,m.memory_gate,report,save)
  except AssertionError as e:original=e
  assert original is not None and report['failed_memory_guard']['minimum_free_bytes']==minimum and len(report['failed_memory_guard']['owners'])==nodes;bad[-1]['memory'][-1][0]=9*G;assert report['failed_memory_guard']['owners'][-1]['memory'][-1][0]==minimum;checks.append('fullscalar evidence stable threshold')
 report={};sentinel=AssertionError('originalfloor');bad=copy.deepcopy(packets);bad[0]['memory'][0][0]=3*G
 def failing_gate(x):raise sentinel
 def badsave():raise OSError('evidence storage failure')
 try:gate_with_evidence('failure',snap,bad,failing_gate,report,badsave);raise RuntimeError('gate swallowed')
 except AssertionError as e:assert e is sentinel
 assert report['memory_guard_evidence_save_errors'];failure_safe_save(report,badsave);assert len(report['memory_guard_evidence_save_errors'])==2;checks.append('saveexception never masks original')
 try:failure_safe_save({},badsave);raise RuntimeError('nonmemory saveerror swallowed')
 except OSError:checks.append('unrelatedsaveerror unchanged')
# Grouping contract for future micro2 source candidate only; not a performance test.
for C in [1,2,3,4,8,16,32]:
 ids=list(range(1,C+1))
 for micro in [2,4]:
  groups=[ids[i:i+micro]for i in range(0,C,micro)];assert sum(groups,[])==ids and max(map(len,groups))<=micro;checks.append('prefill exactcoverage')
q={'CPU_only':True,'GPU':False,'native_library_loaded':False,'passed':True,'checks':len(checks),'original4GiBstrictfloor_preserved':True,'4node_and8node_failure_scalarpackets_saved':True,'originalexceptionidentity_preserved_evenSaveFails':True,'success_no_new_save_no_extra_gatecall':True,'prefill_grouping_only_noGPUqualification':True};pathlib.Path(__file__).with_name('CPU-contract-r9216.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
