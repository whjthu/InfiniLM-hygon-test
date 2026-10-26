import pathlib,json,sys
q=json.loads(pathlib.Path(sys.argv[1]).read_bytes());assert q['completed']and q['normal']and len(q['owners'])==8
for o in q['owners']:
 assert o['normal_owner_closed']and len(o['measurements'])==28 and len(o['boundary_commit_followups'])==4
 for m in o['measurements']:
  assert m['finite']and m['canary']and m['readonly']and m['host_profile']is None
  assert not m['counts']['readback'][1]and m['counts']['bank'][0]==int(m['fixture']in['realB32','trueTopkBoundary32'])
 for m in o['boundary_commit_followups']:assert m['accepted']
for f in q['fixtures'][:2]:assert len(f['performance']['matched_iterations'])==8
print(json.dumps({'qualified':True,'step':q['step'],'level':'WHOLE_ACTUAL_CSA_BANK_COPY_COMPONENT_ONLY','Model_gain_claim':False,'checkpoint_layer':16}))
