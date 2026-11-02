"""Small metadata only, after immutable335 harvest has checked actual BF16 data."""
def validate(packets,epoch):
 expected={0:[19,37],1:[],2:[66,78,90],3:[]}
 assert len(packets)==4 and {v['node']for v in packets}==set(range(4))
 for value in packets:
  assert value['layerIDs']==expected[value['node']]and value['rows']==512 and value['epoch']==epoch
  assert value['finite_checked_by_immutable_harvest']is True
  assert value['local_bytes']==len(expected[value['node']])*512*7168*2
 assert sum(v['local_bytes']for v in packets)==512*35840*2
 return dict(passed=True,arm_ordinal=2,layerIDs=[19,37,66,78,90],rows=512,bytes=512*35840*2,finite=True,routed_leaders=[0,2],route='node2 to node0 independent capture socket; node1/node3 zero tap records',owners=packets,raw_features_saved=False)
