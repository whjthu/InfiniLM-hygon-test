"""Fixed three-arm reversals; no parameter sweep or result-dependent selection."""
ROWS=512
WIDTH=64
CLUSTERED_ROW_ORDER=[3,11,7,15,1,5,9,13,4,12,2,10,6,14,0,8]
ORDERS=((0,1,2,2,1,0),(2,0,1,1,0,2),(1,2,0,0,2,1),(0,2,1,1,2,0))
REPS=2
def records():
 for fixture in range(2):
  for strategy in range(3):yield dict(fixture=fixture,warm=True,block=-1,phase=strategy,strategy=strategy,rep=0)
  for block,order in enumerate(ORDERS):
   for phase,strategy in enumerate(order):
    for rep in range(REPS):yield dict(fixture=fixture,warm=False,block=block,phase=phase,strategy=strategy,rep=rep)
def row_order(fixture):
 if fixture==0:return list(range(16))*32
 if fixture==1:return [i for i in CLUSTERED_ROW_ORDER for _ in range(32)]
 raise ValueError('only two frozen geometry fixtures')
