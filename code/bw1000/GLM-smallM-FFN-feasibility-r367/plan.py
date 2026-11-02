FIXTURES=2
ORDERS=((0,1),(1,0),(0,1),(1,0))
REPS=3
def records(fixture):
 assert 0<=fixture<FIXTURES
 yield from(dict(mode=mode,warm=True,pair=-1,arm=mode,rep=0)for mode in [0,1])
 for pair,order in enumerate(ORDERS):
  for arm,mode in enumerate(order):
   for rep in range(REPS):yield dict(mode=mode,warm=False,pair=pair,arm=arm,rep=rep)
