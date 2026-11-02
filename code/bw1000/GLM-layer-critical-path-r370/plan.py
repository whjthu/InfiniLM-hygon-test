FIXTURES=1
def records(fixture):
 assert fixture==0
 yield from(dict(mode=mode,warm=True,pair=-1,arm=mode,rep=0)for mode in [0,1])
 for arm,mode in enumerate([0,1,1,0]):
  for rep in range(3):yield dict(mode=mode,warm=False,pair=0,arm=arm,rep=rep)
