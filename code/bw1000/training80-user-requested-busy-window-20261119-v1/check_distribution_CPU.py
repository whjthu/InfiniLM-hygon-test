"""Only the newly requested distribution, no sysfs or old collection fixtures."""
import json
from distribution import describe

owners={'a':('1087656','T01'),'b':('1103606','I01')}
pins={n:['0000:%02x:00.0'%i for i in range(8)] for n in owners}
def row(node,value):
    return {'rows':[dict(point=i,opened_epoch=100+2*i,closed_epoch=100+2*i+.1,busy={p:value for p in pins[node]}) for i in range(3)]}
full=describe(owners,pins,{'a':row('a',0),'b':row('b',100)},100)
s=full['overall']
assert (s['planned_cards'],s['observed_cards'],s['zero_cards'],s['over80_cards'],s['missing_cards'])==(16,16,8,8,0)
assert (s['window_mean_min'],s['window_mean_median'],s['window_mean_max'])==(0,50,100)
partial=describe(owners,pins,{'a':row('a',80)},100)
assert partial['overall']['missing_cards']==8 and partial['overall']['over80_cards']==0
assert partial['partitions']['inference']['missing_cards']==8
print(json.dumps(dict(full_distribution=1,missing_denominator_and_strict80=1,actual_sampling=False,SSH=False)))
