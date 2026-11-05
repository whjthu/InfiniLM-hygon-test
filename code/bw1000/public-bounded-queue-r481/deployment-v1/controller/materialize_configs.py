"""Generate the four exact source configs, not a run or a GPU claim."""
from pathlib import Path
import argparse,json
D=Path(__file__).resolve().parent
SLOTS=[('A-OFF','a02r4n01','A',False),('A-ON','a02r4n04','A',True),('B-OFF','a02r4n05','B',False),('B-ON','a02r4n07','B',True)]
def build(template,output=D):
 output=Path(output);target=output/'configs';target.mkdir(exist_ok=False);base=json.loads(Path(template).read_bytes());instances={}
 for name,node,pair,enabled in SLOTS:
  q=dict(base);q.update(execute_authorized=True,experimental_profile=None,runtime_directory='/refactor/results/Deep-bounded-queue-r481',node=node,Native_seconds=21000,service_seconds=20400,bounded_queue_enabled=enabled,physical_pending_capacity=2,queue_wait_seconds=5.,token_clock_enabled=True)
  p=target/(name+'.json');p.write_text(json.dumps(q,indent=2)+chr(10));instances[name]=dict(pair=pair,node=node,queue_policy=enabled,run=name+'-20261106-r481-v1',config='controller/configs/'+name+'.json')
 (output/'MATRIX.json').write_text(json.dumps(dict(instances=instances,cards=32,planned_HTTP=80,warm=0,phaseB_gate='BothAinstances firstfullserial request withownerreset+naturalEOS; not a qualityscoregate',schedule='A singlefiniteopenwave; B cancelwave4..11 thenSTOPbacklogwave12..19',budgets=dict(GPU=21600,Native=21000,service=20400,cleanup=22080,keeper=22200)),indent=2)+chr(10));return instances
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--template',required=True);a=p.parse_args();print(json.dumps(build(a.template)))
