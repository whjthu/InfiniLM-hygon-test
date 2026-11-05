"""Global128 scale/layout arithmetic only, not GPU or Model qualification."""
import json

def check():
 tiles=[]
 for rank in range(16):
  offset=rank*128;rows=list(range(offset,offset+128));assert len(rows)==128 and max(rows)<2048;tiles+=rows
  gate_scales={(offset+row)//128 for row in range(128)};down_scales={offset//128+b for b in range(1)};assert gate_scales==down_scales=={rank};assert max(gate_scales)<16
  assert 256*128*6144==256*6144*128
 assert tiles==list(range(2048));return dict(passed=True,ranks=16,logical_I=128,physical_I=128,scale_global_blocks=16,scale_blocks_per_route=1,expert_count=256,GPU=False,Native=False)
if __name__=='__main__':print(json.dumps(check()))
