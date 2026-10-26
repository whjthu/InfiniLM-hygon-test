"""CPU-only ctx2048 host contract. Does not loadNative/Runtime/weights."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Geometry:
 logical:int=8
 max_context:int=2048
 page_size:int=256
 reserved_blocks:int=4
 @property
 def physical(self):return max(16,2*self.logical)
 @property
 def recurrent_pool(self):return self.physical+1
 @property
 def pages_per_row(self):return self.max_context//self.page_size
 @property
 def pages(self):return self.pages_per_row*self.physical+self.reserved_blocks
 def validate(self):
  assert self.logical==8 and self.max_context==2048 and self.page_size==256 and self.reserved_blocks==4
  assert self.physical==16 and self.recurrent_pool==17 and self.pages_per_row==8 and self.pages==132
  return self
 def table(self,row):
  assert type(row)is int and 1<=row<=self.physical
  return list(range(1+(row-1)*self.pages_per_row,1+row*self.pages_per_row))
 def slot(self,row,position):
  assert type(position)is int and 0<=position<self.max_context
  pages=self.table(row);slot=pages[position//self.page_size]*self.page_size+position%self.page_size
  assert slot<self.pages*self.page_size;return slot
 def admit(self,prompt_tokens,output_cap):
  return type(prompt_tokens)is int and type(output_cap)is int and 1<=prompt_tokens<=1024 and 2<=output_cap<=1024 and prompt_tokens+output_cap<=self.max_context
FEATURE_WIDTH=35840
FEATURE_BYTES_PER_ROW=FEATURE_WIDTH*2
MAX_BODY=128<<20
MAX_PREFILL_ROWS=1024
MAX_VERIFY_WIDTH=8
MAX_FLAT_VERIFY=64

def feature_frame_bytes(rows):
 assert type(rows)is int and 1<=rows<=MAX_PREFILL_ROWS
 n=rows*FEATURE_BYTES_PER_ROW;assert n<=MAX_BODY;return n
