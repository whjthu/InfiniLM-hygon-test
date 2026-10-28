"""Bounded frontend scheduling only; no Runtime calls or background threads."""
def decode_wave_r9225(micros, forward, enabled=False):
 if type(enabled) is not bool:raise TypeError("deferred_feedback must be bool")
 groups=[list(rows) for rows in micros]
 flat=[i for rows in groups for i in rows]
 if any(not rows or len(rows)>32 for rows in groups) or len(flat)!=len(set(flat)) or any(type(i) is not int or not 0<=i<32 for i in flat):raise ValueError("independent stable logical rows required")
 width=2 if enabled else 1
 for begin in range(0,len(groups),width):
  pending=[]
  try:
   for rows in groups[begin:begin+width]:
    out,wall,tick,ended=forward(rows)
    pending.append((rows,out,wall,tick,ended))
    del out
   while pending:
    item=pending.pop(0)
    yield item
    del item
  finally:
   pending.clear()
