"""Bounded runtime fragmentation of a complete prompt, never truncation."""
def fragments(ids,limit):
 if type(limit)is not int or limit not in (512,1024,2048,4096)or not ids:raise ValueError('nonempty complete prompt and declared chunk required')
 offset=0
 while offset<len(ids):
  chunk=ids[offset:offset+limit]
  if not chunk:raise ValueError('empty fragment')
  yield offset,chunk
  offset+=len(chunk)
