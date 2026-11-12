"""Bound a framed file payload without consuming the following file header."""
class FileSlice:
 def __init__(self,stream,total):
  if type(total) is not int or total<=0:raise ValueError('invalid payload length')
  self.stream=stream;self.remaining=total
 def read(self,size):
  if type(size) is not int or size<=0:raise ValueError('explicit positive bounded read required')
  if not self.remaining:return b''
  requested=min(size,self.remaining);raw=self.stream.read(requested)
  if len(raw)>requested:raise ValueError('stream exceeded bounded read')
  self.remaining-=len(raw);return raw
