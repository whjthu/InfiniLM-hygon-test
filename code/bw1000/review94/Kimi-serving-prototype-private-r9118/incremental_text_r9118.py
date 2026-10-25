"""Monotonic incremental tokenizer decode. Counts only committed non-special tokens."""
class IncrementalTextDecoder:
    def __init__(self,tokenizer,eos_ids=(),pad_ids=()):
        self.tokenizer=tokenizer;self.eos_ids=set(eos_ids);self.pad_ids=set(pad_ids);self.ids=[];self.printed=0;self.emitted='';self.finished=False
    def full_decode(self):
        return self.tokenizer.decode(self.ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    @staticmethod
    def cjk(c):
        n=ord(c);return 0x3400<=n<=0x4dbf or 0x4e00<=n<=0x9fff or 0xf900<=n<=0xfaff or 0x20000<=n<=0x323af or 0x3040<=n<=0x30ff or 0xac00<=n<=0xd7af
    def push(self,token_id,kind='committed',eos=False,final=False):
        if self.finished:raise RuntimeError('decoder already finished')
        token_id=int(token_id)
        if kind in ('draft','replay','pad'):return '',False
        if kind not in ('committed','EOS'):raise ValueError('unknown token event kind')
        eos=bool(eos or kind=='EOS' or token_id in self.eos_ids)
        counted=not eos and token_id not in self.pad_ids
        if counted:self.ids.append(token_id)
        text=self.full_decode()
        if not text.startswith(self.emitted):raise RuntimeError('tokenizer revised already emitted prefix; buffer policy insufficient')
        flush=eos or final
        if flush:end=len(text)
        elif text.endswith('\n')or(text and self.cjk(text[-1])):end=len(text)
        else:end=text.rfind(' ')+1
        if not flush:
            incomplete=text.find('\ufffd',self.printed)
            if incomplete>=0:end=min(end,incomplete)
        end=max(self.printed,min(len(text),end));piece=text[self.printed:end];self.printed=end;self.emitted+=piece
        if flush:self.finished=True
        return piece,counted
    def finish(self):
        if self.finished:return ''
        text=self.full_decode()
        if not text.startswith(self.emitted):raise RuntimeError('tokenizer revised emitted prefix at flush')
        piece=text[self.printed:];self.printed=len(text);self.emitted=text;self.finished=True;return piece
