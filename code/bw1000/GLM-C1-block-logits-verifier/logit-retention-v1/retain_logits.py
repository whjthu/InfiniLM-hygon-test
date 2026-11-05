"""Private BF16 files only. Called on original owner after the existing drain."""
import hashlib,os,re,sys

def make_retainer(run,torch,stage,rank):
    if stage not in(0,1)or rank not in range(8)or sys.byteorder!='little':raise ValueError('retention owner/endianness')
    def retain(label,output,metadata):
        if not re.fullmatch(r'(serial-proposal-[0-7]|block-proposal|serial-continuation-[01]|candidate-continuation-[01])',label):raise ValueError('unexpected retention label')
        if stage==0:
            if output is not None:raise ValueError('stage0 has no tail logits')
            return None
        width=8 if label=='block-proposal'else 1
        if output.dtype!=torch.bfloat16 or list(output.shape)!=[width,154880]:raise ValueError('actual BF16 logits shape/dtype differs')
        cpu=output.detach().cpu().contiguous()
        raw=cpu.view(torch.int16).numpy().tobytes(order='C')
        if len(raw)!=width*154880*2:raise ValueError('BF16 raw span differs')
        stem='retained-%d-%d-%s'%(stage,rank,label);path=run.path/(stem+'.bf16');temp=run.path/(stem+'.writing-'+str(os.getpid()))
        with temp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        try:os.link(temp,path)
        finally:temp.unlink()
        value=dict(label=label,stage=stage,rank=rank,path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),dtype='BF16',shape=[width,154880],endianness='little',metadata=metadata,scope='Actual already-computed tail logits; original owner output strong reference until synchronous CPU copy; private R/L only')
        run.write(stem+'.json',value)
        return value
    return retain
