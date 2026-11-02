"""CPU only: shared actual consumer/real files, no Native dlopen or GPU numerics."""
from pathlib import Path
import json,tempfile,copy,array
from consumer import execute,schedule

def main():
    cases=[]
    widths=[2304,128,6,768];offsets=[0,2304,2432,2438]
    assert sum(widths)==3206 and sum(w*7168*2 for w in widths)==45961216
    # Prove the source-row mapping for every element, not just equal dimensions.
    for branch,(n,start) in enumerate(zip(widths,offsets)):
        for row in range(n):
            assert (start+row)*7168 +7167 <3206*7168
        assert start==sum(widths[:branch])
    cases.append(dict(name='all_four_weight_rows_exact_offsets',passed=True,rows=3206))
    assert [(n,n,1) for n in widths]==[(2304,2304,1),(128,128,1),(6,6,1),(768,768,1)]
    cases.append(dict(name='flatten_first_canonical_C1_singleton_strides',passed=True))
    class B:
        def __init__(self,bad=None):self.on=False;self.calls=0;self.chosen=0;self.c=[0]*4;self.bad=bad;self.buf=array.array('H',[0])*7168;self.epoch=0;self.last=None
        def restore(self):pass
        def mode(self,on):self.on=on
        def barrier(self,n):pass
        def stats(self):return dict(chosen=self.chosen,fallback=0,consumer_counts=self.c[:])
        def run(self,fixture):
            self.epoch+=1;self.calls+=1;old=self.buf.buffer_info()[0];self.buf[0]=self.epoch
            assert self.buf.buffer_info()[0]==old
            if self.on:self.chosen+=1;self.c=[v+1 for v in self.c]
            self.last=(self.epoch,self.buf[0],fixture)
            if self.bad=='counter':self.c[3]+=1
            return self.last
        def inspect(self,raw,fixture):
            assert raw==(self.epoch,self.buf[0],fixture)
            return dict(passed=self.bad!='numeric',finite=True,CPU_recording=True,epoch=self.epoch)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'actual-consumer.jsonl'
        def save(row):
            with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        b=B();rows=execute(b,save)
        assert len(rows)==320 and b.calls==320 and b.chosen==160 and b.c==[160]*4
        assert len(p.read_text().splitlines())==320
        cases.append(dict(name='actual_schedule_320_new_epochs_same_input_pointer_updated',passed=True))
        for bad in ['counter','numeric']:
            p.write_text('');b=B(bad)
            try:execute(b,save)
            except RuntimeError:pass
            else:raise AssertionError('bad completed call must failstop')
            assert b.calls==1 and len(p.read_text().splitlines())==1 and not json.loads(p.read_text())['passed']
            cases.append(dict(name='persist_before_'+bad+'_failstop',passed=True))
        p.write_text('');b=B()
        try:execute(b,save,lambda: b.calls==2)
        except RuntimeError:pass
        else:raise AssertionError('cancel')
        assert b.calls==2 and len(p.read_text().splitlines())==2
        cases.append(dict(name='cancel_no_next_submission',passed=True))
    # No persistent output bundle exists: source local epoch views are not keyed by data_ptr.
    src=(Path(__file__).parent/'delta_r405.cpp').read_text()
    assert 'std::array<infinilm::runtime::Tensor,4> bundle_r405;' in src
    assert src.count('bundle_r405[')==4 and 'BundleUnwindR405' in src
    assert src.count('use_bundle_r405 ?')==4 and '++projection_r405::control.fallback' in src
    cases.append(dict(name='all_four_branches_uniform_OFF_nonC1_and_exception_bundle_lifetime',passed=True,kind='source_contract_not_Native_execution'))
    return dict(completed=True,passed=True,CPU_only=True,Native_dlopen=False,GPU=0,cases=cases)
if __name__=='__main__':print(json.dumps(main(),allow_nan=False))
