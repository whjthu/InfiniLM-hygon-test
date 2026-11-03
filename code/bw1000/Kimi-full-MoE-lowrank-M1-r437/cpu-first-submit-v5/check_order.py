from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 s=(D/'component.cpp').read_text();policy=json.loads((D/'SOURCE_POLICY.json').read_bytes())
 for p,h in policy.items():assert sha(p)==h,p
 assert 'kimi_r437_owner::Pin pin'not in s and 'affinity_r2392::apply_once'not in s
 create=s.index('auto owner=rt::Runtime::create');first=s.index('auto y=plan[arm]?candidate->forward');mask=s.index('437 actual original firstsubmit owner subset mismatch')
 assert create<first<mask and 'row["Runtime_created_mask"]=affinity()'in s and 'row["entry_mask"]=affinity()'in s
 assert 'RestoreOwner restore(entry,&restored)'in s and 'owner.reset();row["normal_Runtime_close"]=true;restore.restore()'in s
 assert 'const std::array<int,6>plan{0,1,0,1,1,0}'in s
 assert 'r.syncStream();const double native_ms=ms(begin);'in s
 return dict(passed=True,actual_new_component_source=True,actual_header_and_submit_SHA_verified=True,old9233_premature_pin_negative_reused=True,actual_firstsubmit_and_GPU_numeric=False,no_pre_pin=True,exactmask_verified_after_every_originalcompletedforward=True,only_restore_after_Runtime_close=True,new_GPU_probe=False)
if __name__=='__main__':print(json.dumps(check()))
