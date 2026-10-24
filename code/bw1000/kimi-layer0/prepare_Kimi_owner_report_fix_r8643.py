from pathlib import Path
import ast
r=Path('.');s=(r/'kimi_embedding_layer0_runner_r8636.py').read_text();s=s.replace('import numpy as np','import numpy as np')
s=s.replace('from kimi_native_state_consumer_r8519 import read_snapshot','from kimi_native_state_consumer_r8519 import read_snapshot\nfrom kimi_owner_report_r8643 import OwnerSnapshotReport')
old=""";lock=threading.Lock()
 def save():
  with lock:tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
 save();weights=[];"""
new="""
 reporter=OwnerSnapshotReport(path,result);reporter.flush();weights=[];"""
assert old in s;s=s.replace(old,new)
s=s.replace(" lib=ctypes.CDLL(m['runtime_library']);", " reporter.publish_top(weight_sources=result['weight_sources'])\n lib=ctypes.CDLL(m['runtime_library']);")
s=s.replace("torch.cuda.set_device(rank);ctx=None;own=result['workers'][rank]", "torch.cuda.set_device(rank);ctx=None;own={'rank':rank,'normal_close':False}\n  def save():reporter.publish_worker(rank,own)")
s=s.replace("os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/label);result['phase']='capture-'+label", "os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/label);reporter.publish_top(phase='capture-'+label)")
# Final summary consumes a detached deep snapshot only after all owner futures ended.
old=""" result.update(phase='completed',all8normal=all(w['normal_close']for w in result['workers']),O_faithful=all(ready),snapshot_bytes=sum(p.stat().st_size for p in root.rglob('*')if p.is_file()));save();assert result['snapshot_bytes']<=m['snapshot_byte_upper'];assert result['all8normal']and not any('error'in w for w in result['workers'])"""
new=""" final=reporter.snapshot();size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file());allnormal=all(w['normal_close']for w in final['workers']);reporter.publish_top(phase='completed',all8normal=allnormal,O_faithful=all(ready),snapshot_bytes=size);assert size<=m['snapshot_byte_upper'];assert allnormal and not any('error'in w for w in final['workers'])"""
assert old in s;s=s.replace(old,new);ast.parse(s);(r/'kimi_embedding_layer0_runner_r8643.py').write_text(s,encoding='utf8',newline='\n');assert "own=result['workers']"not in s and "result['phase']='capture"not in s;print('Integrated owner-local records + deep isolated publication; GPU/barriers unchanged')
