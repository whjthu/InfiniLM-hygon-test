"""Exact TopK set/order on actual returned F32 scores; ties need no identical IDs."""
import numpy as np

def check(packets,details):
 for p,d in zip(packets,details):
  f=p['fields'];scores=np.asarray(f[4]['raw'],np.uint8).view(np.float32).reshape(-1);ids=np.asarray(f[5]['raw'],np.uint8).view(np.int64).reshape(-1);values=np.asarray(f[11]['raw'],np.uint8).view(np.float32).reshape(-1)
  assert scores.size==2124 and ids.size==values.size==512 and np.isfinite(scores).all()and np.isfinite(values).all()
  assert len(set(ids.tolist()))==512 and np.all((ids>=0)&(ids<2124))
  assert np.array_equal(values.view(np.uint32),scores[ids].view(np.uint32)),'values must equal original score[id] bits'
  assert np.array_equal(values,np.sort(scores)[::-1][:512]),'exact ranked value set/order required; no tolerance belowcutoff'
  d['TopK_exact_ranked_value_set_on_actual_F32_scores']=True;d['TopK_sorted_largest']=True;d['TopK_tie_IDs_need_not_match_Torch']=True
