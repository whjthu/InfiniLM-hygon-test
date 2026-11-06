import unittest
from prefill_schedule import fragments
class WholePromptTest(unittest.TestCase):
 def test_actual1249_input_all_ids_and_positions_survive(self):
  ids=list(range(1249));calls=list(fragments(ids,512));self.assertEqual([(o,len(c))for o,c in calls],[(0,512),(512,512),(1024,225)]);self.assertEqual([v for o,c in calls for v in c],ids)
 def test_full_context_budget_and_boundary_lengths(self):
  for n in [1,511,512,513,1024,4096,28672]:
   ids=[(i*17)%154880 for i in range(n)];calls=list(fragments(ids,512));self.assertEqual([v for o,c in calls for v in c],ids);self.assertEqual([p for o,c in calls for p in range(o,o+len(c))],list(range(n)));self.assertLessEqual(n+4096,32768)
 def test_undeclared_or_empty_input_refused(self):
  for size in [0,-1,1,8192,512.0,True]:
   with self.assertRaises(ValueError):list(fragments([1],size))
  with self.assertRaises(ValueError):list(fragments([],512))
if __name__=='__main__':unittest.main(verbosity=2)
