import unittest,ctypes as C
from resident_trial import schedule,phases,owner_mask,attach_profile,read_profile,add_profile
class Trial(unittest.TestCase):
 def test_exact_three_pairs_and_fullwarmups(self):
  s=schedule();self.assertEqual(len(s),10);self.assertEqual(sum(x['warmup']for x in s),4)
  for pair in range(3):
   rows=[x for x in s if x['pair']==pair];self.assertEqual({x['arm']for x in rows},{'A','B'});self.assertEqual(len({x['case_index']for x in rows}),1)
  self.assertEqual([x['arm']for x in s],list('ABABABBAAB'));self.assertEqual(len(set(phases())),14)
 def test_disjoint_masks_and_fixed_ownerblocks(self):
  c=list(range(64));m=[set(owner_mask(c,d,4))for d in range(8)]
  for i in range(8):
   self.assertTrue(set(owner_mask(c,i,2))<m[i])
   for j in range(i):self.assertFalse(m[i]&m[j])
  with self.assertRaises(ValueError):owner_mask(c,0,8)
 def test_same_thread_profile64bit_pointer_copied(self):
  a=(C.c_double*6)(1,2,3,4,5,6)
  class F:
   def __call__(self):return C.cast(a,C.POINTER(C.c_double))
  class L:modern_glm_pipeline_last_profile=F()
  f=attach_profile(L());v=read_profile(f);self.assertEqual(v,[1.,2.,3.,4.,5.,6.]);a[0]=9;self.assertEqual(v[0],1.);self.assertEqual(f.restype,C.POINTER(C.c_double))
 def test_profile_aggregate_has_bounded_selected_notlost_count(self):
  b={}
  for i in range(5000):add_profile(b,[.1]*6,1,i)
  self.assertEqual(b['count'],5000);self.assertLessEqual(len(b['selected']),24);self.assertAlmostEqual(b['sum_s'][0],500.)
if __name__=='__main__':unittest.main()
