"""CPU recording regression for895's RUNNING0 after successful srun, no Slurm."""
import unittest
from slurm_accounting import wait_terminal
class Accounting(unittest.TestCase):
 def run_rows(self,rows,deadline=10):
  now=[0.];it=iter(rows)
  def read():return next(it)
  def sleep(n):now[0]+=n
  return wait_terminal(read,deadline,monotonic=lambda:now[0],sleep=sleep),now[0]
 def test_original895_visibility_race(self):
  a=['910371.895','label','RUNNING','0:0','1637'];b=['910371.895','label','COMPLETED','0:0','1637']
  out,t=self.run_rows([a,b]);self.assertEqual(out,dict(terminal_visible=True,row=b,observations=[a,b]));self.assertEqual(t,2.)
 def test_failure_is_terminal_without_relabel(self):
  r=['910371.1','label','FAILED','1:0','10'];out,t=self.run_rows([None,r]);self.assertEqual(out['row'],r);self.assertTrue(out['terminal_visible'])
 def test_deadline_retains_unknown_and_original_rows(self):
  r=['910371.1','label','RUNNING','0:0','10'];out,t=self.run_rows([r,r],deadline=3.);self.assertFalse(out['terminal_visible']);self.assertEqual(out['observations'],[r,r]);self.assertEqual(t,3.)
 def test_missing_rows_are_preserved_bounded(self):
  out,t=self.run_rows([None]*4);self.assertFalse(out['terminal_visible']);self.assertEqual(out['observations'],[None]*4);self.assertEqual(t,6.)
if __name__=='__main__':unittest.main()
