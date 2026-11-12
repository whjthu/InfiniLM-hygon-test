"""Test actual new terminal caller with the unchanged CPU160-qualified helper."""
from pathlib import Path
import ast,importlib.util,math,time,unittest
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('qualified_accounting',D/'slurm_accounting.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tree=ast.parse((D/'pilot_dispatch_remote.py').read_bytes());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal')
compiled=compile(ast.Module(body=[function],type_ignores=[]),'actual_new_dispatch_terminal','exec')
COLUMNS=['step','label','state','exit_code','elapsed_seconds','node']


class AccountingCallerTests(unittest.TestCase):
 def caller(self,samples):
  records=[];queries=[];queue=list(samples)
  def read(timeout):
   self.assertGreater(timeout,0);self.assertLessEqual(timeout,12);queries.append(timeout)
   return queue.pop(0) if len(queue)>1 else queue[0]
  ns=dict(time=time,math=math,COLUMNS=COLUMNS,left=lambda reserve:100,wait_terminal=module.wait_terminal,
          accounting=read,emit=lambda stage,**v:records.append(dict(stage=stage,**v)))
  exec(compiled,ns);return ns['terminal'],records,queries
 def row(self,state):return dict(step='1087656.999',label='new-owned-scope',state=state,exit_code='0:0',elapsed_seconds='1',node='f12r3n12')
 def test_running_zero_exit_waits_for_real_terminal(self):
  call,records,queries=self.caller([[self.row('RUNNING')],[self.row('COMPLETED')]])
  result=call('new-owned-scope',1.2);self.assertEqual(result['state'],'COMPLETED');self.assertEqual(len(queries),2)
  self.assertEqual([r['state'] for r in records[0]['observations']],['RUNNING','COMPLETED'])
 def test_running_is_not_promoted_after_client_return(self):
  call,records,queries=self.caller([[self.row('RUNNING')]])
  self.assertIsNone(call('new-owned-scope',.02));self.assertFalse(records[0]['terminal_visible'])
  self.assertLessEqual(max(queries),.02)
 def test_missing_remains_unknown_with_bounded_query(self):
  call,records,queries=self.caller([[]])
  self.assertIsNone(call('new-owned-scope',.02));self.assertFalse(records[0]['terminal_visible']);self.assertLessEqual(max(queries),.02)
 def test_duplicate_original_label_is_refused(self):
  call,_,_=self.caller([[self.row('COMPLETED'),self.row('RUNNING')]])
  with self.assertRaises(RuntimeError):call('new-owned-scope',1)


if __name__=='__main__':unittest.main()
