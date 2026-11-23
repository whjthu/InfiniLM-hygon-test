"""Check the real host mapping and staged compile/shard bounds locally."""

import json
from pathlib import Path
from shutil import which
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import device
import launch_plan as placement
import prepare
import read_terminal


class Tests(unittest.TestCase):
    def test_CXX_actual_lane_arithmetic_and_bad_bindings(self):
        compiler = which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as local:
            temporary = Path(local).resolve(strict=True)
            executable = temporary/'check.exe'
            cmd = [compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',
                   'check_layout.cpp','-o',str(executable)]
            subprocess.run(cmd,cwd=HERE,timeout=30,check=True)
            actual = subprocess.run([str(executable)],stdout=subprocess.PIPE,timeout=10,check=True,text=True)
            self.assertIn('1048576 BF16 elements per variant',actual.stdout)
            self.assertIn('8 bad bindings rejected',actual.stdout)

    def test_exact_zero_GRES_manifest_targets_one_TU_and_link(self):
        src = prepare.build_inputs()
        plan = json.loads(src['PLAN.json'])
        manifest = json.loads(src['SOURCE_FILES.json'])
        self.assertEqual(manifest['files_SHA256'],{n:prepare.digest(r) for n,r in src.items() if n!='SOURCE_FILES.json'})
        self.assertEqual(plan['geometry']['heads_per_block'],[1,2])
        self.assertEqual(plan['geometry']['block_threads'],128)
        self.assertIn('-arch=gfx936',plan['compiler_prefix'])
        self.assertEqual(plan['actual_GPU_steps'],0)
        invoker = src['compile_once.py']
        compile(invoker,'derived target compile entry','exec')
        self.assertIn(b'ADMITTED_SLURM_STEP_IDENTITY',invoker)
        self.assertIn(b'SLURM_STEP_GPUS',invoker)
        self.assertIn(b'fresh-M',invoker)
        self.assertIn(b'ACTUAL_MD_DEPENDENCY_DRIFT',invoker)
        self.assertIn(b'EXACT_PAIR_DSO_EXPORTS',invoker)
        self.assertIn(b'actual_MD_SHA256=pin(actual)',invoker)
        self.assertIn(b'actual_exports == EXPORTS',invoker)
        with tempfile.TemporaryDirectory() as local:
            staged=Path(local).resolve(strict=True)/'fresh'
            result=prepare.stage(staged)
            self.assertEqual(result['actual_target_compiles'],0)
            self.assertEqual(result['compile_manifest_SHA256'],prepare.digest(src['SOURCE_FILES.json']))
            with self.assertRaises(FileExistsError):
                prepare.stage(staged)

    def base(self, nodes, kind='pilot'):
        request = placement.PILOT_REQUEST if kind=='pilot' else 'a'*32
        before = {'parent':placement.PARENT,'epoch':1000,'deadline_epoch':1150,'request':request,
                  'pre_step':'1103606.200','root_admitted':True,'scope_SHA256':'d'*64}
        root = str(placement.ROOT)
        return dict(kind=kind, request=request, package=root+'/incoming/reviewed-source',
                    campaign=root+'/results/unique-'+request, epoch=1000, deadline_epoch=1150,
                    build_receipt=root+'/candidates/verified-artifact/CPU-result.json',build_receipt_SHA256='b'*64,
                    nodes={node:dict(before,node=node) for node in nodes},
                    project_owned=True, fresh_protected_excluded=True)

    def test_current_parent_pilot_and_protected_node_gate(self):
        packet=self.base(['f12r2n13'])
        result=placement.derive(packet,now=1003)
        self.assertEqual(result['planned_cards'],8)
        self.assertEqual(result['submissions_performed'],0)
        argv=result['shards'][0]['argv']
        self.assertIn('--gres=dcu:8',argv)
        self.assertIn('--jobid=1103606',argv)
        self.assertIn('--time=00:03:00',argv)
        self.assertIn('--nodelist=f12r2n13',argv)
        self.assertEqual(argv[-1],'f12r2n13:'+'d'*64)
        with self.assertRaises(ValueError):
            placement.derive(self.base(['f13r2n03']),now=1003)
        with self.assertRaises(ValueError):
            placement.derive(self.base(['f12r3n17']),now=1003)
        with self.assertRaises(ValueError):
            placement.derive(packet,now=1015)

    def test_followup_requires_actual_clean_pilot_and_is_disjoint(self):
        nodes=sorted(placement.known_nodes()-placement.EXCLUDED)[:9]
        packet=self.base(nodes,'followup')
        term=dict(step='1103606.222',node='f12r2n13',request=placement.PILOT_REQUEST,complete=True,
                  valid_cards=8,completed_arms=32,actual_pairs=16,sharding_candidate=True)
        accounting=[term['step'],'actual-label','COMPLETED','0:0','67','f12r2n13','cpu=12,gres/dcu=8,mem=16G,node=1']
        post=dict(step='1103606.223',node='f12r2n13',verified=True,SHA256='c'*64)
        packet['pilot']=dict(step=term['step'],terminal=term,accounting=accounting,independent_post=post)
        partitions=placement.derive(packet,now=1004)['shards']
        self.assertEqual([len(p['nodes']) for p in partitions],[4,4,1])
        self.assertEqual([p['planned_cards'] for p in partitions],[32,32,8])
        self.assertEqual(sorted(n for p in partitions for n in p['nodes']),nodes)
        self.assertEqual(len({p['request'] for p in partitions}),3)
        post['verified']=False
        with self.assertRaises(ValueError):
            placement.derive(packet,now=1004)

    def test_device_start_gate_refuses_no_artifact_and_wrong_clock(self):
        record=self.base(['f12r2n13'])['nodes']['f12r2n13']
        record.update(PCI=[f'0000:{x:02x}:00.0' for x in range(8)])
        device.read_scope(record,'f12r2n13','1103606.400',1005)
        with self.assertRaises(ValueError):
            device.read_scope(record,'f13r2n01','1103606.400',1005)
        with self.assertRaises(ValueError):
            device.read_scope(record,'f12r2n13','1103606.400',1110)
        with self.assertRaises(ValueError):
            device.bind(None,dict(shared='/unknown/path.so'))

    def test_unknown_and_partial_never_qualify(self):
        scope=self.base(['f12r2n13'])['nodes']['f12r2n13']
        scope.update(PCI=[f'0000:{x:02x}:00.0' for x in range(8)])
        spec=dict(scopes={'f12r2n13':scope},label='fresh-native',accounting_rows=[])
        with tempfile.TemporaryDirectory() as local:
            path=Path(local).resolve(strict=True)
            unknown=read_terminal.reconcile(path,spec)
            self.assertEqual((unknown['planned_cards'],unknown['planned_arms'],unknown['state']),(8,32,'UNKNOWN'))
            step='1103606.400'
            path.joinpath('intent.json').write_text(json.dumps(dict(scope,step=step,planned_cards=8)))
            result=read_terminal.node_result(path,'f12r2n13',step,scope,[],[])
            self.assertFalse(result['device_qualified'])
            self.assertEqual((result['planned_arms'],result['completed_arms']),(32,0))
            self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main()
