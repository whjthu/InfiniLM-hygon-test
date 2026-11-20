"""Exercise the reader with real local x86_64 ELF and labelled mock receipts."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import audit_once as a


PUB = '1'*40
EPOCH = 1900000000.0


def write_json(path,value):
    path.write_text(json.dumps(value)+'\n', encoding='utf8')


class AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build.cleanup)
        tmp=Path(cls.build.name)
        source=tmp/'local.cc'
        source.write_text('\n'.join(f'extern "C" int {symbol}() {{return {i};}}'
                                   for i,symbol in enumerate(sorted(a.EXPORTS))))
        coff=tmp/'local.o'
        elf=tmp/'local.elf'
        subprocess.run(['g++','-std=c++17','-c',str(source),'-o',str(coff)],check=True,timeout=25)
        subprocess.run(['objcopy','-O','elf64-x86-64',str(coff),str(elf)],check=True,timeout=15)
        cls.elf=elf.read_bytes()
        cls.symbols=subprocess.check_output(['nm','-g','--defined-only',str(elf)],timeout=15)

    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root=Path(directory.name)
        self.result=self.root/'CPU-result.json'
        self.launch=self.root/'launch.json'
        self.slurm=self.root/'accounting.txt'

    def fixture(self,arm='down'):
        p,rows,mf,srcbase=a.load(arm)
        self.step='1103606.7766'
        self.arm=arm
        self.plan=p
        self.line=[self.step,p['label'],'COMPLETED','0:0','21',p['node'],'cpu=4,mem=8G,node=1']
        self.slurm.write_text('|'.join(self.line)+'\n')
        write_json(self.launch,dict(publication=PUB,manifest_SHA256=mf,request='2'*32,
                  epoch=EPOCH,deadline=EPOCH+300,GPU=False,pid=51,pgid=51,startticks=7634,
                  intent='/scope/results/'+p['label']+'-intent.json'))
        commands = []
        nm=['/usr/bin/nm','-g','--defined-only']

        def record(label,argv,stdout=b''):
            (self.root/(label+'.stdout')).write_bytes(stdout)
            (self.root/(label+'.stderr')).write_bytes(b'')
            commands.append(dict(label=label,argv=argv,stdout_SHA256=a.sha(stdout),stderr_SHA256=a.sha(b''),rc=0))

        record('original-symbols',nm+[p['original_object']],self.symbols)
        compiled={}
        todo=[]
        for row in rows:
            key=f'rows{row}'
            d=self.root/key
            d.mkdir()
            origin=f'/refactor/candidates/{p["label"]}/{key}/routed_i128'
            src,o,md=(origin+ext for ext in ('.cu','.o','.d'))
            raw=(srcbase/(key+'.cu')).read_bytes()
            meta=(o+': '+src+' /frozen/fixture.hpp\n').encode()
            (d/'routed_i128.cu').write_bytes(raw)
            (d/'routed_i128.o').write_bytes(self.elf)
            (d/'routed_i128.d').write_bytes(meta)
            record(key+'-M',p['compiler_prefix']+['-M',src],meta)
            todo.append((key,src,o,md))
            fixed=p['variants'][str(row)]
            compiled[str(row)]=dict(candidate_id=fixed['id'],source_SHA256=a.sha(raw),object=o,
                                    object_SHA256=a.sha(self.elf),MD_SHA256=a.sha(meta),dependency_count=2,
                                    host_wrapper_ABI=sorted(a.EXPORTS))
        for key,src,o,md in todo:
            record(key+'-compile',p['compiler_prefix']+['-MD','-MF',md,'-c',src,'-o',o])
            record(key+'-symbols',nm+[o],self.symbols)
        self.report=dict(original_step=self.step,node=p['node'],epoch=EPOCH,deadline=EPOCH+300,compiled=compiled,
                         passed=True,completed=True,all_fresh_M_before_compile=True,actual_GPU_steps=0,SDK_calls=0,
                         weights_read=0,numerical_qualified=False,planned_new_device_TUs=len(rows),commands=commands)
        write_json(self.result,self.report)

    def check(self):
        return a.audit(self.arm,self.launch,self.slurm,self.root,PUB)

    def test_inventory(self):
        self.assertEqual({k:(v['node'],v['rows']) for k,v in a.inventory().items()},
                         {'gate':('f13r1n09',[1,2]),'down':('f13r1n08',[2,4])})

    def test_two_arm_fixtures_with_local_host_elf(self):
        for arm in ('gate','down'):
            with self.subTest(arm=arm):
                self.fixture(arm)
                result=self.check()
                self.assertTrue(result['passed'])
                self.assertEqual(len(result['verified_objects']),2)
                self.assertFalse(result['device_numerical_qualified'])
                self.assertTrue(all(not row['embedded_uncompressed_AMDGPU_images'] for row in result['verified_objects']))
                for row in result['planned_variants']:
                    for file in (self.root/f'rows{row}').iterdir():
                        file.unlink()
                    (self.root/f'rows{row}').rmdir()

    def test_duplicate_and_nonterminal_original(self):
        self.fixture()
        self.slurm.write_text(('|'.join(self.line)+'\n')*2)
        with self.assertRaisesRegex(ValueError,'EXACT_SINGLE_NUMBERED_STEP'):
            self.check()
        self.line[2]='RUNNING'
        self.slurm.write_text('|'.join(self.line)+'\n')
        with self.assertRaisesRegex(ValueError,'SLURM_TERMINAL_SCOPE'):
            self.check()

    def test_unsafe_gres_rejected(self):
        self.fixture()
        self.line[-1]+= ',gres/dcu=1'
        self.slurm.write_text('|'.join(self.line)+'\n')
        with self.assertRaisesRegex(ValueError,'SLURM_TERMINAL_SCOPE'):
            self.check()

    def test_modified_object(self):
        self.fixture()
        (self.root/'rows2/routed_i128.o').write_bytes(self.elf+b'wrong')
        with self.assertRaisesRegex(ValueError,'CANDIDATE_ARTIFACT_BYTES'):
            self.check()

    def test_md_drift_with_matching_object_receipt(self):
        self.fixture()
        md=self.root/'rows2/routed_i128.d'
        raw=md.read_bytes().replace(b'fixture.hpp',b'wrong.hpp')
        md.write_bytes(raw)
        self.report['compiled']['2']['MD_SHA256']=a.sha(raw)
        write_json(self.result,self.report)
        with self.assertRaisesRegex(ValueError,'FRESH_M_EQUALS_COMPILED_MD'):
            self.check()

    def test_wrong_target_flag(self):
        self.fixture()
        self.report['commands'][3]['argv'][4]='-arch=gfx900'
        write_json(self.result,self.report)
        with self.assertRaisesRegex(ValueError,'EXACT_CANDIDATE_COMMAND'):
            self.check()

    def test_failure_retains_denominator(self):
        self.fixture()
        self.line[2:4]=['FAILED','1:0']
        self.slurm.write_text('|'.join(self.line)+'\n')
        self.result.unlink()
        result=self.check()
        self.assertEqual(result['planned_variants'],[2,4])
        self.assertFalse(result['passed'])
        self.assertEqual(result['outcome'],'ORIGINAL_TERMINAL_WITHOUT_BUILD_REPORT')

    def test_host_elf_rejects_truncated_or_foreign(self):
        for blob in (self.elf[:55],b'NOT_ELF',self.elf[:18]+b'\xe0\x00'+self.elf[20:]):
            with self.assertRaisesRegex(ValueError,'HOST_RELOCATABLE_ELF'):
                a.host_elf(blob)


if __name__=='__main__':
    unittest.main()
