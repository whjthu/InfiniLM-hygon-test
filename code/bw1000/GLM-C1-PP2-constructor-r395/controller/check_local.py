"""Real final controller main/Popen CPU child; Slurm/device leaves recorded."""
from pathlib import Path
import contextlib,hashlib,io,json,os,subprocess,sys,tempfile
from unittest import mock
import controller
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_CPU import case
from constructor import NATIVE_SHA

def scenario(mode):
    actual_popen=subprocess.Popen;calls=[];state={'started':False};reports=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'incoming/GLM-C1-PP2-constructor-r395';source.mkdir(parents=True)
        for name in ['results','control','logs']: (root/name).mkdir()
        (source/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files_SHA256={})))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source_sha=sha(source/'SOURCE_MANIFEST.json')
        cp=root/'candidates/GLM-C1-PP2-constructor-r395/CPU-result.json';cp.parent.mkdir(parents=True)
        cp.write_text(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,source_manifest_SHA256=source_sha,config_SHA256='1'*64)))
        wrapper=root/'incoming/resource_wave_local_r3097_outer.sh';wrapper.write_text('# Recording leaf, never execute GPUwrapper\n')
        manifest=dict(GPU_GO=True,Model_GO=False,GPU_max_s=300,Native_SHA256=NATIVE_SHA,source_manifest_SHA256=source_sha,controller_files_SHA256={},CPU_receipt_SHA256=sha(cp),config_SHA256='1'*64,outer_wrapper_SHA256=sha(wrapper))
        (source/'GPU_MANIFEST.json').write_text(json.dumps(manifest))
        mf=sha(source/'GPU_MANIFEST.json');step='910376.999'
        def popen(argv,**kw):
            assert not state['started'];state['started']=True;calls.append(argv)
            assert argv[0]=='srun' and '-N2'in argv and '-n2'in argv and '-c32'in argv and '--gres=dcu:8'in argv and '00:05:00'in argv
            env=kw['env'];go=json.loads((root/'control'/Path(env['GLM_R395_GO']).name).read_bytes())
            assert go['Model_GO']is False and float(env['GLM_R395_DEADLINE_UNIX'])==go['original_GPU_start_unix']+300
            for stage in range(2):
                for rank in range(8):
                    q,_,error=case(stage,rank);assert error is None
                    q.update(actual_step=step,source_manifest_SHA256=source_sha)
                    if mode=='bad_identity' and stage==1 and rank==7:q['rank']=8
                    path=root/'results'/('GLM-C1-PP2-constructor-r395-'+controller.RUN+'-stage%d-rank%d.json'%(stage,rank))
                    path.write_text(json.dumps(q));reports.append(path)
                (root/'logs'/(controller.LABEL+'-'+controller.RUN+'-'+step+'-%d.log'%stage)).write_text('GLM_C1_PP2_R395_NORMAL '+str(stage)+'\n')
            # The submitted srun leaf is replaced with a real, bounded CPU child.
            return actual_popen([sys.executable,'-B','-c','pass'],**kw)
        def output(argv,**kw):
            if argv[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=aclsylqidf(7) NodeList=a02r4n[12-13]'
            if argv[:3]==['scontrol','show','hostnames']:return '\n'.join(controller.NODES)+'\n'
            assert argv[0]=='sacct'
            if not state['started']:return ''
            status='FAILED|1:0' if mode=='step_failed' else 'COMPLETED|0:0'
            return step+'|'+controller.LABEL+'-'+controller.RUN+'|'+status+'|a02r4n[12-13]\n'
        def run(argv,**kw):
            assert argv[0]=='srun' and '--gres=none'in argv
            phase=argv[argv.index('--job-name='+controller.LABEL+'-'+controller.RUN+'-post')].endswith('-post') if '--job-name='+controller.LABEL+'-'+controller.RUN+'-post'in argv else False
            bad=mode=='bad_post' and phase
            rows=[dict(node=n,actual_step='910376.998',CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=[],matching_component_processes=[],ports_bindable={'43843':not bad}) for n in controller.NODES]
            return subprocess.CompletedProcess(argv,0,stdout='\n'.join(json.dumps(x) for x in rows),stderr='')
        env=dict(SLURM_JOB_ID='910407',SLURM_STEP_ID='recording')
        with mock.patch.object(controller,'R',root),mock.patch.object(controller,'S',source),mock.patch.object(controller.socket,'gethostname',return_value='a14r2n20'),mock.patch.dict(os.environ,env),mock.patch.object(sys,'argv',['controller','--publication-commit','2'*40,'--manifest-sha',mf]),mock.patch.object(controller.subprocess,'check_output',side_effect=output),mock.patch.object(controller.subprocess,'run',side_effect=run),mock.patch.object(controller.subprocess,'Popen',side_effect=popen),contextlib.redirect_stdout(io.StringIO()):
            rc=controller.main()
            watch=root/'results'/(controller.LABEL+'-'+controller.RUN+'-watch.json');q=json.loads(watch.read_bytes())
            assert q['completed'] and q['passed']==(mode=='normal') and rc==int(mode!='normal') and len(calls)==1
            before=sha(watch)
            try:controller.main()
            except AssertionError:pass
            else:raise AssertionError('Existingwatch replay accepted')
            assert len(calls)==1 and sha(watch)==before
        return dict(mode=mode,passed=True,controller_expected_pass=mode=='normal',real_CPU_Popen_children=1,owned_report_leaves_recorded=16,actual_GPU=False,actual_Slurm=False)

def main():
    cases=[scenario(x) for x in ['normal','step_failed','bad_identity','bad_post']]
    return dict(completed=True,passed=True,cases=cases,GPU=0,Native_dlopen=False,weights=False,source='actualfinalcontroller.main/verify/parser/GOwriter/Popen.wait/reportcollector/immutableintent;Slurm/resource/Nativeleavesrecorded')

if __name__=='__main__':print(json.dumps(main(),indent=2))
