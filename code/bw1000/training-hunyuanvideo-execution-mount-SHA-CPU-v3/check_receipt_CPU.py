"""New receipt corruption/mount identity and minute-floor bounds; no body/Slurm read."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import stat

D=Path(__file__).resolve().parent


def main():
    spec=importlib.util.spec_from_file_location('new_mount_receipt',D/'caller-v3/execution_mount_receipt.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt=dict(schema='T09_EXECUTION_MOUNT_WHOLE_SHA_RECEIPT_V1',passed=True,request_id='a'*32,
        CPU_step='1103606.999999',source_manifest_SHA256='b'*64,publication='c'*40,epoch=100.,deadline=1000.,
        work_deadline=880.,finished_epoch=200.,node='f13r3n13',boot_id='test-boot',uid=154459,parent='1103606',
        relative_weight=module.RELATIVE,payload_bytes_read=module.WEIGHT_BYTES,weight_SHA256=module.WEIGHT_SHA,
        file_identity=[1,2,module.WEIGHT_BYTES,3,4],file_mode=stat.S_IFREG|0o600,
        tensor_decodes=0,Torch_SDK_imports=0,GPU_cards=0,original_replay_inputs_read=False)
    authority=dict(root_independent_terminal_verified=True,passed=True,worker_exit_evidence='EXACT_SACCT_COMPLETED0_FOREGROUND_NO_CHILDREN',
        accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES'],
        actual_accounting=['1103606.999999',module.LABEL,'COMPLETED','0:0','100','f13r3n13','cpu=2,mem=4G,node=1'],
        original_request_id='a'*32,CPU_source_manifest_SHA256='b'*64,CPU_publication='c'*40,CPU_epoch=100.)
    def call(r=receipt,a=authority,stamp=None,boot='test-boot'):
        return module.validate(r,a,stamp=receipt['file_identity'] if stamp is None else stamp,node='f13r3n13',boot=boot,uid=154459)
    assert call()['payload_bytes_read']==0
    rejected=[]
    changes=[('changed-mount-stamp',dict(stamp=[5,2,module.WEIGHT_BYTES,3,4])),('reboot',dict(boot='new-boot')),
             ('producer-only-no-rootaudit',dict(a=dict(authority,root_independent_terminal_verified=False))),
             ('wrong-worker-hash',dict(r=dict(receipt,weight_SHA256='0'*64))),
             ('failed-accounting',dict(a=dict(authority,actual_accounting=['1103606.999999',module.LABEL,'RUNNING','0:0','100','f13r3n13','cpu=2,mem=4G,node=1'])))]
    for label,kwargs in changes:
        try:call(**kwargs)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    tree=ast.parse((D/'launch_once.py').read_bytes());scope={}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('need','step_budget')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-minute-budget','exec'),scope)
    assert scope['step_budget'](899)==780 and scope['step_budget'](121)==60
    try:scope['step_budget'](119)
    except ValueError:rejected.append('late-wholeminute-rejected')
    else:raise AssertionError('late minute accepted')
    print(json.dumps(dict(passed=True,valid_receipt=1,valid_minute_bounds=2,rejected=rejected,
                          Slurm_calls=0,weight_reads=0,Torch_GPU_calls=0)))


if __name__=='__main__':main()
