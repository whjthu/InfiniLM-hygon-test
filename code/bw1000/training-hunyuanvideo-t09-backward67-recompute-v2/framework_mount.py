"""Same-node private 393 tree and manifest gate for the original T09 run."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import os
import stat

from contract import FRAMEWORK, FRAMEWORK_MANIFEST, ROOT, load, need, read
from framework_binding import MANIFEST_SHA256, verify_framework

STAGE = ROOT / 'incoming/t09-private-framework393-ingress-v1/b1a692f520b94308a36818d2e1db252'
STAGE_ALIAS = Path('/t09-stage-ro')
STAGE_REQUEST = STAGE.name
STAGE_SHA = 'df12914b78960076cd134fecaff9abe44559b3a48cd4a44df74219ae7c64bc39'
AUDIT_SHA = '10c4cf55fe87e1324674d450834e4a34bdf3ab8a8c08c490f43e7bf8f36864e3'
STAGE_INODES = {'framework_directory': 23288644965, 'framework_manifest': 10296345640}
CPU_SCOPE = 'a662d07645914562bbf3f56c9dd16f19'
CPU_PROOF_SHA = '7b99ea282351431cf92c339f8eceaae3539c9f3ab70274e2fc01926ac57a035c'
BIND_ENV_KEYS = ('APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH')


@contextmanager
def directory_fd(root, _category):
    """Open canonical ancestor descriptors with Linux search-only permissions."""
    need(root.is_absolute() and root.anchor == '/' and '..' not in root.parts, 'EXACT_POSIX_ANCESTOR')
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    with ExitStack() as stack:
        fd = os.open('/', flags)
        stack.callback(os.close, fd)
        for part in root.parts[1:]:
            fd = os.open(part, flags, dir_fd=fd)
            stack.callback(os.close, fd)
        yield fd


def context_matches(host):
    flags = host.get('environment_bind_override_present')
    return (host.get('host_getcwd') == '/' and type(flags) is dict
            and set(flags) == set(BIND_ENV_KEYS) and all(type(value) is bool for value in flags.values()))


def verify_stage(compute=False):
    base = STAGE_ALIAS if compute else Path(STAGE)
    value, pin = load(base / 'stage-result.json', 65536)
    need(pin['SHA256'] == STAGE_SHA and value['passed'] is True and value['request'] == STAGE_REQUEST
         and value['verified_original'] == 393 and value['manifest_SHA256'] == MANIFEST_SHA256,
         'ORIGINAL_393_PRIVATE_STAGE_BINDING')
    audit = STAGE.parent / (STAGE_REQUEST + '.stage-audit.jsonl')
    _, audit_pin = read(audit, 65536)
    need(audit_pin['SHA256'] == AUDIT_SHA,'ORIGINAL_393_PRIVATE_STAGE_AUDIT')
    return dict(stage=pin, audit=audit_pin, CPU_closure_public_SHA256=CPU_PROOF_SHA)


def before_model(scope):
    """Return a durable-ready report even when mounting or framework verification fails."""
    from mount_evidence import check_objects
    result = dict(scope, passed=False, phase='MOUNT', planned=393, actual_verified=0,
                  after_model=False, GPU_SDK_calls_before_gate=0)
    try:
        mount = result['mount_evidence'] = check_objects()
        need(mount['passed'] and mount['verified_objects'] == 2, 'EFFECTIVE_FD_MOUNT_ADMISSION_FAILED')
        for label, inode in STAGE_INODES.items():
            identities = [path['identity'] for path in mount['objects'][label]['roles'].values()]
            need(len(identities)==2 and all(item['inode']==inode for item in identities),
                 'ORIGINAL_3536_PRIVATE_STAGE_INODE')
        result['phase'] = 'SEALED_PRIVATE_STAGE'
        result['stage'] = verify_stage(compute=True)
        result['phase'] = 'ORIGINAL_FRAMEWORK_VERIFIER'
        framework = result['framework'] = verify_framework(Path(FRAMEWORK), Path(FRAMEWORK_MANIFEST))
        need(framework['framework_files']==393, 'ORIGINAL_FULL_FRAMEWORK393')
        result.update(passed=True, phase='READY_FOR_ORIGINAL_NATIVE_INGRESS', actual_verified=393)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(error_type=type(error).__name__, errno=getattr(error,'errno',None))
        if type(error) is ValueError and str(error).isupper() and len(str(error)) <= 96:
            result['reason'] = str(error)
    return result


def post_mount(value, request, device):
    scope = device['step']
    need(value['request_id']==request and value['step']==scope and value['boot_id']==device['boot_id']
         and value['host_gate']==device['host_gate'] and value['actual_verified']==value['planned']==393
         and value['passed'] is True and value['mount_evidence']['verified_objects']==2
         and value['mount_evidence']['write_probe_performed'] is False
         and value['stage']['stage']['SHA256']==STAGE_SHA and value['stage']['audit']['SHA256']==AUDIT_SHA
         and value['framework']==device['framework'],
         'SAME_GPU_STEP_PRIVATE_FRAMEWORK_GATE_REQUIRED')
