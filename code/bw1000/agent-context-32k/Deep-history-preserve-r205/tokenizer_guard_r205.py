"""CPU561-qualified metadata/backend/weight guard only; no fake modules."""
import importlib.abc,importlib.util,os,pathlib,re,sys,traceback
FORBIDDEN_BACKENDS = frozenset(['torch', '_infinilm', 'infinilm'])

BACKEND_METADATA_PROBES = {}

BLOCKED_BACKEND_ATTEMPTS = []

class BlockBackend(importlib.abc.MetaPathFinder, importlib.abc.Loader):

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN_BACKENDS:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        if len(BLOCKED_BACKEND_ATTEMPTS) < 8:
            BLOCKED_BACKEND_ATTEMPTS.append({'module': spec.name, 'frames': [{'file': pathlib.Path(f.filename).name, 'line': f.lineno} for f in traceback.extract_stack(limit=8)]})
        raise ImportError('Backend import blocked in tokenizer-only CPU comparison')

    def exec_module(self, module):
        raise ImportError('Backend import blocked in tokenizer-only CPU comparison')

def install_backend_guard():
    original = importlib.util.find_spec

    def find_spec(name, package=None):
        backend = name.split('.')[0]
        if backend in FORBIDDEN_BACKENDS:
            BACKEND_METADATA_PROBES[backend] = BACKEND_METADATA_PROBES.get(backend, 0) + 1
            return None
        return original(name, package)
    importlib.util.find_spec = find_spec
    sys.meta_path.insert(0, BlockBackend())

def no_backend():
    assert not any((x.split('.')[0] in ['torch', '_infinilm', 'infinilm'] for x in sys.modules))

def deny_weights(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        suffix = pathlib.Path(path).suffix
        if suffix in ['.safetensors', '.pt', '.pth', '.ckpt'] or re.search('(?:pytorch_model|model-\\d+-of-\\d+)\\.bin$', path):
            raise RuntimeError('Weight payload opening forbidden in tokenizer-only CPU comparison')
