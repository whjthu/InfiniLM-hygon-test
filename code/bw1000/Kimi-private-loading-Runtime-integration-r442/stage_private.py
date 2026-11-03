"""Build an exclusive private source/data overlay. No imports of Torch or Native."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from runtime_guard_r442 import RUNTIME_SHA

PARENT_MF = 'fc7e8bc3e4b846929c7616477283071ab136c8595a5374584641a394b5ce6be6'
TEXT_MF = 'f0821642a154b0ca51172093b8bd4119b117c9977b5a2d22fbbddb53136be7b4'
OLD_RUNTIME = '04a6a1a86c97d176e4304b9bb4c1d6fcb8550784beaa47e0f5841fb7603ffabf'
NATIVE = '8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
EXTENSION = '2e1458609cc81a31351c451004c045b8e84f7581844afedf1a577df2c940ccb2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')


def clone_tree(source, expected, output):
    assert sha(source / 'SOURCE_MANIFEST.json') == expected
    manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_bytes())
    output.mkdir()
    for name, digest in manifest['files_SHA256'].items():
        src = source / name
        assert src.resolve().is_relative_to(source.resolve()) and sha(src) == digest, name
        dst = output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    return manifest


def freeze(root, original):
    result = dict(original)
    result['files_SHA256'] = {str(p.relative_to(root)).replace('\\', '/'): sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'SOURCE_MANIFEST.json'}
    result['CPU_GO'] = result['GPU_GO'] = result['Model_GO'] = False
    result['r442_scope'] = 'Private Runtime integration prepared; no Model qualification'
    write_json(root / 'SOURCE_MANIFEST.json', result)
    return sha(root / 'SOURCE_MANIFEST.json')


def stage(parent, text, config_path, candidate, output):
    parent, text, output = map(Path, (parent, text, output))
    assert not output.exists(), 'Never mutate an existing deployment'
    assert sha(candidate) == RUNTIME_SHA
    cfg = json.loads(Path(config_path).read_bytes())
    assert cfg['Native_SHA256'] == NATIVE and cfg['Common_extension_SHA256'] == EXTENSION
    assert cfg['qualified_parent_root'] == str(parent)
    old_manifest = Path(cfg['model_manifest'])
    assert sha(old_manifest) == cfg['model_manifest_SHA256']
    m = json.loads(old_manifest.read_bytes())
    assert m['native_DSO_SHA256'] == NATIVE
    assert m['loaded_library_SHA256']['libinfinilm_runtime.so'] == OLD_RUNTIME
    output.mkdir(parents=True)
    private_parent, private_text = output / 'parent', output / 'text'
    p_mf = clone_tree(parent, PARENT_MF, private_parent)
    t_mf = clone_tree(text, TEXT_MF, private_text)
    libdir = output / 'runtime'
    libdir.mkdir()
    alias = libdir / 'libinfinilm_runtime.so'
    shutil.copyfile(candidate, alias)  # SAME bytes; no SONAME or ELF edits/relink.
    assert sha(alias) == RUNTIME_SHA
    m['loaded_library_SHA256']['libinfinilm_runtime.so'] = RUNTIME_SHA
    m['private_Runtime_r442'] = dict(path=str(alias.resolve()), SHA256=RUNTIME_SHA)
    # Only the strict loaded Runtime identity changes. No loader/math policy edit.
    private_model = private_parent / 'qualified-model-r442.json'
    write_json(private_model, m)
    binding = private_parent / 'public_kimi_native/runtime/native_binding_r9143.py'
    raw = binding.read_text()
    needle = ' return actual\n'
    assert raw.count(needle) == 1
    raw = raw.replace(needle, " from .runtime_guard_r442 import verify_runtime\n actual['r442_Runtime_binding']=verify_runtime(m, pathlib.Path('/proc/self/maps').read_text())\n return actual\n")
    binding.write_text(raw, encoding='utf8', newline='\n')
    shutil.copyfile(Path(__file__).with_name('runtime_guard_r442.py'), binding.with_name('runtime_guard_r442.py'))
    owner = private_parent / 'public_kimi_native/owner.py'
    raw = owner.read_text()
    before = "      def progress(x):report.update(x);save()\n"
    assert raw.count(before) == 1
    raw = raw.replace(before, "      from .runtime.runtime_guard_r442 import copy_hits, checked_load_delta\n      report['r442_before_weight_load']=copy_hits(manifest);save()\n" + before)
    after = ";report['head_after_finish']=read_head(lib,node,pp,True)"
    assert raw.count(after) == 1
    raw = raw.replace(after, ";report['r442_after_weight_load']=copy_hits(manifest);report['r442_actual_weight_copy_delta']=checked_load_delta(report['r442_before_weight_load'],report['r442_after_weight_load']);save()" + after)
    owner.write_text(raw, encoding='utf8', newline='\n')
    new_parent_sha = freeze(private_parent, p_mf)
    cli = private_text / 'public_kimi_text/cli.py'
    raw = cli.read_text()
    assert raw.count(PARENT_MF) == 1
    cli.write_text(raw.replace(PARENT_MF, new_parent_sha), encoding='utf8', newline='\n')
    new_text_sha = freeze(private_text, t_mf)
    cfg.update(qualified_parent_root=str(private_parent), model_manifest=str(private_model),
               model_manifest_SHA256=sha(private_model), execute_authorized=False, Model_READY=False)
    write_json(output / 'config.json', cfg)
    proof = dict(schema=1, Runtime_only_candidate=True, Model_READY=False, GPU_GO=False,
        Runtime_SHA256=RUNTIME_SHA, runtime_directory=str(libdir),
        parent=str(private_parent), parent_MF_SHA256=new_parent_sha,
        text=str(private_text), text_MF_SHA256=new_text_sha,
        config=str(output / 'config.json'), config_SHA256=sha(output / 'config.json'),
        original_config_SHA256=sha(config_path), original_parent_MF_SHA256=PARENT_MF,
        original_text_MF_SHA256=TEXT_MF, Native_SHA256=NATIVE, Common_extension_SHA256=EXTENSION,
        device_import_and_Model_loading='NOT_EXECUTED',
        remaining='Next necessary Kimi loading only: actual strict maps/dladdr/first-copy chosen and quality remain required')
    write_json(output / 'DEPLOYMENT.json', proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['parent', 'text', 'config', 'candidate', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(stage(a.parent, a.text, a.config, a.candidate, a.output)))
