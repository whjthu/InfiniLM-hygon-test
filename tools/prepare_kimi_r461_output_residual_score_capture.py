"""Materialize the Kimi score pair with optional post-dispatch score capture."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-wave-compile-CPU-v1'
ARMS = ('full-tree', 'wave32')
ORIGINAL_MANIFEST = 'd2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6'
CAPTURE_EXPORT = 'kimi_output_residual_scores_probe_r461'
ORIGINAL_EXPORT = 'kimi_output_residual_component_r461'
PARAMETERS = '''
    int mode, int pre_transpose, double epsilon,
    BF16 *values, std::size_t nv, BF16 *norm, std::size_t nn,
    BF16 *projection, std::size_t np, BF16 *output, std::size_t no,
    double *metrics, std::size_t nm, std::uint64_t *memory, std::size_t ns'''
ARGUMENTS = '''mode, pre_transpose, epsilon, values, nv, norm, nn,
        projection, np, output, no, metrics, nm, memory, ns'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('frozen Kimi residual source boundary differs')
    return source.replace(before, after, 1)


def add_capture(source):
    signature = f'extern "C" int {ORIGINAL_EXPORT}({PARAMETERS}) {{'
    capture_parameters = ', BF16 *score_output, std::size_t score_words'
    source = replace_once(source, signature,
                          f'static int residual_component({PARAMETERS}{capture_parameters}, bool capture) {{')
    source = replace_once(source, '    mutable rt::Tensor fused_score_hold;\n',
                          '    mutable rt::Tensor fused_score_hold;\n    rt::Tensor score_output_host;\n')
    source = replace_once(source, '        int device_count = -1;\n', '''        if (capture && (mode != 2 || !score_output || score_words != shape::full_tokens * shape::residuals)) {
            throw std::invalid_argument("invalid Kimi fused score output buffer");
        }
        int device_count = -1;
''')
    return replace_once(source, '        metrics[0] = std::chrono::duration<double>(done - dispatch).count();', '''        if (capture) {
            const rt::Shape score_shape{shape::full_tokens, shape::residuals, 1};
            const auto &scores = probe.fused_score_hold;
            if (!scores || scores->shape() != score_shape || scores->dtype() != type || !scores->is_contiguous()) {
                throw std::runtime_error("Kimi fused score output layout differs");
            }
            probe.score_output_host = host_data(score_output, score_shape);
            probe.score_output_host->copy_from(runtime, scores);
            runtime.syncStream();
        }
        metrics[0] = std::chrono::duration<double>(done - dispatch).count();''') + f'''
{signature}
    return residual_component({ARGUMENTS}, nullptr, 0, false);
}}

extern "C" int {CAPTURE_EXPORT}({PARAMETERS}{capture_parameters}) {{
    return residual_component({ARGUMENTS}, score_output, score_words, true);
}}
'''


def prepare(output):
    if output.is_symlink() or output.exists():
        raise FileExistsError(output)
    output = output.resolve()
    with TemporaryDirectory(prefix='kimi-score-capture-', dir=output.parent) as temp:
        stage = Path(temp) / 'source'
        runpy.run_path(str(BASE / 'prepare.py'))['prepare'](stage)
        manifest_path = stage / 'SOURCE_FILES.json'
        manifest = manifest_path.read_bytes()
        if sha(manifest) != ORIGINAL_MANIFEST:
            raise ValueError('original score pair identity differs')
        pins = json.loads(manifest)['files_SHA256']
        for arm in ARMS:
            name = f'{arm}/component.cpp'
            path = stage / name
            updated = add_capture(path.read_bytes().decode('utf-8')).encode('utf-8')
            path.write_bytes(updated)
            pins[name] = sha(updated)
        manifest = json.dumps({'files_SHA256': pins}, sort_keys=True, indent=2).encode('utf-8') + b'\n'
        manifest_path.write_bytes(manifest)
        stage.rename(output)
    return {'manifest_SHA256': sha(manifest), 'files': len(pins), 'target_compiles': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output', type=Path)
    print(json.dumps(prepare(parser.parse_args().new_output), indent=2, allow_nan=False))
