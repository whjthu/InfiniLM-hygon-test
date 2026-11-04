"""Only the changed real config/CLI path consumer; no actor/Native retest."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
from unittest.mock import patch

from public_glm_native import cli


def run(prior_path):
    root = Path(__file__).parent
    prior = json.loads(Path(prior_path).read_bytes())
    fixed = json.loads((root / 'config.template.json').read_bytes())
    original = json.loads(json.dumps(fixed))
    assert fixed['Common_python_root'] == str(PurePosixPath(fixed['Common_extension']).parent)
    expected_helpers = {
        '/refactor/candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so':
            'e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8',
        '/refactor/candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_cache_r807.so':
            '1bd9ddbe75aad77a1ee59fa45784037a4aa00b7d2a44fc033a0a5cc41803268c'}
    assert fixed['FP8_helpers_SHA256'] == expected_helpers
    assert cli.validate_deployment_paths(fixed) is fixed and fixed == original
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / 'config.json'
        def consume(value):
            raw = (json.dumps(value, indent=2) + '\n').encode()
            config.write_bytes(raw)
            output = io.StringIO()
            # Only package-source verification is a recorded leaf. Config file,
            # byte hash, real main parser/schema/path consumer are executed.
            with patch.object(cli, 'verify_package', return_value=root), \
                 patch.object(cli, 'prepare_public', side_effect=AssertionError('GPU import must not run')), \
                 patch.object(cli, 'configure_process', side_effect=AssertionError('runtime must not run')), \
                 patch.object(cli, 'verify_native_chain', side_effect=AssertionError('Native must not run')), \
                 redirect_stdout(output):
                cli.main(['--config', str(config), '--config-sha', hashlib.sha256(raw).hexdigest(),
                    '--source-sha', '0' * 64, '--stage', '1', '--run', 'paths-only',
                    '--actual-step', 'CPU.recording', '--gpu-start-epoch', '0'])
            return json.loads(output.getvalue())
        row = consume(fixed)
        assert row['source_and_schema_valid'] and not row['Model_READY'] and not row['Native_dlopen']
        cases.append(dict(case='actual_main_consumes_POSIX_template_without_GPU_or_Native', passed=True))
        try: consume(prior)
        except ValueError as error: assert 'Common_python_root' in str(error)
        else: raise AssertionError('original backslash config accepted')
        cases.append(dict(case='original_v1_Common_python_root_rejected_before_runtime', passed=True))
        for old_key, digest in prior['FP8_helpers_SHA256'].items():
            broken = json.loads(json.dumps(fixed))
            correct_key = next(k for k, v in expected_helpers.items() if v == digest)
            del broken['FP8_helpers_SHA256'][correct_key]
            broken['FP8_helpers_SHA256'][old_key] = digest
            try: consume(broken)
            except ValueError as error: assert 'FP8_helpers_SHA256 key' in str(error)
            else: raise AssertionError('original backslash helper key accepted')
            cases.append(dict(case='original_helper_key_rejected_' + PurePosixPath(correct_key).name, passed=True))
        broken = dict(fixed, Common_python_root='refactor/relative')
        try: consume(broken)
        except ValueError: pass
        else: raise AssertionError('relative deployment path accepted')
        cases.append(dict(case='relative_path_rejected_before_runtime', passed=True))
    assert fixed == original and fixed['FP8_helpers_SHA256'] == expected_helpers
    return dict(passed=True, cases=cases, changed_consumer='actual cli.main and validate_deployment_paths',
        provider_dictionary_exact_equal=True, configuration_mutated=False,
        recording_leaf='package source-manifest verification only; full final manifest checked separately',
        prior_config_SHA256=hashlib.sha256(Path(prior_path).read_bytes()).hexdigest(),
        new_config_SHA256=hashlib.sha256((root / 'config.template.json').read_bytes()).hexdigest(),
        GPU=False, Native_dlopen=False, Model_READY=False, jobs=0,
        old_actor_publiclife_and_422_423_424_suites_repeated=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--prior-config', required=True);args = p.parse_args()
    result = run(args.prior_config)
    (Path(__file__).parent / 'paths-CPU-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
