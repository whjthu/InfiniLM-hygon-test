"""Real frozen source/metadata plans; never inspect model bodies or target state."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time

from build_delta import BUILDER, MODEL_RECEIPT, patched_builder


def pin(raw):
    return {'bytes':len(raw), 'SHA256':hashlib.sha256(raw).hexdigest()}


def namespace(raw, filename):
    ns = {'__name__':'gemma_plan_source_check', '__file__':filename}
    exec(compile(raw, filename, 'exec'), ns)
    return ns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    repo = args.repo.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(mode=0o700)
    sys.path.insert(0, str(repo/'tools'))
    original = (repo/BUILDER[0]).read_bytes()
    revised = patched_builder(repo)
    new = namespace(revised, str(repo/BUILDER[0]))
    old = namespace(original, str(repo/BUILDER[0]))
    receipt = json.loads((repo/MODEL_RECEIPT[0]).read_bytes())
    actual = PurePosixPath(receipt['target_model_directory'])
    target = PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-render72-exposure25-resume20-v1')
    plugins = target/'framework/vlm/plugins'
    checks = []
    def check(name, truth):
        checks.append({'name':name,'passed':bool(truth)})
    old_sources, mistaken = old['candidate'](repo, target, actual, plugins, 29920)
    new_sources, fixed = new['candidate'](repo, target, actual, plugins, 29920)
    _, correct_original = old['candidate'](repo, target, actual.parent, plugins, 29920)
    check('actual_two_pinned_manifests_bind_original_qualified_directory', new['bound_domain_model_directory'](repo) == str(actual))
    check('both_arms_unique_model_argv_and_required_assets_match', all(
        plan['command'].count('--model') == 1
        and plan['command'][plan['command'].index('--model')+1] == str(actual)
        and plan['required_assets']['model_directory'] == str(actual) for plan in fixed.values()))
    check('wrong_parent_argument_reproduces_old_double_basename', all(
        plan['required_assets']['model_directory'] == str(actual/actual.name) for plan in mistaken.values()))
    restored = copy.deepcopy(fixed)
    for arm, plan in restored.items():
        argv = plan['command']
        argv[argv.index('--model')+1] = mistaken[arm]['command'][mistaken[arm]['command'].index('--model')+1]
        plan['required_assets']['model_directory'] = mistaken[arm]['required_assets']['model_directory']
    check('only_model_argv_and_required_asset_differ_from_misconfigured_original', restored == mistaken)
    # Different legacy --models arguments also produce a different informational
    # MODELS_DIR environment value in the original translator. The new builder
    # leaves that original behavior untouched; normalize only that caller value
    # when comparing against the correctly configured old invocation.
    for arm in correct_original:
        correct_original[arm]['environment_overrides']['MODELS_DIR'] = fixed[arm]['environment_overrides']['MODELS_DIR']
    check('correct_original_equivalent_except_legacy_caller_MODELS_DIR',
          json.dumps(fixed,sort_keys=True) == json.dumps(correct_original,sort_keys=True))
    check('all_six_framework_overlay_bytes_unchanged', new_sources == old_sources and len(new_sources) == 6)
    # Execute only the real verifier metadata-prefix, stopping before its file-stat loop.
    # No Path.stat/lstat or PT/safetensors open can occur in this truncated check.
    worker_raw = (repo/'tools/run_gemma_domain_pair.py').read_bytes()
    tree = ast.parse(worker_raw)
    method = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_model')
    stop = next(i for i,n in enumerate(method.body) if isinstance(n,ast.For))
    method.body = method.body[:stop]
    module = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    def require(ok, why):
        if not ok: raise ValueError(why)
    # The admitted worker is Linux. On Windows only this metadata constructor
    # must retain POSIX path spelling; repo remains a real local Path object.
    ns = dict(Path=PurePosixPath,json=json,hashlib=hashlib,require=require,MODEL_BINDING=MODEL_RECEIPT)
    exec(compile(module,'actual_verifier_metadata_prefix_only','exec'),ns)
    ns['verify_model'](repo,fixed)
    check('actual_verify_model_metadata_prefix_accepts_two_fixed_plans',True)
    original_pinned = new['pinned']
    def conflicting(root,path,sha):
        raw = original_pinned(root,path,sha)
        if path == MODEL_RECEIPT[0]:
            value = json.loads(raw)
            value['target_model_directory'] = str(actual.parent)
            return json.dumps(value).encode()
        return raw
    new['pinned'] = conflicting
    try:
        new['bound_domain_model_directory'](repo)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check('synthetic_semantic_root_conflict_rejected_no_hash_qualification_claim',rejected)
    check('original_builder_bytes_unchanged',pin((repo/BUILDER[0]).read_bytes()) == pin(original))
    result = dict(state='GEMMA_MODEL_ROOT_DOMAIN_PLAN_LOCAL_SOURCE_VERIFIED',
                  passed=all(x['passed'] for x in checks),checks=checks,
                  original_builder=pin(original),revised_builder=pin(revised),
                  verifier_metadata_prefix_source=pin(worker_raw),
                  candidate_framework_files=396,overlay_files=6,
                  model_directory=str(actual),plans={k:pin(json.dumps(v,sort_keys=True).encode()) for k,v in fixed.items()},
                  verify_model_weight_stat_loop_executed=False,weights_read=False,
                  actual_target_calls=0,actual_updates=0,training_READY=False,
                  wall_seconds=time.monotonic()-started)
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    (out/'result.json').write_bytes(raw)
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'result':pin(raw)},sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
