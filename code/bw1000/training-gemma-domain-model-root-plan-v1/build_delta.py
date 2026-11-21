"""Explicit Gemma-domain model root; preserve the common translator and training math."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

BUILDER = ('tools/prepare_gemma_domain_pair.py', '025ef1f9db66e967726ce5225194de536f47fa96620d5becdc17ffeb678a07d5')
MODEL_FILES = ('code/bw1000/training-gemma-original-model-v1/MODEL_FILES.json',
               '5c6875cfe190980cad4315703e8d73dbd1fb803d620ea1f707ba35fe5de30c63')
MODEL_RECEIPT = ('coordination/training/gemma-target-original-model16-v1.json',
                 '46428e251757e386835e2832a32308a6a07178d9ef776a3ea1b70062c5cf77b3')

ADDITION = '''
# This explicit binding belongs to the new Gemma domain recipe only.
DOMAIN_MODEL_FILES = %r
DOMAIN_MODEL_RECEIPT = %r


def bound_domain_model_directory(repo):
    files = json.loads(pinned(repo, *DOMAIN_MODEL_FILES))
    receipt = json.loads(pinned(repo, *DOMAIN_MODEL_RECEIPT))
    expected = PurePosixPath(files["target_root"]) / PurePosixPath(files["source_root"]).name
    actual = PurePosixPath(receipt["target_model_directory"])
    if (not within_own_root(actual) or actual != expected
            or receipt["actual_full_target_weight_SHA_verified"] is not True
            or receipt["case"] != "T02.sft" or len(files["files"]) != 16
            or set(files["files"]) != set(receipt["files"])):
        raise ValueError("original Gemma model root/identity receipts disagree")
    for name, entry in files["files"].items():
        if (PurePosixPath(name).name != name or name in {".", ".."}
                or any(receipt["files"][name][k] != entry[k] for k in ("bytes", "SHA256"))):
            raise ValueError("original Gemma model16 member identities disagree")
    return str(actual)

''' % (MODEL_FILES, MODEL_RECEIPT)


def patched_builder(repo):
    raw = (repo / BUILDER[0]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUILDER[1]:
        raise ValueError('Frozen original Gemma builder differs')
    text = raw.decode('utf8')
    replacements = (
        ('def candidate(repo, target, models, plugins, port):', ADDITION + '\ndef candidate(repo, target, models, plugins, port):'),
        ('    sources = build_sources(repo)', '    domain_model_directory = bound_domain_model_directory(repo)\n    sources = build_sources(repo)'),
        ('        argv = plan["command"]', '''        argv = plan["command"]
        if argv.count("--model") != 1:
            raise ValueError("Gemma domain plan requires exactly one model option")
        argv[argv.index("--model") + 1] = domain_model_directory
        plan["required_assets"]["model_directory"] = domain_model_directory'''),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Frozen Gemma insertion anchor differs')
        text = text.replace(old, new, 1)
    data = text.encode('utf8')
    compile(data, 'tools/prepare_gemma_domain_pair.py', 'exec')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = patched_builder(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'bytes':len(raw), 'SHA256':hashlib.sha256(raw).hexdigest(),
                      'weights_read':False, 'actual_updates':0, 'training_READY':False}, sort_keys=True))


if __name__ == '__main__':
    main()
