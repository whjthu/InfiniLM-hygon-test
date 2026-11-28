"""Regenerate a separate continuation owner from the successful .4983 producer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIRST2 = HERE.with_name("training-llava-clevr-first2-owner-v1")
BASE = "0fcebef4691f43ce6dbda091fb251f63456ff63051999a15e2d8ad9152688bd8"
MANIFEST = "dcf03eeaadc8af14c766c5fdb82cf97741add69f60214767d04e97e74a8f9356"
HELPERS = ("snapshot.py", "owned_processes.py", "slurm_accounting.py")


def sha(raw):
    return sha256(raw).hexdigest()


def render():
    exact = (FIRST2 / "SOURCE_FILES.json").read_bytes()
    raw = (FIRST2 / "owner.py").read_bytes()
    if sha(exact) != MANIFEST or sha(raw) != BASE:
        raise ValueError("EXACT_SUCCESSFUL_OWNER_BYTES")
    text = raw.decode()

    def patch(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError("EXACT_4983_OWNER_ANCHOR: " + before[:70])
        text = text.replace(before, after)

    patch("HERE = Path(__file__).resolve().parent\n",
          "HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nimport resume_layout as layout\n")
    patch('SCIENCE = "40b4a9c35489288c633dd6c43a87f84fe315cdcd09b25ad65740d0152bd9a576"',
          "SCIENCE = layout.SCIENCE")
    patch('"training-llava-clevr-first2-" + request[:12]', '"training-llava-clevr-resume25-" + request[:12]')
    patch("    return request\n", "    layout.root_fields(document, need)\n    return request\n")
    patch('(STAGED / "source", "/llava-source", "ro"), (STAGED / "helpers", "/llava-helpers", "ro"),',
          '*layout.mount_arguments(run), (STAGED / "helpers", "/llava-helpers", "ro"),')
    patch('shell, "t04-first2",', 'shell, "t04-resume25",')
    patch('tempfile.mkdtemp(prefix="t04-clevr-", dir="/tmp")', 'tempfile.mkdtemp(prefix="t04-resume25-", dir="/tmp")')
    patch('    owner_sha = write(run / "OWNER.json", owner)',
          '    resume_doc, resume_sha = document(run / "RESUME-AUTH.json")\n'
          '    need(resume_doc == layout.future_auth(auth["request"]), "EXACT_ROOT_GENERATED_RESUME_AUTH")\n'
          '    owner_sha = write(run / "OWNER.json", owner)')
    patch('env = dict(os.environ, LLAVA_OWNER_SHA256=owner_sha, APPTAINERENV_LLAVA_OWNER_SHA256=owner_sha)',
          'env = dict(os.environ, LLAVA_OWNER_SHA256=owner_sha, APPTAINERENV_LLAVA_OWNER_SHA256=owner_sha,\n'
          '                   LLAVA_RESUME_AUTH_SHA256=resume_sha, APPTAINERENV_LLAVA_RESUME_AUTH_SHA256=resume_sha)')
    patch('    here_manifest(STAGED / "source", SCIENCE)',
          '    here_manifest(layout.RESUME_STAGE / "source", SCIENCE)\n'
          '    layout.archive_requirements(pin, need)\n'
          '    layout.independent_cpu(auth, account, matching, tres, pin, need)')
    patch('    run.mkdir(mode=0o700)\n', '    run.mkdir(mode=0o700)\n    write(run / "RESUME-AUTH.json", layout.future_auth(auth["request"]))\n')
    count = text.count("planned_forwards=64") + text.count("planned_rank_updates=16")
    if count != 4:
        raise ValueError("EXACT_PRIOR_TWO_UPDATE_DENOMINATORS")
    text = text.replace("planned_forwards=64", "planned_forwards=736").replace("planned_rank_updates=16", "planned_rank_updates=184")
    patch('[HOST_PYTHON, "-B", str(STAGED / "source/verify.py"), str(run), auth["request"]]',
          '[HOST_PYTHON, "-B", str(layout.RESUME_STAGE / "source/verify.py"), "--run", str(run),\n'
          '                                 "--request", auth["request"], "--plan", str(layout.PLAN),\n'
          '                                 "--plan-sha256", layout.PLAN_SHA]')
    patch('"independent-first2.json"', '"independent-resume25.json"')
    files = {"owner.py": text.encode()}
    proof = json.loads(exact)
    for helper in HELPERS:
        data = (FIRST2 / helper).read_bytes()
        if sha(data) != proof["files"][helper]["SHA256"]:
            raise ValueError("EXACT_SUCCESSFUL_HELPER")
        files[helper] = data
    for name in ("resume_layout.py", "assembly.py"):
        files[name] = (HERE / name).read_bytes()
    for name, data in files.items():
        compile(data, name, "exec")
    value = dict(state="ACTUAL_CPU_ASSEMBLY_AND_FRESH_ROOT_RELEASE_REQUIRED",
                 files={name: dict(bytes=len(data), SHA256=sha(data)) for name, data in files.items()})
    files["SOURCE_FILES.json"] = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = render()
    args.output.mkdir(mode=0o700)
    for name, data in files.items():
        with (args.output / name).open("xb") as stream:
            stream.write(data)
    print(sha(files["SOURCE_FILES.json"]))


if __name__ == "__main__":
    main()
