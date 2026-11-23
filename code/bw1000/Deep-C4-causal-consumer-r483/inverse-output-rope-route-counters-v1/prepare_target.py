"""Render the pinned zero-DCU TU15 build materials for the route counters."""

from hashlib import sha256
import json
from pathlib import Path

from render import checked_replace, render


SCOPE = Path(__file__).resolve().parent
PREVIOUS = SCOPE.parent / "inverse-output-rope-width-caller-v1"
LABEL = "deep-rope-route-tu15-cpu-v1"
REQUEST = "deep-rope-route-tu15-build-20261123-v1"
LIBRARY = "libDeep_inverse_rope_width_routes.so"
ROUTE_HEADER = "inverse_output_route_counts.hpp"
EXPECTED = {
    "CLAIM.json": "ad81ecc4c53e89a41443e94594bde6ae09314cb43e05dff11d1227826673a6c9",
    "build_native.py": "e3ccedc9f23f70abefeabf7aa1950700593429c796f81cc65f9c7e9e792005ec",
    "inside.sh": "cf617311815df5399df109902b7a67dd75de46b9a2729decb8e2d7e14c47d0f9",
    "inverse_output_width_policy.hpp": "82d635ce4671006ef01f3138568f9b0456693f67e171019f373f46d16ff89a6a",
    ROUTE_HEADER: "0ab7d16d60288818c90c1e47875a3149f852492e97eb1506f43b883cb8194cfd",
}


def pinned(path):
    data = path.read_bytes()
    if sha256(data).hexdigest() != EXPECTED[path.name]:
        raise ValueError(f"Frozen material differs: {path.name}")
    return data


def materials():
    claim = json.loads(pinned(PREVIOUS / "CLAIM.json"))
    source_name = "deepseek_v4_attention_prepare.cpp"
    source = render((PREVIOUS / "native-src" / source_name).read_bytes())
    if sha256(source).hexdigest() != "456a32b4add64b811abd54665e4a150c4ffe435d7c231867df109482ac69dcf9":
        raise ValueError("Rendered TU15 differs")
    build = pinned(PREVIOUS / "build_native.py").decode()
    replacements = (
        ('"deep_inverse_rope_h1_pack64_v1"}',
         '"deep_inverse_rope_h1_pack64_v1",\n           "deep_inverse_output_rope_route_counts_v1"}'),
        ('    verify(deps)\n    report["expected_dependencies"]',
         f'    route_header = SCOPE / {ROUTE_HEADER!r}\n'
         '    deps[str(route_header)] = claim["sources"][route_header.name]\n'
         '    need(len(deps) == 988, "Target dependency count differs")\n'
         '    verify(deps)\n    report["expected_dependencies"]'),
        ('"libDeep_inverse_rope_width_caller.so"', f'"{LIBRARY}"'),
        ('Candidate strong exports differ from parent plus the two width providers',
         'Candidate strong exports differ from parent plus the widths and route counter'),
    )
    for before, after in replacements:
        build = checked_replace(build, before, after)
    compile(build, "build_native.py", "exec")
    files = {
        "build_native.py": build.encode(),
        "inside.sh": pinned(PREVIOUS / "inside.sh"),
        source_name: source,
        "inverse_output_width_policy.hpp": pinned(PREVIOUS / "native-src" / "inverse_output_width_policy.hpp"),
        ROUTE_HEADER: pinned(SCOPE / ROUTE_HEADER),
        "launch_once.py": (SCOPE / "launch_once.py").read_bytes(),
    }
    manifest = {name: sha256(data).hexdigest() for name, data in files.items()}
    if (claim["parent"], claim["node"], claim["cpus"], claim["memory_mib"], claim["slurm_seconds"],
            claim["GPU_steps"]) != ("1103606", "f12r2n13", 4, 8192, 300, 0):
        raise ValueError("Previous CPU placement differs")
    claim.update(label=LABEL, request=REQUEST, case="I03.deep-inverse-rope-route-counters",
                 sources={name: manifest[name] for name in (source_name, "inverse_output_width_policy.hpp",
                                                             ROUTE_HEADER)},
                 materials=manifest, runtime_control="CPU build only; GPU and model are separately gated",
                 no_replay=True)
    files["CLAIM.json"] = (json.dumps(claim, sort_keys=True, allow_nan=False) + "\n").encode()
    return files
