"""Render a source-only packet for the independent target zero-GRES Arrow probe."""
import argparse
import json
from pathlib import Path

import binding
import lineage
import native_route
import probe


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ORIGINALS={
    "slot_guard.py":"code/bw1000/training-internvl-gqa56-first2-native-candidate-v1/slot_guard.py",
    "internvl_gqa_training_roster.py":"code/bw1000/training-domain-pilot-v1/internvl_gqa_training_roster.py",
    "sampler_v6.py":"code/bw1000/training-internvl-gqa56-native-sampler-CPU-v6/probe.py",
    "original-launch.json":"code/bw1000/training-internvl-usgs72-native-first2-hip-v1/original-launch.json",
}


def packet():
    files={name:(HERE/name).read_bytes() for name in ("lineage.py","binding.py","native_route.py","probe.py")}
    for name,path in ORIGINALS.items():
        files[name]=binding.pin(REPO/path,65536,binding.FILES[name],exact=False)
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    files["SOURCE_FILES.json"]=lineage.encode(dict(
        scope="InternVL GQA56 metadata-only DataLoaderShard and Arrow CPU v3 probe",
        actual_gpu=False,training_authorized=False,original_4643_result_SHA256=lineage.SAMPLER_RESULT_SHA,
        original_4643_logical_matrix_SHA256=lineage.LOGICAL_REFERENCE_SHA,
        previous_failed_step=probe.PREVIOUS_FAILED_STEP,previous_request=probe.PREVIOUS_REQUESTS[-1],
        previous_actual_failure_archive_SHA256=probe.PREVIOUS_ARCHIVE_SHA,
        previous_actual_result_SHA256=probe.PREVIOUS_RESULT_SHA,
        dataset_routing_method_SHA256=native_route.ROUTING_METHOD_SHA,
        dataset_routing_witness_SHA256=probe.ROUTE_WITNESS_SHA,
        cpu_argument_source_SHA256=native_route.CPU_ARGUMENT_SOURCE[1],
        cpu_argument_witness_SHA256=probe.CPU_ARG_WITNESS_SHA,
        files={name:dict(bytes=len(raw),SHA256=lineage.digest(raw)) for name,raw in files.items()}))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    files=packet()
    if args.output is not None:
        args.output.mkdir(mode=0o700)
        for name,raw in files.items():
            with (args.output/name).open("xb") as output:
                output.write(raw)
    print(json.dumps(dict(files=len(files),bytes=sum(map(len,files.values())),
                          manifest_SHA256=lineage.digest(files["SOURCE_FILES.json"]),target_launched=False),sort_keys=True))


if __name__=="__main__":
    main()
