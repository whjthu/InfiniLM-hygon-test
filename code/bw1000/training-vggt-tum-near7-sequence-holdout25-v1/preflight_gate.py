"""Bind the genuine .922 CPU result and its read-only 398-file candidate."""

from hashlib import sha256
import json
from pathlib import Path


REQUEST = "3653b95925860e0dd3f9bbfb9f1b1fde"
CPU_LABEL = "training-vggt-tum-three-recording-native-preflight-v2-"+REQUEST
CPU_RESULT_SHA = "c67b9b17ca338bebcf0b131e25922ff8f864cbaecf46feb96525674d0ead96c2"
CPU_STDOUT_SHA = "cbdd5cb21326b88867ce4ac499e636e0484a39e80bb5019a25ca2196bdf683d0"
CPU_PUBLIC_SHA = "0b874bdfa1373008bd83fd7eaf7ff1b8cc2ffb08617323eff56f650ebaec74a4"
INSTALL_SHA = "63c58c97468f14e3341a287fb452d7b99c245c19629da3e878055e6936b6baba"
ORDER_SHA = "432012b6fce5fe874dd5883df184cd50f4f6d7dfb45f4b2699536839100c44d0"
VIEW_SHA = "0e1f36ecf41b1ecdc4d898e851a4b12572994d1c8ba8eed701291a1d7423aaba"
TENSOR_SHA = "86cfe8e0fd2f68404a0ab40ea23631836f8759d27efaa8fbca044f50f2553d7a"
BUNDLE_SHA = "647b78e286687cc1d55670f64472fbe0f395c1b81cbeb928999fcf415f6a59f7"
FRAMEWORK_SHA = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
RECORDINGS = {"freiburg3_structure_texture_near":7,"freiburg3_nostructure_texture_far":1,"freiburg3_sitting_xyz":1}


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def identifier(value):
    return digest((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())


def bind_preflight(results, public_file, candidate):
    from sealed_preflight_v2 import load_v1

    read=load_v1(Path(__file__).with_name("sealed_preflight_v1.py")).read
    original=json.loads(read(results/(CPU_LABEL+"-controller")/"original-result.json",CPU_RESULT_SHA,65536))
    public=json.loads(read(public_file,CPU_PUBLIC_SHA,65536))
    installed=json.loads(read(candidate/"install.json",INSTALL_SHA,8192))
    native=original["result"]
    need(original["returncode"]==0 and original["errors"]==[] and original["stdout_SHA256"]==CPU_STDOUT_SHA
         and public["request"]==REQUEST and public["persistent_remote_result_SHA256"]==CPU_RESULT_SHA
         and public["state"]=="ACTUAL_NINE_RGBD_INPUTS_NATIVE_TORCH_CPU_AND_POST_CLEAN" and native==public["native"]
         and [row[0] for row in public["actual_accounting"]]==["1103606.921","1103606.922","1103606.923"]
         and all(row[2:4]==["COMPLETED","0:0"] and row[5]=="f13r4n09" and "gres/" not in row[6]
                 for row in public["actual_accounting"]),
         "predecessor is not the actual original native CPU and independent clear")
    need(native["state"]=="NINE_REAL_NATIVE_TORCH_CPU_INPUTS_VERIFIED" and native["cpu_step"]=="1103606.922"
         and native["native_real_batches"]==9 and native["asset_files"]==74
         and native["recordings"]==RECORDINGS and native["asset_bundle_SHA256"]==BUNDLE_SHA
         and native["source_framework_manifest_SHA256"]==FRAMEWORK_SHA and native["input_order_SHA256"]==ORDER_SHA
         and native["view_order_SHA256"]==VIEW_SHA and native["batch_tensor_signatures_SHA256"]==TENSOR_SHA
         and native["install_SHA256"]==INSTALL_SHA and native["explicit_gpu_calls"]==native["optimizer_updates"]==0
         and native["training_READY"] is False,
         "preflight data order, tensors, placement or original install differs")
    need(installed["final_framework_members"]==398 and installed["common_framework_original_members"]==393
         and installed["asset_files"]==74 and installed["asset_bundle_SHA256"]==BUNDLE_SHA
         and installed["overlay_SHA256"]==native["overlay_SHA256"],"original installed source scope differs")
    original_framework=json.loads(read(candidate/"FRAMEWORK_FILES.json",FRAMEWORK_SHA,1<<18))["files"]
    overlays={"generation/src/gencuda/"+key:pin for key,pin in native["overlay_SHA256"].items()}
    for name,metadata in original_framework.items():
        if name not in overlays:
            read(candidate/"framework"/name,metadata["SHA256"],2<<20,metadata["bytes"])
    for name,pin in overlays.items():
        read(candidate/"framework"/name,pin,70000)
    need(len(original_framework)==393 and len(original_framework.keys()|overlays.keys())==398,
         "read-only common and overlay member closure differs")
    read(candidate/"data/bundle.json",BUNDLE_SHA,2048)
    return dict(preflight_request=REQUEST,preflight_cpu_step="1103606.922",preflight_post_step="1103606.923",
                preflight_result_SHA256=CPU_RESULT_SHA,preflight_public_SHA256=CPU_PUBLIC_SHA,
                preflight_install_SHA256=INSTALL_SHA,view_order_SHA256=VIEW_SHA,
                batch_tensor_signatures_SHA256=TENSOR_SHA,asset_bundle_SHA256=BUNDLE_SHA)
