"""Fixed model-free T09 single-file bind scope."""
from pathlib import Path

REQUEST='c6025678606b4170ba7f9fbb5bf0a34d'
LABEL='t09-miopen-single-bind-'+REQUEST[:12]
PARENT='1103606'
NODE='f13r3n13'
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
IMAGE=R/'containers/dtk26-sandbox-r1'
STAGE=R/'incoming'/LABEL
PREVIOUS_ROOT=R/'results/training-hunyuanvideo-t09-sft-repeat2-ro393-v2/src'
PREVIOUS_SHA='1afde5eb02dc21eed84584d933f1ee0c58253700e5e90533b4c1ec409fe85279'
ARCHIVE_SHA='29240578320fe40a1497bd0c45c7bde095e7ff6923acc838e68663c26433f7d3'
EXPECTED=Path('/opt/dtk/lib/libMIOpen.so.1.0')
EXPECTED_PIN=dict(bytes=18705784,SHA256='1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba')
TORCH_ROOT='/usr/local/lib/python3.10/dist-packages/torch'
TORCH_VERSION_PIN='d324ffd50535bca68131043b76605e36d588a85ec41b2aed9d798ece471aa534'
PRIVATE_LEAVES=('.config/miopen/user-db','.config/miopen/miopen-lockfiles','.cache/miopen')


def need(ok,reason):
    if not ok:raise ValueError(reason)
