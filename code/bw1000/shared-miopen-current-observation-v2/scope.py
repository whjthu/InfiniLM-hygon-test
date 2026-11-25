"""Exact role matrix for the paired read-only MIOpen experiments."""
from pathlib import Path

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
HOME=R.parent
IMAGE=R/'containers/dtk26-sandbox-r1'
LIBRARY=Path('/opt/dtk/lib/libMIOpen.so.1.0')
LIBRARY_BYTES=18705784
LIBRARY_SHA='1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba'
ARCHIVE_SHA='519f9673841964969dbcd93dee2ac4bde0f0ad5788651c37b10bf187da921b62'
PARENT='1103606'
DEPS=('MIOPEN_USER_DB_PATH','MIOPEN_SYSTEM_DB_PATH','MIOPEN_CUSTOM_CACHE_DIR',
      'MIOPEN_LOCKFILE_DIR','MIOPEN_LOCKFILE_PATH','XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME')
RUNTIME_VARS=('MIOPEN_FIND_MODE','MIOPEN_DISABLE_CACHE','MIOPEN_DEBUG_DISABLE_FIND_DB')
ROLES={
 'gemma':dict(node='f13r1n12',historical_step='1103606.1212',current_step='1103606.3562',
              historical=R/'results/training-gemma-usgs72-matched-dev36-supervisor-v1/src',
              historical_sha='8653640636a86d6d2daf1841f23a6dd889f21bf26e9dd628f3548f478fcc180a',
              current=R/'incoming/training-gemma-usgs72-blind36-8f542497578e-v3-source',
              current_sha='acc1eb853c94c3272f4127450bc7f9c7bbf6a6c4c34869a5d826eee5aeb8d6a5'),
 't09':dict(node='f13r3n13',historical_step='1087656.794',current_step='1103606.3548',
            historical=R/'results/training-hunyuanvideo-cached-native25-target-v3/src',
            historical_sha='aff3258198af45367e9d4b260b4052c58e42a4e8490b44a0fa936ff971c3b314',
            current=R/'results/training-hunyuanvideo-t09-sft-repeat2-ro393-v2/src',
            current_sha='1afde5eb02dc21eed84584d933f1ee0c58253700e5e90533b4c1ec409fe85279'),
}
IDS={'gemma': 'c2eba8eaa03f47bfbfcebe7e44776cea', 't09': '39f14f38443d4712b881cbcba6ec07d4'}


def require(ok,reason):
    if not ok: raise ValueError(reason)


def label(role,_mode='current'):
    return f'miopen-{role}-current-v2-{IDS[role][:12]}'


def staged(role,mode='current'):
    return R/'incoming'/label(role,mode)
