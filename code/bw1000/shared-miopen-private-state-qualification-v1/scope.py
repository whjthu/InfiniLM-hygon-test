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
IDS={
 ('gemma','legacy'):'4ca38feb5c6b48e4a98841fdf20c7a31',
 ('gemma','current'):'aac88850495d4b129df57f92e765383b',
 ('gemma','qualify'):'05762883690542569ef390731d30deab',
 ('gemma','conv'):'e055108d586c4d8eb15b91e82708d0cc',
 ('t09','legacy'):'65d28a1d9f134a4da05fe20e38871483',
 ('t09','current'):'e9ee215aac574bf0897cd126256d4c3d',
 ('t09','qualify'):'83878ace0917438c9097691ad0e0d4bb',
 ('t09','conv'):'a32d102837b24f2fa0ec0acd1cbd2769',
}


def require(ok, reason):
    if not ok:raise ValueError(reason)


def label(role,mode):
    return f'miopen-{role}-{mode}-{IDS[role,mode][:12]}'


def staged(role,mode):
    return R/'incoming'/label(role,mode)


def layout_for(observed):
    """Unknown/unsafe ancestry blocks; measured missing parent permits private HOME."""
    parents=observed['candidate_parents']
    require(set(parents)=={'config','cache'} and
            all(state in ('DIRECTORY','ABSENT') for state in parents.values()),'EXACT_MEASURED_ORIGINAL_PARENTS_REQUIRED')
    return 'PRIVATE_HOME' if 'ABSENT' in parents.values() else 'NARROW_MIOPEN_SUBTREES'
