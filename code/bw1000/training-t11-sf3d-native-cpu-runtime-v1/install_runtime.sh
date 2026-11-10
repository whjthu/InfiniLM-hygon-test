set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MAX_JOBS=2 USE_CUDA=0 USE_METAL=0
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
python3 -B -m pip install --no-deps --no-compile --no-build-isolation --target /runtime/site /runtime/inputs/*.whl /runtime/inputs/*.tar.gz
export PYTHONPATH=/runtime/site
mkdir /runtime/native-wheels
for extension in texture_baker uv_unwrapper; do
    python3 -B -m pip wheel --no-deps --no-build-isolation --wheel-dir /runtime/native-wheels /runtime/"$extension"
done
python3 -B -m pip install --no-deps --no-compile --target /runtime/site /runtime/native-wheels/*.whl
export PYTHONPATH=/runtime/site:/sf3d-source
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SF3D_OFFLINE_INIT=1
exec python3 -B -u /runtime/smoke.py
