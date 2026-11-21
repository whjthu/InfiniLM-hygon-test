"""Independent Fréchet Inception Distance computation on frozen feature arrays."""

from hashlib import sha256
import io
import math


DIMENSION = 2048
MAX_ROWS = 50_000
POLICY = 'FID/pool_3/unbiased-sample-covariance-v1'
ARMS = ('A100', 'Hygon', 'Hygon_before')
ABS_TOL = 1e-7


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def fit(raw, count):
    """Compute a factor F such that F.T @ F is the sample covariance."""
    import numpy as np

    path = io.BytesIO(raw)
    version = np.lib.format.read_magic(path)
    require(version in ((1, 0), (2, 0)), 'unsupported FID NPY header')
    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, _, dtype = header(path, max_header_size=4096)
    require(shape == (count, DIMENSION) and dtype.kind == 'f' and dtype.itemsize in (4, 8)
            and path.tell() + math.prod(shape) * dtype.itemsize == len(raw), 'invalid FID feature array shape or bytes')
    path.seek(0)
    features = np.load(path, allow_pickle=False, max_header_size=4096)
    require(np.isfinite(features).all(), 'nonfinite FID feature array')
    centered = features.astype(np.float64, copy=False)
    mean = centered.mean(axis=0)
    centered -= mean
    centered /= math.sqrt(count - 1)
    factor = np.linalg.qr(centered, mode='r') if count > DIMENSION else centered
    return mean, factor


def distance(left, right):
    import numpy as np

    m1, f1 = left
    m2, f2 = right
    mean = float(np.dot(m1 - m2, m1 - m2))
    # The singular values of F1 F2.T are the square roots of the nonzero
    # eigenvalues of the product of the two sample covariances.
    trace = float(np.vdot(f1, f1) + np.vdot(f2, f2))
    coupling = float(np.linalg.svd(f1 @ f2.T, compute_uv=False).sum())
    value = mean + trace - 2 * coupling
    require(math.isfinite(value) and value >= -1e-10 * max(1., mean + trace),
            'invalid FID from frozen features')
    return max(0., value)


def verify_fid(row, identity, planned, inventory, checkpoints, verify_json, read_bytes):
    require(callable(read_bytes) and 'fid_inputs' in inventory, 'missing independently readable FID features')
    manifest = verify_json(inventory['fid_inputs'])
    require(all(manifest.get(key) == value for key, value in identity.items())
            and manifest.get('policy') == POLICY
            and manifest.get('sample_plan_SHA256') == row['sample_plan']['SHA256']
            and manifest.get('checkpoint_SHA256') == checkpoints, 'FID features belong to a different plan or checkpoint')
    extractor = manifest['feature_extractor']
    require(extractor['SHA256'] == identity.get('scorer_SHA256'), 'FID feature extractor is not the paired scorer')
    spec = verify_json(extractor)
    require(spec.get('policy') == POLICY and spec.get('layer') == 'pool_3' and spec.get('dimension') == DIMENSION
            and all(sha(spec.get(field)) for field in ('implementation_SHA256', 'weights_SHA256', 'preprocess_SHA256')),
            'FID scorer source, weights or image preprocessing missing')

    def features(artifact, images):
        require(isinstance(images, dict) and 2 <= len(images) <= MAX_ROWS
                and all(isinstance(name, str) and name.strip() for name in images)
                and all(map(sha, images.values()))
                and isinstance(artifact, dict) and sha(artifact.get('SHA256')), 'invalid frozen FID feature inventory')
        # Permit f32 or f64, with space for the bounded NumPy header.
        raw = read_bytes(artifact, len(images) * DIMENSION * 8 + 8192)
        require(sha256(raw).hexdigest() == artifact['SHA256'], 'FID feature body differs')
        return fit(raw, len(images))

    real = manifest['reference']
    real_images = planned.get('fid_reference_image_SHA256')
    require(isinstance(real_images, dict) and real['image_SHA256'] == real_images
            and real['sample_ids'] == sorted(real_images),
            'FID real reference image identities or row order differ')
    reference = features(real['features'], real_images)
    require(manifest['arms'].keys() == checkpoints.keys(), 'FID checkpoint feature arms differ')
    for arm in ARMS:
        source = manifest['arms'][arm]
        images = {item['sample_id']: item['artifact']['SHA256'] for item in inventory['arms'][arm]}
        require(row[arm + '_failed'] == 0 and source['sample_ids'] == planned['sample_ids']
                and source['image_SHA256'] == images, 'FID requires complete pinned generated outputs in plan order')
        value = distance(reference, features(source['features'], images))
        require(math.isclose(row[arm], value, rel_tol=1e-8, abs_tol=ABS_TOL),
                'FID differs from the independently recomputed feature distance')
