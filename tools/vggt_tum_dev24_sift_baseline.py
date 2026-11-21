"""Compare SIFT matching with the archived ORB baseline on the same TUM development windows."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from vggt_tum_dev24_orb_baseline import INPUTS_SHA, SETTINGS, WHEEL_SHA, measure, summary
from score_vggt_tum_cross_camera_dev import canonical_scorer


ORB_RECEIPT = '34c378231b2cb8456b39c8a24f781c99b78deee5bcf4e1e2db62681cf2a13ab7'
RATIO = 0.8


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def ratio_matches(matcher, first, second):
    return {pair[0].queryIdx: pair[0] for pair in matcher.knnMatch(first, second, k=2)
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance}


def predict(cv, images, intrinsics):
    cv.setRNGSeed(SETTINGS['seed'])
    detector = cv.SIFT_create(nfeatures=SETTINGS['nfeatures'])
    (points0, desc0), (points1, desc1) = [
        detector.detectAndCompute(cv.cvtColor(image, cv.COLOR_RGB2GRAY), None) for image in images
    ]
    counts = dict(keypoints=[len(points0), len(points1)], mutual_ratio_matches=0,
                  ransac_inliers=0, cheirality_inliers=0)
    if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
        return None, dict(counts, state='NO_DESCRIPTORS')
    matcher = cv.BFMatcher(cv.NORM_L2)
    forward = ratio_matches(matcher, desc0, desc1)
    backward = ratio_matches(matcher, desc1, desc0)
    matches = sorted((match for match in forward.values()
                      if match.trainIdx in backward and backward[match.trainIdx].trainIdx == match.queryIdx),
                     key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
    counts['mutual_ratio_matches'] = len(matches)
    if len(matches) < SETTINGS['minimum_matches']:
        return None, dict(counts, state='TOO_FEW_MATCHES')
    first = np.asarray([points0[match.queryIdx].pt for match in matches], dtype=np.float64)
    second = np.asarray([points1[match.trainIdx].pt for match in matches], dtype=np.float64)
    # Match the original ORB pose estimator and settings. Only the correspondence method differs.
    matrix, mask = cv.findEssentialMat(first, second, intrinsics, method=cv.RANSAC, prob=SETTINGS['probability'],
                                      threshold=SETTINGS['threshold_pixels'], maxIters=SETTINGS['max_iterations'])
    if matrix is None or mask is None or matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] % 3:
        return None, dict(counts, state='NO_ESSENTIAL_MATRIX')
    counts['ransac_inliers'] = int(np.count_nonzero(mask))
    candidates = [cv.recoverPose(part, first, second, intrinsics, mask=mask.copy())
                  for part in np.split(matrix, matrix.shape[0] // 3)]
    inliers, rotation, translation, _ = max(candidates, key=lambda item: item[0])
    counts['cheirality_inliers'] = int(inliers)
    if inliers < SETTINGS['minimum_cheirality_inliers']:
        return None, dict(counts, state='TOO_FEW_CHEIRALITY_INLIERS')
    return np.stack((np.eye(3, 4), np.column_stack((rotation, translation.reshape(3))))), dict(counts, state='PREDICTED')


def compare(old, new, new_scores):
    def ident(row):
        return row['sequence'], row['window']

    archived = {ident(row): row for row in old['rows']}
    observed = {ident(row): row for row in new}
    require(len(archived) == len(old['rows']) == len(observed) == len(new) == 24
            and archived.keys() == observed.keys() and old['inputs_SHA256'] == INPUTS_SHA
            and old['opencv_wheel_SHA256'] == WHEEL_SHA, 'frozen ORB and SIFT cohorts or runtime differ')
    require(summary(old['rows'], canonical_scorer()['summarize_development']) == old['summary'],
            'original ORB summary differs from its per-window errors')
    joint = recovered = lost = better = worse = tied = 0
    for key, row in observed.items():
        reference = archived[key]
        require(row.get('input_SHA256') == reference.get('input_SHA256'), 'paired image identity differs')
        a, b = reference['pair_error_deg'], row['pair_error_deg']
        if a is None:
            recovered += b is not None
        elif b is None:
            lost += 1
        else:
            joint += 1
            better += b < a
            worse += b > a
            tied += b == a
    original, current = old['summary']['all_windows'], new_scores['all_windows']
    delta = {name: current[name] - original[name] for name in current if name.startswith('development_AUC_at_')}
    return dict(joint_success=joint, only_SIFT_success=recovered, only_ORB_success=lost,
                SIFT_smaller_error=better, SIFT_larger_error=worse, tied_error=tied, delta_AUC=delta,
                original_ORB_summary=old['summary'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'opencv-site', 'opencv-wheel', 'original-orb', 'private-output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    site = args.opencv_site.resolve(strict=True)
    with args.opencv_wheel.open('rb') as wheel:
        require(sha256(wheel.read(42 * 1024**2)).hexdigest() == WHEEL_SHA and wheel.read(1) == b'',
                'OpenCV wheel differs from the frozen runtime')
    with args.original_orb.open('rb') as stream:
        prior = stream.read(128 * 1024)
    require(sha256(prior).hexdigest() == ORB_RECEIPT, 'original ORB receipt has changed')
    sys.path.insert(0, str(site))
    import cv2
    require(Path(cv2.__file__).resolve().is_relative_to(site) and cv2.__version__ == '4.13.0',
            'expected frozen OpenCV 4.13 CPU runtime')
    cv2.setNumThreads(SETTINGS['threads'])
    cv2.ocl.setUseOpenCL(False)
    started = perf_counter()
    rows, metrics = measure(cv2, args.inputs, predictor=predict)
    result = dict(scope='REAL_FROZEN_TUM_DEV24_CLASSICAL_RGB_COMPARISON', inputs_SHA256=INPUTS_SHA,
                  source_opencv_wheel_SHA256=WHEEL_SHA, source_ORB_receipt_SHA256=ORB_RECEIPT,
                  matching='SIFT with bidirectional nearest/second-nearest ratio and mutual membership',
                  ratio=RATIO, pose_settings={name: value for name, value in SETTINGS.items() if name != 'edgeThreshold'},
                  actual_model_predictions=0, actual_training_updates=0, actual_GPU_steps=0, formal_CO3D=False,
                  planned_windows=len(rows), attempted_pairs=sum('input_SHA256' in row for row in rows),
                  wall_seconds=perf_counter() - started, summary=metrics, paired=compare(json.loads(prior), rows, metrics))
    private = (json.dumps(dict(result, rows=rows), indent=2, allow_nan=False) + '\n').encode()
    with args.private_output.open('xb') as output:
        output.write(private)
    print(json.dumps(dict(result, private_evidence_SHA256=sha256(private).hexdigest()), allow_nan=False))


if __name__ == '__main__':
    main()
