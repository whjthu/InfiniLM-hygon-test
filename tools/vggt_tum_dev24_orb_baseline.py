"""Measure a fixed calibrated ORB relative-pose baseline on the real TUM development cohort."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from bind_vggt_tum_dev_rgb import CALIBRATION, calibration
from score_vggt_tum_cross_camera_dev import canonical_scorer, read_cases


INPUTS_SHA = '3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6'
WHEEL_SHA = '77a82fe35ddcec0f62c15f2ba8a12ecc2ed4207c17b0902c7a3151ae29f37fb6'
SETTINGS = dict(nfeatures=1500, edgeThreshold=15, seed=20261121,
                minimum_matches=8, minimum_cheirality_inliers=8, probability=.999, threshold_pixels=1.0,
                max_iterations=1000, threads=1)


def pose_from_images(cv, images, intrinsics):
    cv.setRNGSeed(SETTINGS['seed'])
    detector = cv.ORB_create(nfeatures=SETTINGS['nfeatures'], edgeThreshold=SETTINGS['edgeThreshold'])
    features = [detector.detectAndCompute(cv.cvtColor(image, cv.COLOR_RGB2GRAY), None) for image in images]
    (points0, desc0), (points1, desc1) = features
    counts = dict(keypoints=[len(points0), len(points1)], cross_matches=0, ransac_inliers=0, cheirality_inliers=0)
    if desc0 is None or desc1 is None:
        return None, dict(counts, state='NO_DESCRIPTORS')
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(desc0, desc1), key=lambda m:(m.distance,m.queryIdx,m.trainIdx))
    counts['cross_matches'] = len(matches)
    if len(matches) < SETTINGS['minimum_matches']:
        return None, dict(counts, state='TOO_FEW_MATCHES')
    a = np.asarray([points0[m.queryIdx].pt for m in matches], dtype=np.float64)
    b = np.asarray([points1[m.trainIdx].pt for m in matches], dtype=np.float64)
    matrix, mask = cv.findEssentialMat(a,b,intrinsics,method=cv.RANSAC,prob=SETTINGS['probability'],
                                      threshold=SETTINGS['threshold_pixels'],maxIters=SETTINGS['max_iterations'])
    if matrix is None or mask is None or matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0]%3:
        return None, dict(counts, state='NO_ESSENTIAL_MATRIX')
    counts['ransac_inliers'] = int(np.count_nonzero(mask))
    candidates = [cv.recoverPose(part,a,b,intrinsics,mask=mask.copy()) for part in np.split(matrix,matrix.shape[0]//3)]
    inliers, rotation, translation, _ = max(candidates, key=lambda candidate:candidate[0])
    counts['cheirality_inliers'] = int(inliers)
    if inliers < SETTINGS['minimum_cheirality_inliers']:
        return None, dict(counts, state='TOO_FEW_CHEIRALITY_INLIERS')
    extrinsics = np.stack((np.eye(3,4),np.column_stack((rotation,translation.reshape(3)))))
    return extrinsics, dict(counts,state='PREDICTED')


def summary(rows, summarize):
    def score(group):
        value = summarize([dict(sequence=row['sequence'], frame_ids=[row['window']],
                                baseline=row,updated=row) for row in group])['updated']
        return dict(value, planned=len(group), successful=sum(row['pair_error_deg'] is not None for row in group))
    return dict(all_windows=score(rows),by_sequence={seq:score([row for row in rows if row['sequence']==seq])
                                                  for seq in CALIBRATION})


def measure(cv, inputs, *, predictor=pose_from_images):
    scoring = canonical_scorer()
    cases, planned = read_cases(inputs,INPUTS_SHA,scoring['relative_pose'])
    rows = {(row['sequence'],row['window']):dict(sequence=row['sequence'],window=row['window'],
             state='MISSING_INPUT',pair_error_deg=None) for row in planned}
    for item,images,truth in cases:
        seq, window = item['sequence'],item['window']
        fx, fy, cx, cy = calibration(seq)['output_K']
        intrinsics = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
        extrinsics, diagnostic = predictor(cv, images, intrinsics)
        row = rows[(seq,window)]
        row.update(diagnostic,input_SHA256=item['input_SHA256'])
        if extrinsics is not None:
            errors = scoring['angular_errors'](extrinsics,truth)
            if errors['pair_error_deg'] is None:
                row['state']='DEGENERATE_TRANSLATION'
            row.update(errors,extrinsics=extrinsics.tolist())
    ordered = [rows[key] for key in sorted(rows)]
    return ordered, summary(ordered,scoring['summarize_development'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--opencv-site',required=True,type=Path)
    parser.add_argument('--private-output',required=True,type=Path)
    args = parser.parse_args()
    site = args.opencv_site.resolve(strict=True)
    sys.path.insert(0,str(site))
    import cv2
    if not Path(cv2.__file__).resolve().is_relative_to(site) or cv2.__version__ != '4.13.0':
        raise ValueError('expected the isolated OpenCV 4.13 runtime')
    cv2.setNumThreads(SETTINGS['threads'])
    cv2.ocl.setUseOpenCL(False)
    started = perf_counter()
    rows,scores = measure(cv2,args.inputs)
    report = dict(inputs_SHA256=INPUTS_SHA, planned_windows=len(rows), attempted_pairs=sum('input_SHA256' in r for r in rows),
                  implementation='ORB_Hamming_crosscheck_Essential_Matrix_RANSAC_recoverPose',settings=SETTINGS,
                  opencv_version=cv2.__version__,opencv_wheel_SHA256=WHEEL_SHA,numpy_version=np.__version__,
                  wall_seconds=perf_counter()-started, formal_CO3D=False,model_predictions=0,training_updates=0,
                  GPU_used=False,summary=scores)
    private = json.dumps(dict(report,rows=rows),allow_nan=False,indent=2).encode()+b'\n'
    with args.private_output.open('xb') as stream:
        stream.write(private)
    print(json.dumps(dict(report,private_receipt_SHA256=hashlib.sha256(private).hexdigest()),allow_nan=False))


if __name__ == '__main__':
    main()
