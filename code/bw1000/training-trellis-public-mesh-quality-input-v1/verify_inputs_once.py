"""One local real-reference audit; no network/model or synthesized prediction score."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from score_once import reference, pin

D = Path(__file__).resolve().parent
G = D.parents[2]
R = G / 'local-only/training-trellis-public-mesh-quality-input-v1-original'
OUT = G / 'local-only/training-trellis-public-mesh-quality-input-v1-reference-audit-original'


def main():
    start = time.monotonic()
    OUT.mkdir()
    raw = (R / 'result.json').read_bytes()
    assert pin(raw) == {'bytes':14607,'SHA256':'a31cfdcc1cbc32798328284622bbeec6b5a9bbfd15cb7c738fe64e094a7798a3'}
    producer = json.loads(raw)
    registry = json.loads((D / 'REFERENCE_INPUTS.json').read_bytes())
    rows = {}; fingerprints = []
    for name, spec in registry['objects'].items():
        assert spec['split'] == producer['objects'][name]['split']
        a = reference(R, spec)
        psha = hashlib.sha256(a['reference_points'].tobytes()).hexdigest()
        vsha = hashlib.sha256(a['vertices'].tobytes()).hexdigest()
        fingerprints.append((psha, vsha))
        rows[name] = {'split':spec['split'],'file':spec['file'],'vertices':len(a['vertices']),
                      'faces':len(a['faces']),'reference_points':len(a['reference_points']),
                      'points_array_SHA256':psha,'vertices_array_SHA256':vsha,'finite':True}
    assert len(set(x[0] for x in fingerprints)) == len(set(x[1] for x in fingerprints)) == 3
    assert {k for k,v in rows.items() if v['split']=='train'} == {'WaterBottle','Corset'}
    assert {k for k,v in rows.items() if v['split']=='dev'} == {'BoomBox'}
    parsed = 0
    for p in D.glob('*.py'):
        ast.parse(p.read_bytes(),filename=str(p)); parsed += 1
    # No prediction supplied: the real dev object must remain failed with its
    # full 2048-point denominator, never become an identity/model score.
    missing = OUT / 'missing-predictions-private.json'
    missing.write_text(json.dumps({'schema':'TRELLIS_PUBLIC_REFERENCE_PREDICTIONS_V1','split':'dev','objects':{}}),encoding='utf8')
    scored = OUT / 'missing-dev-private.json'
    run = subprocess.run([sys.executable,'-B',str(D/'score_once.py'),'--reference-root',str(R),
                          '--predictions',str(missing),'--predictions-SHA256',pin(missing.read_bytes())['SHA256'],
                          '--split','dev','--output',str(scored)],capture_output=True,timeout=15)
    s=json.loads(scored.read_bytes())
    assert run.returncode == 1 and s['planned_objects'] == s['failed_objects'] == 1
    assert s['completed_objects'] == 0 and s['planned_reference_points'] == 2048
    assert s['objects']['BoomBox']['Fscore_at_1percent'] == 0 and not s['model_quality_qualified']
    result={'schema':'TRELLIS_REAL_REFERENCE_NPZ_AND_SCORE_ABI_LOCAL_AUDIT_V1','passed':True,
            'objects':rows,'completed_objects':3,'reference_points':6144,'train_groups':2,'dev_groups':1,'blind_groups':0,
            'exact_cross_group_points_or_vertices_hash_duplicates':0,
            'semantic_near_duplicate_or_new_domain_independence_proven':False,
            'local_NPZ_body_SHA_shapes_finite_checked':True,'producer_result':pin(raw),
            'new_score_missing_prediction_case':{'passed':True,'planned_objects':1,'failed_objects':1,
                  'reference_point_denominator':2048,'actual_model_predictions':0,'artifact':pin(scored.read_bytes())},
            'Python_AST_files':parsed,'wall_seconds':time.monotonic()-start,
            'actual_model_GPU_optimizer_calls':0,'READY':False,'Torch_or_original_A100_operands_read':False}
    output=OUT/'result.json'
    with output.open('x',encoding='utf8') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'objects':3,'reference_points':6144,'audit':pin(output.read_bytes())}))


if __name__ == '__main__':
    main()
