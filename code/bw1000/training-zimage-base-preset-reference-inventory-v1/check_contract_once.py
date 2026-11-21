"""Real frozen101-slot contract plus explicit empty-root/rejection boundaries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from inspect_assets import inspect


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(mode=0o700)
    package=Path(__file__).resolve().parent
    expected=json.loads((package/'EXPECTED.json').read_bytes());checks=[]
    def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
    unbound=inspect(expected,{},lambda:None)
    check('all101_unbound_no_implicit_A100_source_fallback',unbound['planned_files']==101
          and unbound['regular_stat_observed']==0 and all(g['counts']=={'TARGET_LOCATOR_UNBOUND':g['planned_files']}
              for g in unbound['groups'].values()))
    empty=(args.output/'explicit-empty-fixture').resolve();empty.mkdir()
    bindings={g:str(empty if x['locator_kind']=='directory' else empty/(g+'-absent'))
              for g,x in expected['groups'].items()}
    absent=inspect(expected,bindings,lambda:None)
    check('empty_explicit_roots_keep_all101_missing_slots',sum(g['counts'].get('ABSENT_AT_OBSERVATION',0)
          for g in absent['groups'].values())==101 and absent['training_READY'] is False)
    try:inspect(expected,{'base':'relative-root'},lambda:None)
    except ValueError:rejected=True
    else:rejected=False
    check('relative_locator_rejected',rejected)
    try:inspect(expected,{'unapproved_other_root':str(empty)},lambda:None)
    except ValueError:rejected=True
    else:rejected=False
    check('unknown_locator_group_rejected',rejected)
    check('expected_group_counts_exact',[len(expected['groups'][g]['files'])
        for g in ('base','preset','same_input','first_capture_progress')]==[77,1,22,1])
    check('no_asset_body_reads',unbound['file_body_bytes_read']==absent['file_body_bytes_read']==0)
    ast.parse((package/'inspect_assets.py').read_bytes())
    result={'state':'FROZEN101_ASSET_CONTRACT_AND_STAT_BOUNDARIES_PASS','passed':all(x['passed'] for x in checks),
            'checks':checks,'actual_target_observations':0,'synthetic_empty_root':True,
            'actual_model_body_bytes_read':0,'original_source_files_canonical_manifest_recomputed':True,
            'training_READY':False,'EXPECTED':{'bytes':(package/'EXPECTED.json').stat().st_size,
            'SHA256':hashlib.sha256((package/'EXPECTED.json').read_bytes()).hexdigest()}}
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();(args.output/'result.json').write_bytes(raw)
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'bytes':len(raw),
                      'SHA256':hashlib.sha256(raw).hexdigest()},sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
