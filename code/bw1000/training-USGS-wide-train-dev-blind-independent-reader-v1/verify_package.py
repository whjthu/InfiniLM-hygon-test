"""Static exact manifest and no-producer-import check, never a data audit."""
import ast
import hashlib
import json
from pathlib import Path


def verify(prepared=False):
    d=Path(__file__).resolve().parent
    mf=json.loads((d/'MANIFEST.json').read_bytes())['files']
    assert {p.name for p in d.iterdir() if p.is_file()}==set(mf)|{'MANIFEST.json'}
    for name,pin in mf.items():
        assert Path(name).name==name and not (d/name).is_symlink()
        b=(d/name).read_bytes();assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
        if name.endswith('.py'):
            tree=ast.parse(b,filename=name)
            if name in ('independent_math.py','read_data.py'):
                forbidden={'saved_parser','select_groups','chart_contracts','build_confirmation','render','source_http'}
                for n in ast.walk(tree):
                    if isinstance(n,ast.Import):assert all(v.name.split('.')[0] not in forbidden for v in n.names)
                    if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in forbidden
    c=json.loads((d/'CLAIM.json').read_bytes());b=json.loads((d/'BINDING.json').read_bytes())
    assert all(type(c[k]) is bool and c[k] is (not prepared) for k in ('root_GO','READ_GO','CPU_GO'))
    assert all(c[k] is False for k in ('Dataset_GO','training_GO','GPU_GO','Model_GO'))
    assert c['max_invocations']==1 and c['whole_seconds']==b['whole_seconds']==120 and c['CPU']==2
    assert c['WindowsJob_commit_bytes']==2147483648 and c['GETs']==c['render_calls']==c['model_calls']==c['score_calls']==0
    proof=json.loads((d/'SOURCE_PROOF.json').read_bytes())
    for name,pin in proof['qualified_job_helpers'].items():
        assert mf[name]==pin
    return dict(passed=True,prepared=prepared,source_members=len(mf),data_reads=0,fixture_calls=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepared',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.prepared),sort_keys=True))
